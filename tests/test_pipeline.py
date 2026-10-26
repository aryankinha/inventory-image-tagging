"""Scientific checks: split isolation, descriptors, training-only scaling, saved evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.inventory_tagging import data
from src.inventory_tagging.features import extract, normalized
from src.inventory_tagging.models import build


def test_split_is_balanced_disjoint_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    labels = np.repeat(np.arange(10), 6000)
    train, val = data.split_indices(labels)
    again = data.split_indices(labels)
    assert len(train) == 48000 and len(val) == 12000
    assert not np.intersect1d(train, val).size
    assert np.all(np.bincount(labels[train]) == 4800)
    assert np.all(np.bincount(labels[val]) == 1200)
    assert np.array_equal(np.sort(np.r_[train, val]), np.arange(60000))
    assert np.array_equal(train, again[0]) and np.array_equal(val, again[1])


def test_changed_split_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    (tmp_path / 'data').mkdir()
    np.savez(tmp_path / 'data/split.npz', train=np.arange(48000), validation=np.arange(48000, 60000))
    with pytest.raises(ValueError, match='differs'):
        data.split_indices(np.tile(np.arange(10), 6000))


def test_pixel_values_and_orientation_survive_round_trip():
    images = np.random.default_rng(4).integers(0, 256, (3, 28, 28), dtype=np.uint8)
    pixels = extract(images, 'pixels')
    assert pixels.dtype == np.float32 and pixels.shape == (3, 784)
    assert np.array_equal(np.rint(pixels.reshape(3, 28, 28) * 255).astype('uint8'), images)
    with pytest.raises(TypeError):
        normalized(pixels)


@pytest.mark.parametrize('representation,dimension', [('hog', 324)])
def test_hog_detects_a_real_edge(representation, dimension):
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[1, :, 14:] = 255
    descriptors = extract(images, representation)
    assert descriptors.shape == (2, dimension)
    assert np.isfinite(descriptors).all()
    assert np.linalg.norm(descriptors[0]) == 0
    assert np.linalg.norm(descriptors[1]) > 0


def test_scaler_does_not_learn_from_validation():
    train = np.array([[0, 0], [1, 2], [2, 1], [3, 3]], dtype=float)
    model = build('logistic_regression', {'C': 1})
    model.fit(train, [0, 0, 1, 1])
    before = model[0].mean_.copy()
    model.predict(np.array([[1000, -1000]], dtype=float))
    assert np.allclose(before, train.mean(axis=0))
    assert np.array_equal(model[0].mean_, before)


def test_all_saved_validation_scores_are_recomputable():
    files = list((data.ROOT / 'reports/training').glob('*_validation.npz'))
    if not files:
        pytest.skip('Run training to test generated validation evidence')
    for path in files:
        meta = json.loads(path.with_name(path.name.replace('_validation.npz', '.json')).read_text())
        with np.load(path, allow_pickle=False) as saved:
            assert len(saved['indices']) == 12000
            assert len(np.unique(saved['indices'])) == 12000
            assert np.isclose(accuracy_score(saved['truth'], saved['prediction']), meta['accuracy'])
            assert np.isclose(f1_score(saved['truth'], saved['prediction'], average='macro'), meta['macro_f1'])
        assert meta['training_images'] == 48000
        model = data.ROOT / 'models' / f"{meta['name']}.joblib"
        if model.exists():
            assert data.sha256(model) == meta['model_sha256']


def test_final_metrics_and_confusions_match_predictions():
    folder = data.ROOT / 'reports/test'
    if not (folder / 'summary.json').exists():
        pytest.skip('Final evaluation has not been performed')
    table = pd.read_csv(folder / 'results.csv')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['indices'], np.arange(10000))
        assert np.all(np.bincount(saved['truth']) == 1000)
        for row in table.itertuples():
            prediction = saved[row.name]
            assert np.isclose(accuracy_score(saved['truth'], prediction), row.test_accuracy)
            assert np.isclose(f1_score(saved['truth'], prediction, average='macro'), row.test_macro_f1)
            recorded = pd.read_csv(folder / f'{row.name}_confusion.csv', index_col=0).to_numpy()
            assert np.array_equal(confusion_matrix(saved['truth'], prediction), recorded)
    frozen = json.loads((data.ROOT / 'reports/frozen.json').read_text())
    validation = pd.read_csv(data.ROOT / 'reports/validation_results.csv')
    assert frozen['selected'] == validation.iloc[0]['name']


def test_frozen_training_guard(monkeypatch, tmp_path):
    from src.inventory_tagging import train
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports/frozen.json').write_text('{}')
    with pytest.raises(FileExistsError, match='frozen'):
        train.run()


def test_training_uses_all_shared_ids_and_never_requests_test_data(monkeypatch, tmp_path):
    from src.inventory_tagging import train
    calls = []
    # No fitting is needed to check the core orchestration and data contract.
    images = np.zeros((60000, 1, 1), dtype=np.uint8)
    labels = np.tile(np.arange(10), 6000)
    ids = np.arange(48000)
    val = np.arange(48000, 60000)
    def load(split):
        assert split == 'train'
        return images, labels
    def record(representation, family, x, y, fit_ids, validation_ids):
        assert np.array_equal(fit_ids, ids)
        assert np.array_equal(validation_ids, val)
        calls.append((representation, family))
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    monkeypatch.setattr(train, 'load_images', load)
    monkeypatch.setattr(train, 'split_indices', lambda y: (ids, val))
    monkeypatch.setattr(train, 'fit_one', record)
    monkeypatch.setattr(train, 'summarize', lambda: None)
    train.run()
    assert len(calls) == 8 and len(set(calls)) == 8


def test_phase1_contains_exactly_the_required_eight_combinations():
    from src.inventory_tagging.train import FAMILIES
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / 'reports/frozen.json').read_text())
    required = {f'{rep}_{family}' for rep in ('pixels', 'hog') for family in FAMILIES}
    assert {r['name'] for r in frozen['models']} == required
    assert len(frozen['models']) == 8
    validation = pd.read_csv(root / 'reports/validation_results.csv')
    assert frozen['selected'] == validation.sort_values(['macro_f1', 'accuracy'], ascending=False).iloc[0]['name']
    for representation in ('unknown',):
        with pytest.raises(ValueError):
            extract(np.zeros((1, 28, 28), dtype=np.uint8), representation)
