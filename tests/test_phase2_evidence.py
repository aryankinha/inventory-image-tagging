"""Re-check the saved Phase 2 evidence without retraining."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/phase2'


def load(name):
    return json.loads((OUT / name).read_text())


def test_choices_were_frozen_before_the_test_and_match_the_selection_files():
    frozen, selection, hybrid = load('frozen.json'), load('selection.json'), load('hybrid_selection.json')
    started = load('test/started.json')
    assert frozen['cnn_config'] == selection['config']
    assert (frozen['hybrid_family'], frozen['hybrid_parameters']) == (hybrid['family'], hybrid['parameters'])
    import hashlib
    assert started['frozen_sha256'] == hashlib.sha256((OUT / 'frozen.json').read_bytes()).hexdigest()
    assert selection['config'] != 'plain' and frozen['cnn_settings']['dropout'] > 0   # the brief asks for dropout


def test_selection_used_validation_numbers_only():
    table = pd.read_csv(OUT / 'ablation_summary.csv')
    assert (table.runs == 3).all() and len(table) == 7
    chosen = load('selection.json')
    with_dropout = {'full', 'no_batch_norm', 'light_dropout', 'no_augmentation', 'three_blocks'}
    best = table[table.config.isin(with_dropout)].sort_values('validation_macro_f1_mean').iloc[-1]
    assert chosen['config'] == best.config


def test_reported_scores_are_recomputable_from_saved_predictions():
    predictions = np.load(OUT / 'test/predictions.npz')
    truth = predictions['truth']
    assert len(truth) == 10000 and np.array_equal(truth, np.load(ROOT / 'reports/test/predictions.npz')['truth'])
    for _, row in pd.read_csv(OUT / 'test/results.csv').iterrows():
        predicted = predictions[f"{row.model}_seed{int(row.seed)}"]
        assert accuracy_score(truth, predicted) == pytest.approx(row.test_accuracy)
        assert f1_score(truth, predicted, average='macro') == pytest.approx(row.test_macro_f1)
    summary = load('test/summary.json')
    results = pd.read_csv(OUT / 'test/results.csv')
    assert results[results.model == 'cnn'].test_accuracy.mean() == pytest.approx(summary['cnn_test_accuracy_mean'])


def test_committed_checkpoints_match_the_frozen_hashes():
    import hashlib
    frozen = load('frozen.json')
    for name, expected in frozen['checkpoint_sha256'].items():
        assert hashlib.sha256((OUT / 'checkpoints' / f'{name}.pt').read_bytes()).hexdigest() == expected


def test_committed_checkpoint_reproduces_saved_test_predictions():
    pytest.importorskip('torch')
    if not (ROOT / 'data/raw/t10k-images-idx3-ubyte.gz').exists():
        pytest.skip('Test archives are optional on a fresh clone.')
    import torch
    from src.inventory_tagging.data import load_images
    from src.inventory_tagging_dl import cnn, experiments
    frozen = load('frozen.json')
    images, _ = load_images('test')
    model = experiments.load_cnn(frozen['cnn_config'], 42, folder=OUT / 'checkpoints')
    logits, _ = cnn.forward_all(model, cnn.to_tensor(images), torch.device('cpu'))
    saved = np.load(OUT / 'test/predictions.npz')['cnn_seed42']
    assert (logits.argmax(1).numpy() == saved).mean() > 0.999     # the GPU and CPU may differ in a last digit
