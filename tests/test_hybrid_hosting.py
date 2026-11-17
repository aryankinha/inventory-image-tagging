"""Hosted hybrid downloads must preserve the existing verified inference path."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil

import pytest

from dashboard import hosting, hybrid_hosting

ROOT = Path(__file__).resolve().parents[1]


def publication(tmp_path, monkeypatch, content=b'published forest'):
    monkeypatch.setattr(hybrid_hosting, 'ROOT', tmp_path)
    folder = tmp_path / 'reports/phase2'
    folder.mkdir(parents=True)
    (folder / 'frozen.json').write_text('{}')
    (tmp_path / 'assets').mkdir()
    record = {'frozen_sha256': hashlib.sha256(b'{}').hexdigest(),
              'hybrids': {'42': {'file': 'hybrid_seed42.joblib',
                                'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}}}
    (tmp_path / 'assets/hosted_hybrid.json').write_text(json.dumps(record))
    return folder, record


def test_hybrid_download_verified_and_reused(tmp_path, monkeypatch):
    folder, record = publication(tmp_path, monkeypatch)
    calls = []
    def download(url, timeout):
        calls.append(url)
        return BytesIO(b'published forest')
    monkeypatch.setattr(hosting, 'urlopen', download)
    assert hybrid_hosting.ensure_hybrid(folder) == record
    hybrid_hosting.ensure_hybrid(folder)
    assert len(calls) == 1
    assert json.loads((folder / 'inference.json').read_text()) == record


@pytest.mark.parametrize('content', [b'corrupt', b'x' * 100])
def test_corrupt_download_leaves_no_classifier_or_record(tmp_path, monkeypatch, content):
    folder, _ = publication(tmp_path, monkeypatch)
    monkeypatch.setattr(hosting, 'urlopen', lambda *a, **k: BytesIO(content))
    with pytest.raises(ValueError, match='checksum|larger'):
        hybrid_hosting.ensure_hybrid(folder)
    assert not (folder / 'inference.json').exists()
    assert list((folder / 'checkpoints').iterdir()) == []


def test_published_hybrid_never_substitutes_another_experiment(tmp_path, monkeypatch):
    folder, _ = publication(tmp_path, monkeypatch)
    assert hybrid_hosting.published_hybrid(tmp_path / 'runs/reproduction/reports/phase2') is None
    (folder / 'frozen.json').write_text('{"changed":true}')
    assert hybrid_hosting.published_hybrid(folder) is None


def test_single_seed_download_does_not_block_full_local_export(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from src.inventory_tagging_dl import predict as inference, experiments
    for name in ['frozen.json', 'hybrid_selection.json']:
        shutil.copyfile(ROOT / 'reports/phase2' / name, tmp_path / name)
    record = json.loads((ROOT / 'assets/hosted_hybrid.json').read_text())
    (tmp_path / 'inference.json').write_text(json.dumps(record))
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/hybrid_seed42.joblib').write_bytes(b'present single-seed export')
    frozen = json.loads((tmp_path / 'frozen.json').read_text())
    original_hash = inference.sha256
    monkeypatch.setattr(inference, 'sha256', lambda p: frozen['split_sha256'] if Path(p).name == 'split.npz' else original_hash(p))
    def training_data():
        raise RuntimeError('full export reached training data')
    monkeypatch.setattr(experiments, 'training_data', training_data)
    with pytest.raises(RuntimeError, match='full export reached'):
        inference.export_hybrids(tmp_path)


def test_fresh_install_can_select_download_and_predict_both_models(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(hybrid_hosting, 'ROOT', tmp_path)
    for relative in ['reports/frozen.json', 'reports/test/results.csv',
                     'reports/phase2/frozen.json', 'assets/hosted_hybrid.json']:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    frozen = json.loads((ROOT / 'reports/phase2/frozen.json').read_text())
    name = frozen['cnn_config'] + '_seed42.pt'
    checkpoints = tmp_path / 'reports/phase2/checkpoints'
    checkpoints.mkdir()
    shutil.copyfile(ROOT / 'reports/phase2/checkpoints' / name, checkpoints / name)
    calls = []
    def download(url, timeout):
        calls.append(url)
        return (ROOT / 'reports/phase2/checkpoints/hybrid_seed42.joblib').open('rb')
    monkeypatch.setattr(hosting, 'urlopen', download)
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=40)
    app.sidebar.text_input[0].set_value(str(tmp_path)).run()
    app.sidebar.radio[0].set_value('Phase 2').run()
    assert app.selectbox(key='p2_model').options == ['CNN', 'CNN + Random Forest']
    assert not calls  # Opening the CNN page never downloads the hybrid.
    app.button(key='p2_predict').click().run(timeout=40)
    assert app.success[0].value == 'Predicted category: Shirt'
    app.selectbox(key='p2_model').set_value('hybrid').run(timeout=40)
    assert not app.success  # Switching models clears the old displayed answer.
    app.button(key='p2_predict').click().run(timeout=40)
    assert not app.error and not app.exception
    assert app.success[0].value == 'Predicted category: Shirt'
    assert len(calls) == 1
    app.run()
    assert len(calls) == 1
    assert app.success[0].value == 'Predicted category: Shirt'
