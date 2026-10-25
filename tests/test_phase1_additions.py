"""New analysis, evidence protection and setup safeguards for Phase 1."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from dashboard.analysis import paired_outcomes

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_comparison_is_reproducible_and_directional():
    truth = np.array([0, 1, 2, 3])
    a, b = np.array([0, 1, 0, 0]), np.array([0, 0, 2, 0])
    report = paired_outcomes(truth, a, b)
    assert [report[k] for k in ('both_right', 'first_only_right', 'second_only_right', 'both_wrong')] == [1, 1, 1, 1]
    assert report == paired_outcomes(truth, a, b)
    same = paired_outcomes(truth, a, a)
    assert same['accuracy_difference'] == 0 and same['interval_95'] == [0, 0]
    winner = paired_outcomes(truth, truth, np.array([1, 2, 3, 4]))
    assert winner['accuracy_difference'] == 1 and winner['interval_95'] == [1, 1]
    with pytest.raises(ValueError):
        paired_outcomes(truth, a[:2], b)


def test_recorded_extra_analysis_matches_its_saved_input():
    comparison = script('compare_saved_models').compare(ROOT)
    recorded = json.loads((ROOT / 'reports/additional_analysis/paired_comparison.json').read_text())
    assert recorded == comparison
    assert comparison['images'] == 10000
    assert sum(comparison[k] for k in ('both_right', 'first_only_right', 'second_only_right', 'both_wrong')) == 10000


def test_integrity_verifier_rejects_a_changed_or_missing_sample(tmp_path):
    from src.inventory_tagging.data import sha256
    (tmp_path / 'reports').mkdir()
    path = tmp_path / 'picture.png'
    path.write_bytes(b'original fixture')
    manifest = {'files': {'picture.png': sha256(path)}}
    (tmp_path / 'reports/artifact_hashes.json').write_text(json.dumps(manifest))
    verify = script('verify_artifacts')
    assert verify.check(tmp_path)['status'] == 'passed'
    path.write_bytes(b'changed fixture')
    assert verify.check(tmp_path)['status'] == 'failed'
    path.unlink()
    assert verify.check(tmp_path)['failures'] == ['Missing: picture.png']


def test_committed_evidence_and_additional_gallery_verify():
    assert script('verify_artifacts').check(ROOT)['status'] == 'passed'
    assert script('verify_results').check_results(ROOT)['status'] == 'passed'


def test_reproduction_refuses_to_overwrite_the_original_project():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce.py'), '--work-dir', str(ROOT)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'cannot be overwritten' in result.stderr


def test_no_models_still_allows_gallery_and_previews():
    from streamlit.testing.v1 import AppTest
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        (folder / 'reports/test').mkdir(parents=True)
        for name in ('frozen.json', 'test/results.csv'):
            shutil.copyfile(ROOT / 'reports' / name, folder / 'reports' / name)
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        app.sidebar.text_input[0].set_value(str(folder)).run()
        app.sidebar.radio[0].set_value('Prediction Lab').run()
        assert not app.exception and not app.error
        assert len(app.selectbox(key='sample_category').options) == 10
        assert app.button[0].disabled
