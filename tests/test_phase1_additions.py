"""New analysis, evidence protection and setup safeguards for Phase 1."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys



ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_integrity_manifest_requires_only_git_versioned_files():
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    manifest = json.loads((ROOT / 'reports/artifact_hashes.json').read_text())
    assert set(manifest['files']).issubset(tracked)


def test_reproduction_keeps_an_interrupted_final_evaluation(tmp_path):
    folder = tmp_path / 'run'
    marker = folder / 'reports/test/started.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{"started": true}\n')
    before = marker.read_bytes()
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce.py'), '--work-dir', str(folder)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'already started' in result.stderr
    assert marker.read_bytes() == before
    assert list(folder.rglob('*')) == [folder / 'reports', folder / 'reports/test', marker]
