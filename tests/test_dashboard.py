"""Check dashboard use on a fresh clone without downloaded data or fitted models."""
import json
from pathlib import Path
import shutil

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_shows_saved_results_without_models(tmp_path):
    reports = tmp_path / 'reports'
    (reports / 'test').mkdir(parents=True)
    for name in ('frozen.json', 'test/results.csv'):
        shutil.copyfile(ROOT / 'reports' / name, reports / name)
    frozen = json.loads((reports / 'frozen.json').read_text())
    assert not (tmp_path / 'models').exists()
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    app.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == '90.87%'
    assert app.metric[1].value == '0.9084'
    assert frozen['selected'].replace('_', ' ') in [item.value for item in app.markdown]
    assert any('Prediction needs the fitted model' in item.value for item in app.info)


def test_dashboard_explains_missing_experiment(tmp_path):
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    app.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=30)
    assert not app.exception
    assert any('Choose a completed experiment folder' in item.value for item in app.info)
