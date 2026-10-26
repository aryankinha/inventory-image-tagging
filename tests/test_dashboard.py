"""Exercise saved-result navigation, prediction and fresh-clone fallbacks."""
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from streamlit.testing.v1 import AppTest

from src.inventory_tagging.data import sha256
from src.inventory_tagging.features import extract
from dashboard.views import model_label

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment_folder(tmp_path):
    reports = tmp_path / 'reports'
    (reports / 'test').mkdir(parents=True)
    for name in ('frozen.json', 'test/results.csv'):
        shutil.copyfile(ROOT / 'reports' / name, reports / name)
    return tmp_path


def open_app(folder):
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    app.sidebar.text_input[0].set_value(str(folder)).run(timeout=30)
    return app


def test_dashboard_shows_saved_results_without_models(experiment_folder):
    app = open_app(experiment_folder)
    assert not app.exception
    assert app.sidebar.radio[0].value == 'Prediction Lab'
    assert app.file_uploader[0].label == 'Upload a clothing picture'
    assert app.button[0].label == 'Predict'
    app.sidebar.radio[0].set_value('About').run()
    assert app.metric[0].value == '90.87%'
    assert app.metric[1].value == '0.9084'
    app.sidebar.radio[0].set_value('Prediction Lab').run()
    assert not app.exception
    assert any('Prediction needs the saved model' in item.value for item in app.info)


def test_dashboard_explains_missing_experiment(tmp_path):
    app = open_app(tmp_path)
    assert not app.exception
    assert any('Choose a completed experiment folder' in item.value for item in app.info)


def test_all_sections_and_missing_optional_artifacts(experiment_folder):
    app = open_app(experiment_folder)
    for section in ('Model Comparison', 'Error Analysis', 'About', 'Prediction Lab'):
        app.sidebar.radio[0].set_value(section).run(timeout=30)
        assert not app.exception, section
        assert not app.error, section
    assert app.sidebar.radio[0].value == 'Prediction Lab'
    app.sidebar.radio[0].set_value('About').run()
    app.checkbox[0].check().run()
    assert not app.exception


def test_comparison_filters_update_table(experiment_folder):
    app = open_app(experiment_folder)
    app.sidebar.radio[0].set_value('Model Comparison').run()
    app.multiselect[0].set_value(['hog']).run()
    app.multiselect[1].set_value(['knn']).run()
    app.selectbox[0].set_value('model_bytes').run()
    assert not app.exception
    assert list(app.dataframe[0].value['Model']) == ['Edge directions + Closest examples']
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any('at least one matching' in item.value for item in app.info)


def test_prediction_selects_and_compares_saved_models(experiment_folder):
    # Tiny synthetic fixtures test UI routing; they are never written to real reports/models.
    frozen_path = experiment_folder / 'reports/frozen.json'
    frozen = json.loads(frozen_path.read_text())
    names = ['pixels_logistic_regression', 'hog_linear_svm']
    images = np.zeros((10, 28, 28), dtype=np.uint8)
    (experiment_folder / 'models').mkdir()
    for label, name in enumerate(names):
        record = next(r for r in frozen['models'] if r['name'] == name)
        model = DummyClassifier(strategy='constant', constant=label).fit(extract(images, record['representation']), np.arange(10))
        path = experiment_folder / 'models' / (name + '.joblib')
        joblib.dump(model, path)
        record['model_sha256'] = sha256(path)
    frozen_path.write_text(json.dumps(frozen))
    app = open_app(experiment_folder)
    app.sidebar.radio[0].set_value('Prediction Lab').run()
    app.selectbox(key='predictor').set_value(names[0]).run()
    app.multiselect(key='compare_predictors').set_value([names[1]]).run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    table = app.dataframe[0].value
    assert list(table.Model) == [model_label(n, {r['name']:r for r in frozen['models']}) for n in names]
    assert list(table['Predicted category']) == ['T-shirt/top', 'Trouser']
    assert (table['Prediction time (ms)'] > 0).all()
    app.selectbox(key='predictor').set_value(names[1]).run()
    app.multiselect(key='compare_predictors').set_value([]).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]['Predicted category'] == 'Trouser'
    app.radio(key='input_mode').set_value('Prepare an ordinary picture').run()
    assert not app.exception
    assert any('have not been tested for accuracy' in item.value for item in app.warning)
    app.sidebar.radio[0].set_value('About').run()
    app.sidebar.radio[0].set_value('Prediction Lab').run()
    assert app.selectbox(key='predictor').value == names[1]
    assert app.radio(key='input_mode').value == 'Prepare an ordinary picture'


def test_error_filters_use_saved_predictions(experiment_folder):
    reports = experiment_folder / 'reports/test'
    for name in ('predictions.npz', 'hog_fine_rbf_svm_confusion.csv', 'per_class.csv'):
        shutil.copyfile(ROOT / 'reports/test' / name, reports / name)
    app = open_app(experiment_folder)
    app.sidebar.radio[0].set_value('Error Analysis').run(timeout=30)
    app.selectbox(key='error_true').set_value('Shirt').run()
    app.selectbox(key='error_predicted').set_value('T-shirt/top').run()
    assert not app.exception
    matches = app.dataframe[-1].value
    assert len(matches) > 0
    assert (matches['Actual category'] == 'Shirt').all()
    assert (matches['Model guessed'] == 'T-shirt/top').all()
    assert not matches['Correct answer'].any()


def test_incomplete_final_results_are_explained(experiment_folder):
    path = experiment_folder / 'reports/test/results.csv'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines[:2]) + '\n')
    app = open_app(experiment_folder)
    assert not app.exception
    assert any('not saved all of its final scores' in item.value for item in app.error)
    assert not app.metric
