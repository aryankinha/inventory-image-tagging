"""Upload safety and optional CNN scores without changing the benchmark pipeline."""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dashboard.services import prepare_image

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('isolate', [False, True])
def test_high_bit_depth_upload_is_rejected_before_conversion(isolate):
    # Pillow's implicit conversion clips values above 255, losing useful detail.
    image = Image.fromarray(np.arange(1600, dtype=np.uint16).reshape(40, 40))
    content = BytesIO()
    image.save(content, format='PNG')
    with pytest.raises(ValueError, match='8-bit PNG or JPEG'):
        prepare_image(content.getvalue(), experimental=True, isolate=isolate)


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
def test_invalid_cnn_scores_do_not_become_a_category(bad_value):
    torch = pytest.importorskip('torch')
    from src.inventory_tagging_dl.predict import predict

    class InvalidModel:
        embedding = torch.nn.Identity()

        def head(self, features):
            return torch.full((len(features), 10), bad_value)

    with pytest.raises(ValueError, match='invalid scores'):
        predict(np.zeros((1, 28, 28), dtype=np.uint8), (InvalidModel(), None))


def test_neural_page_shows_actual_ranked_scores_and_keeps_them_on_rerun():
    pytest.importorskip('torch')
    from streamlit.testing.v1 import AppTest
    from src.inventory_tagging.data import CLASSES

    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=40)
    app.sidebar.radio[0].set_value('Phase 2').run()
    assert not any(item.label == 'Other possible categories' for item in app.expander)
    app.button(key='p2_predict').click().run(timeout=40)
    assert not app.exception and not app.error
    result = app.session_state['p2_prediction_run'][1]
    order = np.argsort(-np.asarray(result['probabilities'][0]), kind='stable')[:3]
    table = app.dataframe[0].value
    assert list(table.Category) == [CLASSES[i] for i in order]
    np.testing.assert_allclose(table['Model score'], np.asarray(result['probabilities'][0])[order])
    assert any('not calibrated confidence' in item.value for item in app.caption)
    app.run()
    assert not app.exception
    assert list(app.dataframe[0].value.Category) == list(table.Category)


def test_hybrid_has_no_invented_probability_scores():
    torch = pytest.importorskip('torch')
    from src.inventory_tagging_dl.predict import predict

    class Head:
        def predict(self, features):
            return np.full(len(features), 6)

    class Model:
        embedding = torch.nn.Identity()

    result = predict(np.zeros((1, 28, 28), dtype=np.uint8), (Model(), Head()))
    assert result['categories'] == ['Shirt']
    assert result['probabilities'] is None
