"""Input, feature and checksum contracts for the exploration dashboard."""
from io import BytesIO
import json

import joblib
import numpy as np
from PIL import Image
import pytest
from sklearn.dummy import DummyClassifier

from dashboard.services import (confusion_table, feature_view, load_model, local_dataset,
                                prepare_image, saved_predictions, stamp)
from src.inventory_tagging.data import sha256
from src.inventory_tagging.features import extract


def image_bytes(mode='L', size=(28, 28), fmt='PNG'):
    buffer = BytesIO()
    Image.new(mode, size, color=128).save(buffer, format=fmt)
    return buffer.getvalue()


def test_benchmark_input_is_unchanged_and_invalid_inputs_explained():
    pixels, _ = prepare_image(image_bytes())
    assert pixels.dtype == np.uint8 and pixels.shape == (28, 28)
    assert (pixels == 128).all()
    for content in (image_bytes('RGB'), image_bytes(size=(40, 28)), image_bytes(fmt='JPEG')):
        with pytest.raises(ValueError, match='Benchmark input'):
            prepare_image(content)
    with pytest.raises(OSError):
        prepare_image(b'not an image')


def test_photo_preparation_is_deterministic_and_explicit():
    content = image_bytes('RGB', (80, 40), 'JPEG')
    a, steps = prepare_image(content, True, 'Pad', True, 1., 1.)
    b, _ = prepare_image(content, True, 'Pad', True, 1., 1.)
    assert np.array_equal(a, b) and a.shape == (28, 28) and a.dtype == np.uint8
    assert len(steps) == 4
    assert a[0, 0] == 0  # light padding is inverted to the dark benchmark background


@pytest.mark.parametrize('representation', ['pixels', 'hog', 'hog_fine'])
def test_visualized_feature_vector_is_the_real_extractor(representation):
    pixels = np.random.default_rng(2).integers(0, 256, (28, 28), dtype=np.uint8)
    values, drawing = feature_view(pixels, representation)
    assert np.array_equal(values, extract(pixels[None], representation)[0])
    assert drawing.shape == (28, 28)


def test_checksum_is_checked_before_unpickling_and_after_file_replacement(tmp_path):
    path = tmp_path / 'model.joblib'
    joblib.dump(DummyClassifier().fit([[0], [1]], [0, 1]), path)
    expected = sha256(path)
    assert load_model(path, expected) is not None
    path.write_bytes(b'damaged artifact')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_model(path, expected)


def test_invalid_confusion_and_prediction_arrays_are_rejected(tmp_path):
    path = tmp_path / 'matrix.csv'
    path.write_text(',not a category\nwrong,1\n')
    with pytest.raises(ValueError, match='class order'):
        confusion_table(path)
    path = tmp_path / 'predictions.npz'
    np.savez(path, truth=np.array([0, 1]), broken=np.array([0]), indices=np.array([0, 1]))
    with pytest.raises(ValueError, match='incompatible shapes'):
        saved_predictions(stamp(path), 'broken')


def test_dataset_adapter_does_not_mutate_shared_root_or_download(tmp_path):
    from src.inventory_tagging import data
    original = data.ROOT
    with pytest.raises(FileNotFoundError):
        local_dataset(str(tmp_path), 'train', ())
    assert data.ROOT == original
    assert not (tmp_path / 'data').exists()
