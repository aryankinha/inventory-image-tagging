"""Input, feature and checksum contracts for the exploration dashboard."""
from io import BytesIO
import json

import joblib
import numpy as np
from PIL import Image, ImageDraw
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
    source = Image.new('RGB', (80, 40), 'white')
    ImageDraw.Draw(source).rectangle((20, 5, 60, 35), fill='black')
    buffer = BytesIO()
    source.save(buffer, format='JPEG')
    content = buffer.getvalue()
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


def test_transparent_upload_uses_the_selected_background():
    from PIL import ImageDraw
    rgba = Image.new('RGBA', (28, 28), (255, 255, 255, 0))
    ImageDraw.Draw(rgba).rectangle((7, 7, 20, 20), fill=(255, 255, 255, 255))
    content = BytesIO()
    rgba.save(content, format='PNG')
    pixels, _ = prepare_image(content.getvalue(), experimental=True)
    assert pixels[0, 0] == 0 and pixels[10, 10] == 255
    light = Image.new('RGBA', (28, 28), (0, 0, 0, 0))
    ImageDraw.Draw(light).rectangle((7, 7, 20, 20), fill=(0, 0, 0, 255))
    content = BytesIO()
    light.save(content, format='PNG')
    inverted, _ = prepare_image(content.getvalue(), experimental=True, invert=True)
    assert np.array_equal(pixels, inverted)


@pytest.mark.parametrize('mode', ['L', 'RGB', 'RGBA'])
def test_blank_photo_is_rejected_without_changing_exact_input_contract(mode):
    with pytest.raises(ValueError, match='blank|visible detail'):
        prepare_image(image_bytes(mode), experimental=True)
    # Constant benchmark images still pass through unchanged, as the original contract requires.
    pixels, _ = prepare_image(image_bytes('L'))
    assert (pixels == 128).all()


def test_gallery_has_ten_checked_grayscale_categories():
    from dashboard.services import ROOT, sample_gallery
    samples = sample_gallery()
    assert len(samples) == 10
    assert sorted(s['label'] for s in samples) == list(range(10))
    for sample in samples:
        with Image.open(ROOT / 'assets/examples' / sample['filename']) as image:
            assert image.mode == 'L' and image.size == (28, 28)


def test_very_narrow_photo_still_has_valid_dimensions():
    image = Image.new('L', (1, 1000), 0)
    ImageDraw.Draw(image).rectangle((0, 400, 0, 600), fill=255)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    pixels, _ = prepare_image(buffer.getvalue(), experimental=True)
    assert pixels.shape == (28, 28) and pixels.max() > 0


def test_gallery_matches_original_training_ids_when_data_are_present():
    from src.inventory_tagging import data
    from dashboard.services import sample_gallery
    if not (data.ROOT / 'data/split.npz').exists():
        pytest.skip('Training archives are optional on a fresh clone.')
    images, labels = data.load_images('train')
    train, _ = data.split_indices(labels)
    for sample in sample_gallery():
        index = sample['original_training_index']
        assert index in train and labels[index] == sample['label']
        with Image.open(data.ROOT / 'assets/examples' / sample['filename']) as image:
            assert np.array_equal(np.asarray(image), images[index])
