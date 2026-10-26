"""Dashboard adapters. The frozen training/evaluation pipeline stays untouched."""
from io import BytesIO
from pathlib import Path
import hashlib
import importlib.util
import json
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from skimage.feature import hog
from skimage.exposure import rescale_intensity
import streamlit as st
from threadpoolctl import threadpool_limits

from .hosting import ensure_model, model_available

from src.inventory_tagging import data
from src.inventory_tagging.data import CLASSES, sha256
from src.inventory_tagging.features import HOG, extract, normalized

ROOT = Path(__file__).resolve().parents[1]


def stamp(path):
    """Include file identity in cache keys so replacements cannot reuse stale entries."""
    path = Path(path)
    info = path.stat()
    return str(path.resolve()), info.st_size, info.st_mtime_ns


@st.cache_data(max_entries=32, show_spinner=False)
def read_csv(identity, index_col=None):
    return pd.read_csv(identity[0], index_col=index_col)


@st.cache_data(max_entries=32, show_spinner=False)
def read_json(identity):
    return json.loads(Path(identity[0]).read_text())


def experiment(work_dir):
    reports = work_dir / 'reports'
    results = read_csv(stamp(reports / 'test/results.csv'))
    frozen = read_json(stamp(reports / 'frozen.json'))
    required = {'name', 'representation', 'family', 'test_accuracy', 'test_macro_f1', 'train_images'}
    if not required.issubset(results.columns) or results.name.duplicated().any():
        raise ValueError('The results table is missing required columns or has duplicate model names.')
    records = {r['name']: r for r in frozen['models']}
    if set(results.name) != set(records):
        raise ValueError('This run has not saved all of its final scores yet. Wait for the model-building command to finish, or choose a completed results folder.')
    if frozen['selected'] not in records or frozen['selected'] not in set(results.name):
        raise ValueError('The validation-selected model is missing from the experiment records.')
    # Model names are filenames, never arbitrary paths from a manifest.
    for name, record in records.items():
        if Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Invalid model name in the frozen records.')
        if record['representation'] not in ('pixels', 'hog'):
            raise ValueError('Unsupported feature representation: ' + record['representation'])
        row = results[results.name == name]
        if not row.empty and (row.iloc[0].representation != record['representation'] or row.iloc[0].family != record['family']):
            raise ValueError('The results and frozen model records disagree: ' + name)
    return results, frozen, records


@st.cache_resource(max_entries=4, show_spinner=False)
def _verified_model(identity, expected_hash):
    path = identity[0]
    if sha256(path) != expected_hash:
        raise ValueError('The saved model does not match the frozen experiment (SHA-256 checksum mismatch).')
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError('The model changed while being read. Restore the frozen artifact.')
    return joblib.load(BytesIO(content))


def load_model(path, expected_hash):
    return _verified_model(stamp(path), expected_hash)


def prepare_image(content, experimental=False, sizing='Pad', invert=False, brightness=1., contrast=1.):
    """Benchmark pixels pass through exactly. Photo preparation is explicit and deterministic."""
    with Image.open(BytesIO(content)) as opened:
        if opened.format not in ('PNG', 'JPEG'):
            raise ValueError('Choose a PNG or JPEG image.')
        if opened.width * opened.height > 20_000_000:
            raise ValueError('Use an image with at most 20 million pixels.')
        if not experimental:
            if opened.format != 'PNG' or opened.size != (28, 28) or opened.mode != 'L':
                raise ValueError('Dataset image mode needs a 28×28 grayscale PNG. To convert other pictures, switch to Prepare an ordinary picture.')
            image = opened.copy()
            return np.asarray(image, dtype=np.uint8).copy(), [('Benchmark input (unchanged)', image)]
        oriented = ImageOps.exif_transpose(opened)
        has_alpha = oriented.mode in ('RGBA', 'LA') or (oriented.mode == 'P' and 'transparency' in oriented.info)
        if has_alpha:
            fill = 255 if invert else 0
            background = Image.new('RGBA', oriented.size, (fill, fill, fill, 255))
            oriented = Image.alpha_composite(background, oriented.convert('RGBA'))
        original = oriented.convert('RGB')
    gray = ImageOps.grayscale(original)
    if gray.getextrema()[0] == gray.getextrema()[1]:
        raise ValueError('This picture is blank or has no visible detail. Choose a picture with a visible clothing item.')
    if sizing == 'Crop':
        sized = ImageOps.fit(gray, (28, 28), method=Image.Resampling.LANCZOS)
    elif sizing == 'Stretch':
        sized = gray.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        # Clamp each side to one pixel, including extremely narrow uploads.
        ratio = min(28 / gray.width, 28 / gray.height)
        size = tuple(max(1, min(28, round(side * ratio))) for side in gray.size)
        reduced = gray.resize(size, Image.Resampling.LANCZOS)
        sized = Image.new('L', (28, 28), 255 if invert else 0)
        sized.paste(reduced, ((28 - size[0]) // 2, (28 - size[1]) // 2))
    adjusted = ImageEnhance.Contrast(ImageEnhance.Brightness(sized).enhance(brightness)).enhance(contrast)
    if invert:
        adjusted = ImageOps.invert(adjusted)
    if adjusted.getextrema()[0] == adjusted.getextrema()[1]:
        raise ValueError('The prepared picture has no visible detail. Try different picture settings or another image.')
    return np.asarray(adjusted, dtype=np.uint8).copy(), [
        ('Original (transparent areas filled)' if has_alpha else 'Original', original), ('Shades of gray', gray), ('Resized to 28×28', sized),
        ('Prepared input', adjusted)]


def sample_gallery():
    """Small, checked training examples work even when the full dataset is absent."""
    path = ROOT / 'assets/examples/manifest.json'
    manifest = read_json(stamp(path))
    for record in manifest['samples']:
        name = record['filename']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid sample filename.')
        sample = path.parent / name
        if sha256(sample) != record['sha256']:
            raise ValueError('A sample picture has changed. Restore the original sample files.')
    return manifest['samples']


@st.cache_data(max_entries=32, show_spinner=False)
def feature_view(pixels, representation):
    values = extract(pixels[None], representation)[0]
    if representation == 'pixels':
        return values, normalized(pixels)
    settings = dict(HOG)
    _, drawing = hog(normalized(pixels), visualize=True, **settings)
    return values, rescale_intensity(drawing, in_range=(0, .1))


def model_identity(work_dir, record):
    path = work_dir / 'models' / (record['name'] + '.joblib')
    return stamp(path) if path.is_file() else ('published', record['model_sha256'])


def predict_one(work_dir, record, pixels):
    model = load_model(ensure_model(work_dir, record), record['model_sha256'])
    with threadpool_limits(limits=2):
        start = time.perf_counter()
        values = extract(pixels[None], record['representation'])
        label = int(model.predict(values)[0])
        elapsed = (time.perf_counter() - start) * 1000
        alternatives, kind = None, None
        if callable(getattr(model, 'predict_proba', None)):
            alternatives = model.predict_proba(values)[0]
            kind = 'Model probability (not independently calibrated)'
        elif callable(getattr(model, 'decision_function', None)) and getattr(model, 'decision_function_shape', 'ovr') != 'ovo':
            alternatives = model.decision_function(values)[0]
            kind = 'Decision score (not a probability)'
        if alternatives is not None and np.asarray(alternatives).shape == (len(model.classes_),):
            alternatives = pd.DataFrame({'Category': [CLASSES[int(c)] for c in model.classes_],
                                         'Value': alternatives}).sort_values('Value', ascending=False)
        else:
            alternatives, kind = None, None
    if not 0 <= label < len(CLASSES):
        raise ValueError('The model returned an invalid category.')
    return {'Model': record['name'], 'Prediction': CLASSES[label], 'Inference ms': elapsed,
            'alternatives': alternatives, 'kind': kind}


@st.cache_data(max_entries=16, show_spinner=False)
def saved_predictions(identity, name):
    with np.load(identity[0], allow_pickle=False) as saved:
        truth, prediction, indices = saved['truth'], saved[name], saved['indices']
    if truth.ndim != 1 or truth.shape != prediction.shape or indices.shape != truth.shape:
        raise ValueError('Saved prediction arrays have incompatible shapes.')
    if not np.isin(truth, range(10)).all() or not np.isin(prediction, range(10)).all():
        raise ValueError('Saved predictions contain invalid category IDs.')
    if indices.dtype.kind not in 'iu' or (indices < 0).any() or len(np.unique(indices)) != len(indices):
        raise ValueError('Saved predictions contain invalid or duplicate image indices.')
    return pd.DataFrame({'test_index': indices, 'true_id': truth, 'predicted_id': prediction,
                         'True category': [CLASSES[int(i)] for i in truth],
                         'Predicted category': [CLASSES[int(i)] for i in prediction],
                         'Correct': truth == prediction})


def isolated_loader(work_dir):
    # Reuse the existing verified IDX loader without changing its shared ROOT.
    spec = importlib.util.spec_from_file_location('_dashboard_dataset', data.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(work_dir)
    return module


@st.cache_data(max_entries=2, show_spinner=False)
def local_dataset(work_dir, split, identities):
    loader = isolated_loader(work_dir)
    images, labels = loader.load_images(split)
    if split == 'train':
        # Require the persisted split: the dashboard must never create or change it.
        if not (Path(work_dir) / 'data/split.npz').is_file():
            raise ValueError('The saved data/split.npz is needed to restrict exploration to training IDs.')
        train, _ = loader.split_indices(labels)
        return images[train], labels[train]
    return images, labels


def dataset_identity(work_dir, split):
    prefix = 'train' if split == 'train' else 't10k'
    paths = [work_dir / 'data/raw' / f'{prefix}-{kind}-idx{dim}-ubyte.gz'
             for kind, dim in [('images', 3), ('labels', 1)]]
    if split == 'train':
        paths.append(work_dir / 'data/split.npz')
    return tuple(stamp(path) for path in paths)


def confusion_table(path):
    table = read_csv(stamp(path), index_col=0)
    if set(table.index) != set(CLASSES) or set(table.columns) != set(CLASSES):
        raise ValueError('Confusion matrix categories differ from the recorded class order.')
    table = table.loc[list(CLASSES), list(CLASSES)]
    values = table.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Confusion matrix must contain nonnegative integer counts.')
    return table
