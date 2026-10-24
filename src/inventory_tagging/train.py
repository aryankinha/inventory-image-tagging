"""Fit on training images, select on validation images, never read test data."""
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import skimage
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from .data import ROOT, load_images, split_indices, sha256, write_json
from .features import HOG, extract
from .models import build, choices

FAMILIES = ('logistic_regression', 'knn', 'random_forest', 'linear_svm')


def scores(truth, prediction):
    return {'accuracy': float(accuracy_score(truth, prediction)),
            'macro_f1': float(f1_score(truth, prediction, average='macro'))}


def feature_cache(images, representation):
    """The cache contains fixed descriptors, never a fitted scaler."""
    folder = ROOT / 'reports/feature_cache'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'train_{representation}.npy'
    if not path.exists():
        print(f'Extracting {representation} for {len(images)} official training images...', flush=True)
        np.save(path, extract(images, representation), allow_pickle=False)
    values = np.load(path, allow_pickle=False, mmap_mode='r')
    dimension = {'pixels': 784, 'hog': 324, 'hog_fine': 1296}[representation]
    if values.shape != (len(images), dimension) or values.dtype != np.float32:
        raise ValueError('Feature cache does not match the documented input contract')
    return values


def fit_one(representation, family, images, labels, train_ids, validation_ids):
    name = f'{representation}_{family}'
    report = ROOT / 'reports/training'
    report.mkdir(parents=True, exist_ok=True)
    meta_path = report / f'{name}.json'
    model_path = ROOT / 'models' / f'{name}.joblib'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if not model_path.exists() or sha256(model_path) != meta['model_sha256']:
            raise ValueError(f'{name}: saved model missing or changed; restore it or start a new run folder')
        print(f'Reusing completed {name}', flush=True)
        return
    values = feature_cache(images, representation)
    x_train, x_val = values[train_ids], values[validation_ids]
    trials, best, best_prediction = [], None, None
    for parameters in choices(family):
        model = build(family, parameters)
        print(f'{name}: {parameters}', flush=True)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x_train, labels[train_ids])
        seconds = time.perf_counter() - start
        problems = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        prediction = model.predict(x_val)
        row = {'name': name, 'representation': representation, 'family': family,
               'parameters': parameters, **scores(labels[validation_ids], prediction),
               'fit_seconds': seconds, 'convergence_warnings': problems}
        trials.append(row)
        write_json(report / f'{name}_trials.json', trials)
        print(f"  validation accuracy={row['accuracy']:.4%}, F1={row['macro_f1']:.4f}, fit={seconds:.1f}s", flush=True)
        # Converged candidates only. Ties retain the first (simpler) configuration.
        if not problems and (best is None or (row['macro_f1'], row['accuracy']) >
                             (best['macro_f1'], best['accuracy'])):
            best, best_prediction = row.copy(), prediction.copy()
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path, compress=3)
    if best is None:
        raise RuntimeError(f'No converged candidate for {name}; inspect the trial log')
    np.savez_compressed(report / f'{name}_validation.npz', indices=validation_ids,
                        truth=labels[validation_ids], prediction=best_prediction)
    best.update(training_images=len(train_ids), validation_images=len(validation_ids),
                feature_count=x_train.shape[1], model_sha256=sha256(model_path),
                split_sha256=sha256(ROOT / 'data/split.npz'))
    write_json(meta_path, best)


def run(improve=False):
    if (ROOT / 'reports/frozen.json').exists():
        raise FileExistsError('Final choices are frozen. Do not tune against the reported test results.')
    images, labels = load_images('train')
    train_ids, val_ids = split_indices(labels)
    if len(train_ids) != 48000 or len(val_ids) != 12000:
        raise ValueError('The full 48k/12k protocol is required for this experiment')
    with threadpool_limits(limits=2):
        if improve:
            # Kernel nonlinearity and finer HOG are extra validation-only experiments.
            for representation in ('pixels', 'hog', 'hog_fine'):
                fit_one(representation, 'rbf_svm', images, labels, train_ids, val_ids)
        else:
            for representation in ('pixels', 'hog'):
                for family in FAMILIES:
                    fit_one(representation, family, images, labels, train_ids, val_ids)
    summarize()


def summarize():
    records = [json.loads(p.read_text()) for p in sorted((ROOT / 'reports/training').glob('*.json'))
               if not p.name.endswith('_trials.json')]
    table = pd.DataFrame(records).sort_values(['macro_f1', 'accuracy'], ascending=False, kind='stable')
    table.assign(parameters=table.parameters.map(json.dumps)).to_csv(ROOT / 'reports/validation_results.csv', index=False)
    print(table[['name', 'accuracy', 'macro_f1', 'training_images']].to_string(index=False), flush=True)
    return table


def freeze():
    """Record actual model hashes before test evaluation can start."""
    path = ROOT / 'reports/frozen.json'
    if path.exists():
        return json.loads(path.read_text())
    table = summarize()
    required = {f'{rep}_{family}' for rep in ('pixels', 'hog') for family in FAMILIES}
    if not required.issubset(set(table.name)):
        raise ValueError('Complete all eight required baselines before final evaluation')
    records = table.to_dict('records')
    for record in records:
        if sha256(ROOT / 'models' / f"{record['name']}.joblib") != record['model_sha256']:
            raise ValueError('Model changed after validation')
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'selected': records[0]['name'], 'selection': 'validation macro F1, then accuracy',
                'train_count': 48000, 'validation_count': 12000, 'seed': 42,
                'models': records, 'hog': HOG,
                'training_archives_sha256': {p.name: sha256(p) for p in (ROOT / 'data/raw').glob('train*.gz')},
                'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                'numpy': np.__version__, 'sklearn': sklearn.__version__, 'skimage': skimage.__version__}}
    write_json(path, manifest)
    return manifest


