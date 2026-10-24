"""Evaluate frozen choices once; export metrics, costs, and explainable errors."""
from datetime import datetime, timezone
import json
import hashlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import ROOT, CLASSES, prepare, load_images, write_json, sha256
from .features import extract
from .models import complexity
from .train import freeze, scores
from .eda import save


def predict_batches(model, images, representation, batch_size=128):
    return np.concatenate([model.predict(extract(images[i:i + batch_size], representation))
                           for i in range(0, len(images), batch_size)])


def measure(model, images, representation):
    """Time new images to labels, including HOG and any pipeline scaler."""
    repeats = []
    for batch_size, count in ((1, 1), (128, 1024)):
        sample = images[:count]
        predict_batches(model, sample, representation, batch_size)
        # Short predictions need more repetitions than a 1,024-image throughput batch.
        for repeat in range(20 if batch_size == 1 else 5):
            start = time.perf_counter()
            predict_batches(model, sample, representation, batch_size)
            seconds = time.perf_counter() - start
            repeats.append({'batch_size': batch_size, 'images': count, 'repeat': repeat,
                            'seconds': seconds, 'ms_per_image': seconds * 1000 / count})
    frame = pd.DataFrame(repeats)
    one = frame[frame.batch_size == 1].ms_per_image
    batch = frame[frame.batch_size == 128].ms_per_image
    return {'single_ms_median': float(one.median()), 'single_ms_p95': float(one.quantile(.95)),
            'batch_ms_per_image': float(batch.median()),
            'batch_images_per_second': float(1000 / batch.median())}, repeats


def run(source_dir=None):
    output = ROOT / 'reports/test'
    if (output / 'started.json').exists():
        raise FileExistsError('Final test evaluation already started; use saved outputs instead of tuning on test results.')
    frozen = freeze()
    output.mkdir(parents=True, exist_ok=True)
    # Mark before opening test files. A partial run is visible rather than silently retried.
    write_json(output / 'started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                                       'frozen_sha256': sha256(ROOT / 'reports/frozen.json')})
    prepare(source_dir, include_test=True)
    images, labels = load_images('test')
    training_images, _ = load_images('train')
    training_hashes = {hashlib.sha256(image.tobytes()).digest() for image in training_images}
    test_hashes = {hashlib.sha256(image.tobytes()).digest() for image in images}
    overlap = len(training_hashes & test_hashes)
    write_json(output / 'data_integrity.json', {
        'test_images': len(images), 'unique_test_images': len(test_hashes),
        'exact_train_test_image_overlap': overlap,
        'archives_sha256': {p.name: sha256(p) for p in (ROOT / 'data/raw').glob('t10k*.gz')},
        'note': 'Post-freeze integrity audit only; no changes to split, labels or models.'})
    if overlap:
        raise ValueError('Unexpected exact train/test image overlap; inspect provenance')
    del training_images, training_hashes, test_hashes
    all_rows, all_recalls, all_times, predictions = [], [], [], {}
    with threadpool_limits(limits=2):
        for record in frozen['models']:
            name, representation = record['name'], record['representation']
            path = ROOT / 'models' / f'{name}.joblib'
            if sha256(path) != record['model_sha256']:
                raise ValueError(f'Frozen model changed: {name}')
            model = joblib.load(path)
            print(f'Evaluating {name}...', flush=True)
            pred = predict_batches(model, images, representation)
            predictions[name] = pred
            metrics = scores(labels, pred)
            report = classification_report(labels, pred, labels=np.arange(10), target_names=CLASSES,
                                           output_dict=True, zero_division=0)
            write_json(output / f'{name}_classes.json', report)
            matrix = confusion_matrix(labels, pred, labels=np.arange(10))
            pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(output / f'{name}_confusion.csv')
            for c, label in enumerate(CLASSES):
                all_recalls.append({'name': name, 'category': label, **report[label]})
            timing, repeats = measure(model, images, representation)
            all_times.extend({'name': name, **r} for r in repeats)
            capacity = {k: int(v) if isinstance(v, np.integer) else v for k, v in complexity(model).items()}
            write_json(output / f'{name}_complexity.json', capacity)
            all_rows.append({'name': name, 'representation': representation, 'family': record['family'],
                             'validation_macro_f1': record['macro_f1'], 'test_accuracy': metrics['accuracy'],
                             'test_macro_f1': metrics['macro_f1'], 'train_images': 48000, 'test_images': 10000,
                             'feature_count': record['feature_count'], 'fit_seconds': record['fit_seconds'],
                             'model_bytes': path.stat().st_size, **timing,
                             'complexity_kind': capacity['kind'], 'complexity_count': capacity['count']})
            pd.DataFrame(all_rows).to_csv(output / 'results.csv', index=False)
            print(f"  accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.4f}", flush=True)
    np.savez_compressed(output / 'predictions.npz', truth=labels, indices=np.arange(len(labels)), **predictions)
    pd.DataFrame(all_recalls).to_csv(output / 'per_class.csv', index=False)
    pd.DataFrame(all_times).to_csv(output / 'timing_repeats.csv', index=False)
    figures(images, labels, predictions, pd.DataFrame(all_rows), frozen['selected'])
    selected = next(row for row in all_rows if row['name'] == frozen['selected'])
    write_json(output / 'summary.json', {'status': 'complete', 'selected_before_test': frozen['selected'],
                'selected_results': selected, 'models_evaluated': len(all_rows),
                'timing_protocol': {'device': 'CPU', 'threads': 2, 'single_repetitions': 20,
                                   'batch_repetitions': 5, 'batch_size': 128, 'batch_images': 1024,
                                   'scope': 'uint8 resident images to labels, including descriptors and scaler; excludes file decoding and model loading'},
                'limitations': ['one training split and random seed', 'bounded validation tuning',
                                'test set was also used in the supplied reference, so this is an existing public benchmark, not new unseen domain data',
                                'real catalogue/phone images and near-duplicate products were not evaluated']})
    print('Final test outputs saved. Models were selected before these test scores were calculated.')


def figures(images, truth, predictions, table, selected):
    chosen = predictions[selected]
    # Both required representations, across all four core families.
    core = table[table.family != 'rbf_svm'].copy()
    ax = core.set_index('name')[['test_accuracy', 'test_macro_f1']].plot.barh(figsize=(10, 5), xlim=(0, 1))
    ax.set(xlabel='Official test score', ylabel='', title='Eight required classical comparisons')
    save(ax.figure, 'core_comparison.png')
    ax = table.set_index('name').test_accuracy.plot.barh(figsize=(10, 6), xlim=(0, 1))
    ax.set(xlabel='Test accuracy (choice already frozen)', title='Core models and validation-only improvements')
    save(ax.figure, 'all_models.png')
    matrix = confusion_matrix(truth, chosen, labels=range(10))
    fig, ax = plt.subplots(figsize=(9, 8))
    proportions = matrix / matrix.sum(axis=1, keepdims=True)
    im = ax.imshow(proportions, vmin=0, vmax=1, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{matrix[i,j]}', ha='center', va='center', fontsize=8,
                    color='white' if proportions[i,j] > .5 else 'black')
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted category', ylabel='True category', title=f'Validation-selected model: {selected}')
    plt.setp(ax.get_xticklabels(), rotation=55, ha='right')
    fig.colorbar(im, ax=ax, label='Fraction of true class'); save(fig, 'selected_confusion.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    recall = matrix.diagonal() / matrix.sum(1)
    ax.bar(CLASSES, recall, color='#147d92'); ax.set(ylim=(0, 1), ylabel='Recall', title='Recall for every category')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right'); save(fig, 'selected_recall.png')
    errors = np.flatnonzero(chosen != truth)
    focus = errors[np.isin(truth[errors], [0, 2, 4, 6])]
    rng = np.random.default_rng(42)
    chosen_ids = rng.choice(focus, min(12, len(focus)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    rows = []
    for ax in axes.flat:
        ax.axis('off')
    for ax, i in zip(axes.flat, chosen_ids):
        ax.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'True: {CLASSES[truth[i]]}\nPredicted: {CLASSES[chosen[i]]}', fontsize=9)
        rows.append({'test_index': int(i), 'true': CLASSES[truth[i]], 'predicted': CLASSES[chosen[i]]})
    fig.suptitle('Illustrative upper-body errors (fixed-seed sample; not all errors)')
    save(fig, 'error_examples.png')
    pd.DataFrame(rows).to_csv(ROOT / 'reports/test/error_examples.csv', index=False)
    pairs = [{'true': CLASSES[i], 'predicted': CLASSES[j], 'count': int(matrix[i,j]),
              'rate_within_true_class': float(proportions[i,j])}
             for i in [0, 2, 4, 6] for j in [0, 2, 4, 6] if i != j]
    pd.DataFrame(pairs).sort_values('count', ascending=False).to_csv(ROOT / 'reports/test/upper_body_confusions.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in table.itertuples():
        ax.scatter(row.batch_ms_per_image, row.test_accuracy, s=45)
        ax.annotate(row.name.replace('logistic_regression', 'LR').replace('random_forest', 'RF'),
                    (row.batch_ms_per_image, row.test_accuracy), xytext=(5, 3), textcoords='offset points', fontsize=7)
    ax.set(xscale='log', xlabel='Batch CPU milliseconds per image (log scale)', ylabel='Test accuracy',
           title='Accuracy and inference cost; includes feature extraction')
    save(fig, 'accuracy_cost.png')
