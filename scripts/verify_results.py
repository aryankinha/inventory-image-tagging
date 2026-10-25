"""Independently recompute saved scores and verify the original frozen contracts."""
import argparse
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_results(root=ROOT):
    root = Path(root)
    reports = root / 'reports'
    frozen_path = reports / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    summary = json.loads((reports / 'test/summary.json').read_text())
    started = json.loads((reports / 'test/started.json').read_text())
    require(started['frozen_sha256'] == digest(frozen_path), 'The frozen manifest changed after test evaluation started.')
    table = pd.read_csv(reports / 'test/results.csv')
    per_class = pd.read_csv(reports / 'test/per_class.csv')
    timing = pd.read_csv(reports / 'test/timing_repeats.csv')
    require(set(table.name) == {r['name'] for r in frozen['models']}, 'The model list differs from the frozen manifest.')
    with np.load(reports / 'test/predictions.npz', allow_pickle=False) as arrays:
        truth = arrays['truth']
        require(np.array_equal(arrays['indices'], np.arange(10000)), 'Unexpected test-image indices.')
        require(np.array_equal(np.bincount(truth, minlength=10), np.full(10, 1000)), 'Unexpected test-category counts.')
        for row in table.itertuples():
            prediction = arrays[row.name]
            cm = np.zeros((10, 10), dtype=np.int64)
            np.add.at(cm, (truth, prediction), 1)
            expected = pd.read_csv(reports / f'test/{row.name}_confusion.csv', index_col=0).to_numpy()
            require(np.array_equal(cm, expected), row.name + ': confusion counts differ.')
            accuracy = np.diag(cm).sum() / cm.sum()
            macro_f1 = (2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))).mean()
            require(np.isclose(accuracy, row.test_accuracy) and np.isclose(macro_f1, row.test_macro_f1), row.name + ': scores differ.')
            recalls = np.diag(cm) / cm.sum(1)
            precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(10, dtype=float), where=cm.sum(0) != 0)
            f1 = 2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))
            # CSV order is validated against the actual saved class names.
            class_report = json.loads((reports / f'test/{row.name}_classes.json').read_text())
            from src.inventory_tagging.data import CLASSES
            for label, category in enumerate(CLASSES):
                saved = per_class[(per_class.name == row.name) & (per_class.category == category)]
                require(len(saved) == 1, row.name + ': a per-class row is missing or duplicated.')
                for metric, expected in {'precision': precision[label], 'recall': recalls[label], 'f1-score': f1[label], 'support': cm[label].sum()}.items():
                    require(np.isclose(saved.iloc[0][metric], expected), row.name + ': per-class ' + metric + ' differs.')
                    require(np.isclose(class_report[category][metric], expected), row.name + ': class report ' + metric + ' differs.')
            repeats = timing[timing.name == row.name]
            single = repeats[repeats.batch_size == 1].ms_per_image
            batch = repeats[repeats.batch_size == 128].ms_per_image
            require(len(single) == 20 and len(batch) == 5, row.name + ': timing repetitions differ.')
            require(np.isclose(single.median(), row.single_ms_median) and np.isclose(single.quantile(.95), row.single_ms_p95)
                    and np.isclose(batch.median(), row.batch_ms_per_image)
                    and np.isclose(1000 / batch.median(), row.batch_images_per_second), row.name + ': saved timing summary differs.')
            capacity = json.loads((reports / f'test/{row.name}_complexity.json').read_text())
            require(capacity['kind'] == row.complexity_kind and capacity['count'] == row.complexity_count, row.name + ': saved model complexity differs.')
    require(summary['selected_before_test'] == frozen['selected'], 'The selected model differs.')
    chosen = table.set_index('name').loc[frozen['selected']]
    for key in ('test_accuracy', 'test_macro_f1'):
        require(np.isclose(summary['selected_results'][key], chosen[key]), 'Summary differs: ' + key)
    for filename, expected in frozen['source_sha256'].items():
        require(digest(ROOT / 'src/inventory_tagging' / filename) == expected, 'Pipeline source changed: ' + filename)
    present_models = 0
    for record in frozen['models']:
        path = root / 'models' / (record['name'] + '.joblib')
        if path.exists():
            require(digest(path) == record['model_sha256'], 'Frozen model changed: ' + str(path))
            expected_size = table.set_index('name').loc[record['name'], 'model_bytes']
            require(path.stat().st_size == expected_size, 'Model size changed: ' + str(path))
            present_models += 1
        split = root / 'data/split.npz'
        if split.exists():
            require(digest(split) == record['split_sha256'], 'The persisted split changed.')
    for name, expected in frozen['training_archives_sha256'].items():
        path = root / 'data/raw' / name
        if path.exists():
            require(digest(path) == expected, 'Training archive changed: ' + name)
    return {'status': 'passed', 'models_recomputed': len(table), 'source_hashes_verified': len(frozen['source_sha256']),
            'present_model_hashes_verified': present_models,
            'note': 'Data and fitted models may be absent on a fresh clone; saved scores, recall, timing and frozen evidence are still checked.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT, help='Completed experiment folder to verify')
    args = parser.parse_args()
    print(json.dumps(check_results(args.work_dir.resolve()), indent=2))


if __name__ == '__main__':
    main()
