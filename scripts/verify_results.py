"""Independently recompute final metrics with NumPy and verify frozen file hashes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
frozen = json.loads((root / 'reports/frozen.json').read_text())
summary = json.loads((root / 'reports/test/summary.json').read_text())
table = pd.read_csv(root / 'reports/test/results.csv')
checks = []
with np.load(root / 'reports/test/predictions.npz', allow_pickle=False) as arrays:
    y = arrays['truth']
    for row in table.itertuples():
        prediction = arrays[row.name]
        cm = np.zeros((10,10), dtype=np.int64)
        np.add.at(cm, (y, prediction), 1)
        expected = pd.read_csv(root / f'reports/test/{row.name}_confusion.csv', index_col=0).to_numpy()
        assert np.array_equal(cm, expected), row.name
        accuracy = np.diag(cm).sum() / cm.sum()
        macro_f1 = (2*np.diag(cm)/(cm.sum(0)+cm.sum(1))).mean()
        assert np.isclose(accuracy, row.test_accuracy), row.name
        assert np.isclose(macro_f1, row.test_macro_f1), row.name
        checks.append(row.name)
assert summary['selected_before_test'] == frozen['selected']
for filename, expected in frozen['source_sha256'].items():
    path = root / 'src/inventory_tagging' / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, filename
present_models = 0
for record in frozen['models']:
    path = root / 'models' / f"{record['name']}.joblib"
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['model_sha256'], path
        present_models += 1
result = {'status':'passed', 'models_recomputed':len(checks),
          'source_hashes_verified':len(frozen['source_sha256']), 'present_model_hashes_verified':present_models,
          'note':'Ignored model files may be absent on a fresh clone; generated evidence is still checked.'}
print(json.dumps(result,indent=2))
