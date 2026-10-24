"""Check only the libraries and model operations used in Phase 1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise SystemExit('Use Python 3.12 (recommended) or 3.11 for this project.')
import platform
import importlib.metadata
import numpy as np
from src.inventory_tagging.features import extract
from src.inventory_tagging.models import build

print('Python:', platform.python_version(), '|', platform.platform())
for package in ('numpy','pandas','scipy','scikit-learn','scikit-image','matplotlib','pillow','joblib','streamlit'):
    print(package, importlib.metadata.version(package))
images = np.random.default_rng(42).integers(0, 256, (20,28,28), dtype=np.uint8)
x = extract(images, 'hog')
model = build('logistic_regression', {'C': 1})
model.fit(x, np.repeat([0,1],10))
assert model.predict(x).shape == (20,)
assert x.shape == (20,324) and np.isfinite(x).all()
print('Phase 1 imports, HOG and classifier fit/predict passed; no dataset downloaded.')
