"""Regenerate the short teaching notebook; executing it never silently retrains."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md('''# Project 11 - Inventory Image Tagging\n## Phase 1: pixels, HOG, and classical ML\n\nWe predict one of ten clothing categories from a 28x28 grayscale image. The notebook follows the saved Phase 1 experiment and checks its results. We will test CNNs in Phase 2.\n\nThis notebook reads the saved results. Prepare its training images with `bash scripts/run_python.sh -m src.inventory_tagging.cli prepare`. To reproduce all eight models in a separate folder, use `bash scripts/build_models.sh`. The original reports stay protected.'''),
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.inventory_tagging.data import CLASSES, load_images, split_indices
from src.inventory_tagging.features import extract
from src.inventory_tagging.models import build
print("Project:", ROOT.name)
print("Categories:", ", ".join(CLASSES))'''),
md('''## 1. Dataset and split\nThe official dataset has 60,000 training images and 10,000 test images. Using seed 42, we divide the official training set into 48,000 training and 12,000 validation images, keeping class proportions equal. Every model uses the same training IDs. We choose settings using validation macro F1 and fix those choices before calculating test scores.\n\nThe public test benchmark was already used in the supplied reference. The results describe this public benchmark. Performance on new kinds of images still needs testing.'''),
code('''audit = json.loads((ROOT / "reports/data_audit.json").read_text())
display(pd.Series(audit).drop("archives"))
display(pd.read_csv(ROOT / "reports/class_counts.csv"))
images, labels = load_images("train")
train_ids, validation_ids = split_indices(labels)
assert not np.intersect1d(train_ids, validation_ids).size
assert len(train_ids) == 48000
'''),
code('''display(Image(filename=str(ROOT / "reports/figures/training_samples.png")))
display(Image(filename=str(ROOT / "reports/figures/class_means.png")))
display(Image(filename=str(ROOT / "reports/figures/intensity_distribution.png")))'''),
md('''## 2. Feature engineering\nPixels become float32 values in [0,1], then flatten to 784 entries. Reshaping the vector recovers every pixel in the original image.\n\nHOG measures edge directions and groups them into histograms within small cells. It normalizes overlapping blocks of cells, then joins the values into one vector. Nine bins with 7x7-pixel cells and 2x2-cell blocks give 324 values. We keep image orientation fixed.\n\nStandardScaler is fitted on training data only for Logistic Regression. Linear SVM uses the already normalized descriptors directly with solver tolerance 1e-3 and max_iter 2,000. k-NN and RF also use the fixed pixel/HOG values directly.'''),
code('''sample = images[train_ids[:3]]
for representation in ("pixels", "hog"):
    values = extract(sample, representation)
    print(representation, values.shape, values.dtype)
assert np.allclose(extract(sample, "pixels").reshape(3,28,28), sample / 255)
display(Image(filename=str(ROOT / "reports/figures/hog_visualization.png")))'''),
md('''## 3. Four classical decision rules\n- Logistic Regression learns linear class scores using log loss and regularization.\n- k-NN uses nearby training examples to vote, giving closer examples more weight.\n- Random Forest combines randomized decision trees.\n- Linear SVM learns class boundaries by balancing margins and training errors.\n\nEight required combinations = two representations x four classifiers. Each has two candidate settings.\n\nThe next cell displays the library model and its scaler. It loads the saved validation results without fitting a model.'''),
code('''print(build("logistic_regression", {"C": 1.0}))
validation = pd.read_csv(ROOT / "reports/validation_results.csv")
display(validation[["name","feature_count","parameters","training_images","accuracy","macro_f1","fit_seconds"]])
frozen = json.loads((ROOT / "reports/frozen.json").read_text())
print("Selected before test evaluation:", frozen["selected"])
assert frozen["selected"] == validation.iloc[0]["name"]'''),
md('''## 4. Final test evaluation\nAccuracy measures overall correctness. For each class, precision measures how many predicted positives are correct and recall measures how many true positives are recovered. F1 = 2TP/(2TP+FP+FN); macro F1 averages the ten class scores.\n\nWe compare test results for every retained model. Our selected model remains the one chosen using validation data.'''),
code('''test = pd.read_csv(ROOT / "reports/test/results.csv")
display(test[["name","test_accuracy","test_macro_f1","feature_count","complexity_kind","complexity_count","model_bytes"]])
display(Image(filename=str(ROOT / "reports/figures/core_comparison.png")))
selected = frozen["selected"]
selected_row = test.set_index("name").loc[selected]
print(f"Validation-selected model test accuracy: {selected_row.test_accuracy:.2%}")'''),
md('''## 5. Shirts, T-shirts, pullovers, and coats\nConfusion rows are true categories and columns are predictions. A Shirt predicted as Coat and a Coat predicted as Shirt count as different mistakes. We sample upper-body errors with a fixed seed for inspection. This sample does not represent all failures.'''),
code('''per_class = pd.read_csv(ROOT / "reports/test/per_class.csv")
display(per_class[per_class.name == selected].drop(columns="name"))
display(pd.read_csv(ROOT / "reports/test/upper_body_confusions.csv"))
display(Image(filename=str(ROOT / "reports/figures/selected_confusion.png")))
display(Image(filename=str(ROOT / "reports/figures/error_examples.png")))'''),
md('''## 6. Accuracy versus computational cost\nTimings include descriptor extraction and learned pipeline scaling on the same CPU. Single-image latency is separate from throughput over batches of 128. Timing excludes file decoding and model loading. Model capacity takes different forms: forests store nodes, k-NN stores images, and linear models store weights. These counts cannot be treated as equivalent parameter counts.'''),
code('''display(test[["name","fit_seconds","single_ms_median","single_ms_p95","batch_ms_per_image","batch_images_per_second","model_bytes"]])
display(Image(filename=str(ROOT / "reports/figures/accuracy_cost.png")))'''),
md('''## 7. Reusable prediction\nThis example loads a saved model and predicts one training image. It demonstrates prediction only and provides no additional accuracy estimate. For a command-line demonstration use `bash scripts/run_python.sh -m src.inventory_tagging.cli predict path/to/image.png` with a 28x28 grayscale image on a dark background. Arbitrary phone-photo performance has not been measured.'''),
code('''import joblib
from src.inventory_tagging.data import sha256
# A new clone builds into runs/phase1; the original reports stay separate.
demo_root = ROOT if (ROOT / "models" / f"{selected}.joblib").exists() else ROOT / "runs/phase1"
demo_manifest = demo_root / "reports/frozen.json"
if demo_manifest.exists():
    demo_frozen = json.loads(demo_manifest.read_text())
    record = next(m for m in demo_frozen["models"] if m["name"] == demo_frozen["selected"])
    model_path = demo_root / "models" / f"{record['name']}.joblib"
else:
    model_path = None
if model_path is not None and model_path.exists():
    if sha256(model_path) != record["model_sha256"]:
        raise ValueError("The model does not match this run's saved record.")
    if sha256(ROOT / "data/split.npz") != record["split_sha256"]:
        raise ValueError("The demo's training images use a different split.")
    model = joblib.load(model_path)
    index = int(train_ids[0])
    prediction = int(model.predict(extract(images[index:index+1], record["representation"]))[0])
    print("Model folder for this demonstration:", demo_root)
    print("Known category:", CLASSES[int(labels[index])])
    print("Predicted category:", CLASSES[prediction])
else:
    print("Saved results are available. To enable prediction, run: bash scripts/build_models.sh")'''),
md('''## 8. Conclusions, limitations, and viva\nRead `reports/phase1_findings.md` for result-specific conclusions, `docs/literature_review.md` for primary sources, and `docs/viva_guide.md` for explanations. The findings apply to one split and a limited search. Full catalogue photos, unseen classes, near-duplicates and calibration need separate evaluation.\n\nThe reference is acknowledged in `docs/ATTRIBUTION.md`. All result tables use our new experiment with 48k training images. Phase 2 will use this same split for a small CNN with pooling, dropout and early stopping, then compare errors and compute cost.''')
]
nb.metadata.kernelspec = {'display_name': 'Python (Inventory Tagging)', 'language': 'python', 'name': 'inventory-tagging'}
nb.metadata.language_info = {'name': 'python', 'version': '3.12'}
nbf.write(nb, root / 'notebooks/phase1.ipynb')
print('Notebook source generated.')
