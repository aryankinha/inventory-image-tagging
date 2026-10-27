"""Build and run notebooks/phase2.ipynb. It reads saved results and never retrains."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
md('''# Project 11: Inventory Image Tagging
## Phase 2: a small CNN, an ablation study and a hybrid

Phase 1 described each picture with fixed numbers (pixels, HOG) and used classical classifiers. Here a small CNN learns its own description, and we test whether that helps the categories Phase 1 mixed up: Shirt, T-shirt, Pullover and Coat.

This notebook reads the saved Phase 2 results in `reports/phase2/`. It does not retrain anything. It needs PyTorch (`requirements-phase2.txt`) but no dataset download.'''),
code('''from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from IPython.display import display, Image as Show
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.inventory_tagging_dl import cnn
OUT = ROOT / "reports/phase2"
frozen = json.loads((OUT / "frozen.json").read_text())
summary = json.loads((OUT / "test/summary.json").read_text())
CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
print("PyTorch", torch.__version__, "| final CNN:", frozen["cnn_config"], frozen["cnn_settings"])'''),
md('''## 1. The network

Each convolution block slides 3x3 filters over the picture (they learn edges and corners), applies batch normalisation and ReLU, then max pooling halves the width and height: 28 to 14 to 7. Two dense layers turn the 64 x 7 x 7 map into a 64-number summary of the picture. Dropout switches off a share of units during training only. A final linear layer gives ten scores.

The 64-number summary is what the hybrid model hands to a classical classifier.'''),
code('''model = cnn.SmallCNN(**frozen["cnn_settings"])
print(model)
x = torch.zeros(1, 1, 28, 28)
for layer in model.features:
    x = layer(x)
    if layer.__class__.__name__ == "MaxPool2d":
        print("after pooling:", tuple(x.shape))
print("trainable parameters:", f"{cnn.count_parameters(model):,}", "| without the last layer:", f"{cnn.count_parameters(model, head=False):,}")'''),
md('''## 2. Augmentation

Training pictures are randomly mirrored and shifted by up to 2 pixels. A mirrored or nudged shirt is still a shirt, so the network sees more variety. Test pictures are never changed. Top row: ten training examples. Bottom row: one random augmentation of each.'''),
code('''import matplotlib.pyplot as plt
files = sorted((ROOT / "assets/examples").glob("*.png"))
pictures = np.stack([np.asarray(Image.open(f).convert("L")) for f in files])
original = cnn.to_tensor(pictures)
torch.manual_seed(1)
changed = cnn.augment(original)
fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for i in range(10):
    axes[0][i].imshow(original[i, 0], cmap="gray"); axes[1][i].imshow(changed[i, 0], cmap="gray")
    axes[0][i].axis("off"); axes[1][i].axis("off")
plt.show()'''),
md('''## 3. Ablation study

We remove one ingredient at a time from the full model and train each version with three seeds on the same 48,000 pictures. Scores are on the 12,000 validation pictures. Bars in the figure show the average and the seed-to-seed spread. If two versions differ by less than that spread, we do not claim one is better.

`light_dropout` was added after the first runs showed that dropout 0.4 hurt. It was judged on validation data only.'''),
code('''table = pd.read_csv(OUT / "ablation_summary.csv")
display(table[["config", "validation_macro_f1_mean", "validation_macro_f1_std", "validation_accuracy_mean", "best_epoch_mean", "parameters"]].round(4))
display(Show(filename=str(OUT / "figures/ablation.png")))
selection = json.loads((OUT / "selection.json").read_text())
print("Final CNN:", selection["config"], "| rule:", selection["rule"])
print("Best overall:", selection["best_overall_config"], round(selection["best_overall_validation_macro_f1_mean"], 4),
      "vs selected", round(selection["validation_macro_f1_mean"], 4), "| seed spread of selected", round(selection["seed_std_of_selected"], 4))'''),
md('''## 4. Early stopping

Training loss keeps falling, while validation loss eventually stops improving. We stop after 5 epochs without improvement and keep the weights from the best epoch. Only validation data decides this.'''),
code('''display(Show(filename=str(OUT / "figures/learning_curves.png")))'''),
md('''## 5. Hybrid: CNN features into a classical classifier

The CNN's 64-number summary is given to a Phase 1 classifier instead of the CNN's own last layer. The classifier is fitted on training pictures only. As a control, we do the same with features from an untrained CNN of the same shape.'''),
code('''hv = pd.read_csv(OUT / "hybrid_validation.csv")
display(hv.groupby(["family", "parameters"])[["validation_accuracy", "validation_macro_f1"]].mean().round(4))
hs = json.loads((OUT / "hybrid_selection.json").read_text())
print("Selected:", hs["family"], hs["parameters"])
print("validation macro F1  hybrid:", round(hs["validation_macro_f1_mean"], 4),
      "| CNN's own last layer:", round(hs["cnn_own_head_validation_macro_f1_mean"], 4),
      "| untrained-CNN features:", round(hs["untrained_cnn_features_validation_macro_f1_mean"], 4))'''),
md('''## 6. Final test, run once

The settings above were frozen in `reports/phase2/frozen.json` before the 10,000 official test pictures were scored. The CNN and the hybrid were each trained with three seeds.'''),
code('''comparison = pd.read_csv(OUT / "test/comparison.csv")
display(comparison.sort_values("test_accuracy", ascending=False).round(4))
display(Show(filename=str(OUT / "figures/test_comparison.png")))
for key in ("cnn_vs_phase1_seed42", "hybrid_vs_cnn_seed42"):
    d = summary[key]
    print(f"{key}: accuracy difference {d['difference']*100:+.2f} points, 95% interval [{d['low']*100:+.2f}, {d['high']*100:+.2f}]")'''),
md('''## 7. Does the CNN separate the similar categories better?

Recall is the share of each category found. The table counts how often one upper-body category is mistaken for another on the test set (seed 42 for the CNN and the hybrid).'''),
code('''display(pd.read_csv(OUT / "test/per_class_recall.csv").round(3))
display(Show(filename=str(OUT / "figures/recall_comparison.png")))
display(pd.read_csv(OUT / "test/upper_body_confusions.csv").sort_values("phase1", ascending=False).head(8))'''),
code('''display(Show(filename=str(OUT / "figures/error_examples.png")))
display(Show(filename=str(OUT / "figures/confusion_cnn.png")))'''),
md('''## 8. Cost

Timed on the CPU with two threads, from a 28x28 picture to a label, like Phase 1. The CNN's last layer is quick; in the hybrid, the 400-tree Random Forest dominates the time.'''),
code('''costs = summary["costs"]
display(pd.DataFrame({"CNN": costs["cnn"], "Hybrid": {k: v for k, v in costs["hybrid"].items() if k != "classifier"}}))
print("Hybrid classifier size:", costs["hybrid"]["classifier"])'''),
md('''## 9. Predict a picture with the saved CNN

Loads the committed checkpoint (seed 42) and predicts the included sample, a training picture used only to show how prediction works.'''),
code('''saved = cnn.SmallCNN(**frozen["cnn_settings"])
saved.load_state_dict(torch.load(OUT / f"checkpoints/{frozen['cnn_config']}_seed42.pt", map_location="cpu", weights_only=True))
picture = np.asarray(Image.open(ROOT / "assets/sample.png").convert("L"))[None]
logits, _ = cnn.forward_all(saved, cnn.to_tensor(picture), torch.device("cpu"))
scores = logits.softmax(1)[0]
print("true category:", json.loads((ROOT / "assets/sample.json").read_text())["true_class"])
for index in scores.argsort(descending=True)[:3]:
    print(f"{CLASSES[index]:12s} {scores[index]:.3f}")'''),
md('''## 10. Conclusions and limits'''),
code('''p1, c, h = summary["phase1_test_accuracy"], summary["cnn_test_accuracy_mean"], summary["hybrid_test_accuracy_mean"]
print(f"Test accuracy: Phase 1 best {p1:.2%}, CNN {c:.2%} (+/- {summary['cnn_test_accuracy_std']:.2%}), hybrid {h:.2%} (+/- {summary['hybrid_test_accuracy_std']:.2%}).")
recall = pd.read_csv(OUT / "test/per_class_recall.csv").set_index("category")
print("Shirt recall: Phase 1", f"{recall.loc['Shirt'].iloc[0]:.1%}", "-> CNN", f"{recall.loc['Shirt','cnn_mean']:.1%}", "-> hybrid", f"{recall.loc['Shirt','hybrid_mean']:.1%}")
print("Limits: Fashion-MNIST is a small, centred, familiar benchmark; three seeds; one split; a small settings search; GPU training is not exactly repeatable.")''')
]
client = NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(root / 'notebooks')}})
client.execute()
nbf.write(nb, root / 'notebooks/phase2.ipynb')
print('Phase 2 notebook built and every cell ran.')
