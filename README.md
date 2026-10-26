# Inventory Image Tagging

[Open the dashboard](https://inventory-image-tagging.streamlit.app/)

Project 11, Phase 1: recognise ten Fashion-MNIST clothing categories using image
brightness, edge measurements and classical machine learning.

We compare raw pixels and HOG with Logistic Regression, k-NN, Random Forest and
linear SVM: two representations × four classifiers = eight models. Every model uses
the same 48,000 training and 12,000 validation images. The supplied reference used
a smaller training subset; see [attribution](docs/ATTRIBUTION.md).

<!-- RESULTS_START -->
Validation selects Pixels + RF. It achieves 87.61% accuracy and macro F1 0.8747 on the official test set after training on all 48,000 images. See the results, errors and timings in [Phase 1 findings](reports/phase1_findings.md).
<!-- RESULTS_END -->

## Start on a Mac

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you want
it to manage Python and the project dependencies. Then:

```bash
git clone https://github.com/aryankinha/inventory-image-tagging.git
cd inventory-image-tagging
bash scripts/setup_mac.sh
bash scripts/start_dashboard.sh
```

Open the local address printed in Terminal, usually <http://localhost:8501>.
Stop the app with **Control+C**. In a new Terminal, run the start command again;
you do not need to activate an environment or remember a Python executable.

If you prefer pip, install Python **3.11 or 3.12**, then replace the setup command
with this one. The start and other helper commands stay the same:

```bash
bash scripts/setup_mac.sh --pip
bash scripts/start_dashboard.sh
```

Setup installs the exact locked dependencies, checks imports and saved evidence,
and installs a notebook kernel **inside that environment**. It does not download
the dataset or train models. Each clone gets its own environment. The pip helper
puts its environment under `~/venvs/`, so spaces and a colon in the project path
are handled without renaming your folder. `INVENTORY_PIP_ENV` can select another
absolute environment path without a colon.

## What works immediately after cloning?

The dashboard opens directly on **Predict**. Upload a picture or choose a sample,
then click **Predict**. Model selection and photo adjustments are under
**Picture and model settings**. Ordinary photos are experimental and have not been
benchmarked. Exact 28×28 grayscale PNG inputs remain unchanged.

The other pages are **Results**, **Mistakes**, and **About**. Longer explanations,
extra comparisons and training-picture browsing are optional.

The eight model files are published in the [Phase 1 model release](https://github.com/aryankinha/inventory-image-tagging/releases/tag/phase1-models-v2).
A new clone or hosted app downloads only a requested model on the first prediction,
checks its size and frozen SHA-256 checksum, and reuses it locally. The first
prediction therefore needs internet access and can take longer. It does not train
models or download the dataset. Full image browsing still requires local data.


## Build the models once on your Mac

After setup, run:

```bash
bash scripts/build_models.sh
```

This downloads and checks the official data, creates the shared training split,
trains the eight required combinations, fixes all choices
using validation data, and evaluates the official test set once. Keep Terminal
open until it finishes. Models and this run's own reports go under
`runs/phase1/`; the original committed reports stay untouched.

When it finishes, enter **runs/phase1** in the dashboard's **Results folder**
box under **Local setup** in the sidebar. The app then shows that run's scores and uses that run's matching model files.
A reproduction can differ between computers; its results are not substituted into
the original report. Running the build command again checks a completed run and
exits without repeating its final test. An interrupted final test requires a new
work folder, keeping the interrupted evidence:

```bash
bash scripts/build_models.sh --work-dir runs/phase1-2
```

To reuse original downloaded archives, add `--source /absolute/path/to/archives`.
If a fit is interrupted before choices are frozen, already completed fits are reused.
You cannot use the project root as the reproduction folder.

For command-line prediction from a completed reproduction:

```bash
bash scripts/run_python.sh -m src.inventory_tagging.cli --work-dir runs/phase1 predict assets/sample.png
```

This command expects an upright 28×28 grayscale image. The dashboard also offers
an explicitly experimental mode for ordinary photos, with visible resizing,
brightness, contrast and background controls. Photos have not been benchmarked.
Transparent areas are filled with the selected background; blank photo inputs
show a helpful error. Dataset-image mode preserves the original pixels exactly.

## Check the installation and saved results

```bash
bash scripts/run_python.sh -m pytest -q
bash scripts/run_python.sh scripts/verify_results.py
bash scripts/run_python.sh scripts/verify_artifacts.py
bash scripts/run_python.sh scripts/check_dashboard.py
```

The result check independently recomputes scores from saved answers and checks
recall, timing summaries, model sizes/capacity records, frozen source hashes and
any local model/split/archive hashes. The artifact check additionally protects
the versioned reports, figures and documented training examples.
The dashboard check opens all four pages and runs any locally available models;
it checks models already present on disk. Published model downloads are tested separately.

After building the models, check the reproduction explicitly:

```bash
bash scripts/run_python.sh scripts/verify_results.py --work-dir runs/phase1
bash scripts/run_python.sh scripts/check_dashboard.py --work-dir runs/phase1 --require-models
```

## Open the teaching notebook

Prepare the original training data without loading the final-test set:

```bash
bash scripts/run_python.sh -m src.inventory_tagging.cli prepare
bash scripts/run_python.sh -m jupyterlab notebooks/phase1.ipynb
```

Choose **Python (Inventory Tagging)** from the project environment. The notebook
reads the committed evidence and training images; it does not silently
retrain the experiment. CNN training and results belong to Phase 2.

## Report and presentation

- [Teaching notebook](notebooks/phase1.ipynb)
- [Phase 1 findings](reports/phase1_findings.md)
- [IEEE report PDF](reports/phase1_report/main.pdf) and [LaTeX source](reports/phase1_report/main.tex)
- [Presentation](reports/presentation/phase1.pptx)
- [Presentation PDF](reports/presentation/phase1.pdf)
- [Prediction flow (Mermaid)](reports/presentation/prediction_flow.mmd)
- [Viva guide](docs/viva_guide.md) and [requirements checklist](docs/phase1_checklist.md)
- [Dashboard details](docs/DASHBOARD.md) and [verification record](docs/FRESH_CLONE_CHECK.md)

The report has three pages and the presentation has nine slides with speaker notes, including the dashboard prediction flow.
[Team details](docs/team.md) still need the course section, formal affiliation and
each member's actual contribution before submission.

## Project structure and dependencies

`src/inventory_tagging/` contains the frozen data, feature, model, training and
evaluation code. `dashboard/` contains display code and read-only analysis.
`scripts/` provides setup, startup, reproduction and verification commands.
`reports/` holds saved evidence; `assets/examples/` holds ten documented training
pictures. Generated `data/`, `models/` and `runs/` stay local.

Python 3.11/3.12, uv or pip, Streamlit, Plotly, NumPy, pandas, SciPy, scikit-image,
scikit-learn, matplotlib, Pillow, joblib, JupyterLab and pytest are used. No deep
learning library is needed for Phase 1. Dependencies are pinned in `uv.lock` and
exported to `requirements.txt`. When dependencies change, update both files:

```bash
uv lock
uv export --locked --all-groups --no-hashes --no-emit-project --format requirements-txt --output-file requirements.txt
```

The existing `scripts/uv.sh` remains available for direct uv commands and handles
colon-containing project paths. Do not copy virtual environments between Macs.

## Host on Streamlit Community Cloud

Deploy this GitHub repository from branch `main`, with entrypoint `app.py` and
Python **3.12**. The first prediction fetches its original verified model from the
release above; no training or secrets are needed. The results-folder setting can
stay at `.`. See [dashboard details](docs/DASHBOARD.md).
