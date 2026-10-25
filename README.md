# Inventory Image Tagging

Project 11, Phase 1: handcrafted visual features and classical machine learning.

We classify 28x28 grayscale clothing images into ten Fashion-MNIST categories. We compare pixels and HOG using Logistic Regression, k-NN, Random Forest and linear SVM, then test RBF SVMs with finer HOG cells. Phase 2 will add a small CNN.

Every model trains on all 48,000 images from the same 48k/12k stratified split. The supplied reference used a 12k training subset. We acknowledge the choices we adapted in [attribution](docs/ATTRIBUTION.md).

<!-- RESULTS_START -->
Validation selects Fine HOG + RBF SVM. It achieves 90.87% accuracy and macro F1 0.9084 on the official test set after training on all 48,000 images. See the results, errors and timings in [Phase 1 findings](reports/phase1_findings.md).
<!-- RESULTS_END -->

## Start here

1. [Executed teaching notebook](notebooks/phase1.ipynb)
2. [Phase 1 findings](reports/phase1_findings.md)
3. [IEEE report PDF](reports/phase1_report/main.pdf) ([LaTeX source](reports/phase1_report/main.tex))
4. [Presentation](reports/presentation/phase1.pptx)
5. [Viva guide](docs/viva_guide.md)
6. [Requirements checklist](docs/phase1_checklist.md)

## Technology

Python 3.12 (3.11 also allowed), pip or `uv`, Streamlit, Plotly, NumPy, pandas, SciPy, scikit-image, scikit-learn, matplotlib, Pillow, joblib, JupyterLab and pytest. Exact dependencies are in `uv.lock`; `requirements.txt` exports the same locked versions for pip, including the notebook and test tools. PyTorch is reserved for Phase 2 and is not needed to reproduce this ML-only study.

```bash
uv sync --locked
bash scripts/uv.sh run --locked python3 scripts/check_environment.py
bash scripts/uv.sh run --locked python3 -m pytest -q
bash scripts/uv.sh run --locked python3 scripts/verify_results.py
bash scripts/uv.sh run --locked python3 -m jupyterlab notebooks/phase1.ipynb
```

The launcher delegates to `uv`. It also handles the colon in the current parent folder (`Deep:ALML project`) by making a temporary alias to the local `.venv`; it does not move your source or environment. In an ordinary path without a colon, normal `uv run ...` commands work directly. If Jupyter opens a browser, select the project interpreter/kernel, not an unrelated global Python.

## Interactive dashboard

The sidebar opens six sections: **Overview**, **Prediction Lab**, **Model Comparison**, **Error Analysis**, **Dataset Explorer** and **How It Works**.

Prediction Lab can compare any available frozen models on the same image, shows real inference latency, and visualizes the exact pixel/HOG representation. Exact 28×28 grayscale PNG input is preserved. A separate experimental PNG/JPEG preparation mode previews grayscale, sizing, brightness, contrast and inversion; ordinary-photo accuracy has not been measured.

Plotly charts support filtered model comparisons and confusion-matrix exploration. Dataset Explorer uses only the persisted training split. Saved results remain useful without local models or downloaded images. Every loaded model still requires its recorded SHA-256 checksum; the dashboard never retrains or changes frozen evidence.

See [dashboard usage and artifact fallbacks](docs/DASHBOARD.md) for details.

## Run the Streamlit dashboard

Run these commands from the `inventory-image-tagging` folder. The dashboard shows the saved model comparison, confusion matrix and error examples. It also supports prediction when the fitted model is available locally.

### Using pip

Use Python 3.12 (recommended) or 3.11. Check `python3 --version` first.

```bash
python3 -m venv .venv-pip
source .venv-pip/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

### Using uv

```bash
uv python install 3.12
uv sync --locked
bash scripts/uv.sh run --locked python3 -m streamlit run app.py
```

In a folder path without a colon, you can also use `uv run --locked python3 -m streamlit run app.py` directly.

Open [localhost:8501](http://localhost:8501) in your browser. Press `Ctrl+C` in the terminal to stop the dashboard. These launch commands follow the [Streamlit installation guide](https://docs.streamlit.io/get-started/installation/command-line).

The saved result tables and figures work on a fresh clone. Prediction also needs the fitted model files, which are excluded from Git. To generate them with pip, run this in the activated environment:

```bash
python3 scripts/reproduce.py
```

With uv, use the reproduction command in the next section. After the run completes, enter `runs/reproduction` in the dashboard's **Experiment folder** field. The default `.` uses the original local experiment. The included sample is a training image for demonstrating prediction, not an extra accuracy test.

You can run other Python commands from this README with pip by replacing `bash scripts/uv.sh run --locked python3` with `python3` in your activated environment. No uv installation is needed for that workflow.

When dependencies change, maintain them in `pyproject.toml`, update `uv.lock`, and refresh the pip export:

```bash
uv export --locked --all-groups --no-hashes --no-emit-project --format requirements-txt --output-file requirements.txt
```

## Mac setup

The team uses Macs. The project instructions and upload scripts target macOS with Python 3.11 or 3.12. Saved results need no GPU; training and prediction run on the CPU. A phone or tablet can view the dashboard hosted by a Mac on the same network, but this repository does not provide a mobile app.

Do not copy a `.venv` folder between computers. Recreate the environment from `requirements.txt` or `uv.lock`. Keep fitted models with their matching locked scikit-learn version, or reproduce the experiment on the new computer.

Local checks have passed on the development Mac. GitHub Actions checks installation, the pipeline and the dashboard on macOS with Python 3.11 and 3.12 after the changes are pushed. Remote checks have not run yet.

## Reproduce the complete experiment

Git contains the report and saved predictions. Data, caches and fitted models stay local, so a fresh clone needs training before prediction. Run the complete experiment in a separate directory:

```bash
bash scripts/uv.sh run --locked python3 scripts/reproduce.py
```

This runs data preparation and EDA, trains the eight core combinations and additional RBF models, then freezes the choices before one official test evaluation under `runs/reproduction/`. It preserves the committed `reports/` evidence. Some SVM fits take several minutes. The full run can take much longer than the reference experiment because it uses more training images. The result tables record fit times.

To reuse an already downloaded original Fashion-MNIST archive folder:

```bash
bash scripts/uv.sh run --locked python3 scripts/reproduce.py --source /absolute/path/to/archives
```

If you interrupt a fit, completed model/feature combinations are reused before the final freeze. If a final test run has started, inspect its outputs and use a new work directory for another experiment. Keep the final marker so the original evaluation remains traceable:

```bash
bash scripts/uv.sh run --locked python3 scripts/reproduce.py --work-dir runs/reproduction-2
```

The first complete experiment is already present in the local checkout, including its ignored data and model files. A later reproduction may differ across platforms/library environments; keep its results separate.

## Step-by-step commands

For a new, empty experiment directory, set the same `--work-dir` on every stage:

```bash
bash scripts/uv.sh run --locked python3 -m src.inventory_tagging.cli --work-dir runs/my-study prepare
bash scripts/uv.sh run --locked python3 -m src.inventory_tagging.cli --work-dir runs/my-study eda
bash scripts/uv.sh run --locked python3 -m src.inventory_tagging.cli --work-dir runs/my-study train
bash scripts/uv.sh run --locked python3 -m src.inventory_tagging.cli --work-dir runs/my-study improve
bash scripts/uv.sh run --locked python3 -m src.inventory_tagging.cli --work-dir runs/my-study evaluate
```

`prepare` and `evaluate` also accept `--source /path/to/original/archives`. `train` never reads test files. `improve` is optional for the eight-model core but included in our completed experiment. `evaluate` freezes choices and refuses repeat test runs. Once choices are frozen, further tuning in that work directory is refused.

## Simple code structure

```text
app.py                Streamlit results dashboard and image prediction
requirements.txt      locked dependencies for pip
src/inventory_tagging/
  data.py       original archives, checksums and saved shared split
  features.py   normalized pixels, coarse HOG and fine HOG
  models.py     textbook classifier builders and capacity measurements
  train.py      explicit settings, validation selection and saved models
  eda.py        training-only audit and exploratory figures
  evaluate.py   frozen test evaluation, errors, recall and inference costs
  cli.py        one command for each stage and a prediction command
scripts/
  uv.sh               uv launcher including the local-path workaround
  reproduce.py        complete workflow into an isolated output directory
  check_environment.py
  make_notebook.py     regenerate the readable teaching notebook source
notebooks/phase1.ipynb
reports/              generated evidence, report source and slides
models/               ignored local fitted models (regenerate on a fresh clone)
data/                 ignored verified archives and saved split
```

## Experiment setup

- Data: [Fashion-MNIST official source](https://github.com/zalandoresearch/fashion-mnist); 60,000 official training and 10,000 official test images. License retained in `docs/FASHION_MNIST_LICENSE.txt`.
- Split: seed 42, stratified 48,000/12,000 from official training only; every model sees the same full training IDs.
- Features: float32 pixels divided by 255; HOG with 9 bins, 7x7 cells and 2x2 blocks (324 values). Fine HOG uses 4x4 cells (1,296 values).
- Learned preprocessing: StandardScaler is fitted only on training features for LR and saved in its pipeline. Linear SVM uses fixed normalized descriptors directly (tolerance 1e-3, max_iter 2,000).
- Search: two settings per required combination; validation macro F1, then accuracy, chooses the best converged candidate. Three additional RBF feature combinations test two C values each.
- Test: representative model and all compared configurations frozen before evaluation. The reference already used this public benchmark. Performance on a new image domain remains unmeasured.
- Metrics: accuracy, macro F1, all-class precision/recall/F1/support, confusion matrices and upper-body errors.
- Costs: common two-thread CPU, warm repetitions, preprocessing/HOG included, separate single-image latency and batch throughput. Trees, stored samples, linear weights and support vectors are documented as different capacity measures.

## Prediction demonstration

Use a benchmark-style 28x28 grayscale PNG, upright on a dark background:

```bash
bash scripts/uv.sh run --locked python3 -m src.inventory_tagging.cli predict assets/sample.png
```

After a fresh reproduction, point to its outputs:

```bash
bash scripts/uv.sh run --locked python3 -m src.inventory_tagging.cli --work-dir runs/reproduction predict assets/sample.png
```

We have not evaluated ordinary colour images or phone photos. The prediction command accepts only the expected image dimensions and grayscale mode.

## Report, presentation and viva

The report uses the official IEEEtran class, included with its original license notice. Open `reports/phase1_report/` in the built-in LaTeX editor or compile `main.tex` using an existing TeX setup. The report has three pages. The editable PowerPoint has 15 slides with speaker notes and source references. [Team details](docs/team.md) list the five supplied names and enrolment numbers. Add the course section, formal affiliation and each member's actual contribution before submission.

The literature review and viva guide explain feature dimensions, classifier objectives, split isolation, macro F1, recall, capacity and latency. Each member should practise explaining the full workflow for the individual viva.

## Optional Docker workflow

```bash
docker compose run --build --rm phase1
```

This uses the same lockfile and writes under `runs/reproduction/`. Docker configuration is provided, but the Docker daemon was unavailable during development, so the container path has not been executed. The native `uv` workflow is the verified path.
