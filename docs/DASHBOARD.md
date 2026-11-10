# Using the dashboard

After setup, start it from the project folder:

```bash
bash scripts/start_dashboard.sh
```

The app opens four pages: Predict, Results, Mistakes and About. Predict is the first page. Its default **Results folder**, under **Local setup** in the sidebar, is `.`: the original
saved experiment. After building models on your Mac, change it to
`runs/phase1` to use that run's own scores and model files.

## Try a picture

Choose a known-category sample from the ten included training examples, or upload
a PNG/JPEG. The samples work without the full dataset and record their original
training-image indices in `assets/examples/manifest.json`. They are not a new
accuracy test and were not chosen based on model predictions.

**Dataset image (unchanged)** requires exactly 28×28 pixels in single-channel
grayscale PNG format. It preserves every pixel, including constant images,
matching the original benchmark contract.

**Prepare an ordinary picture** removes a plain background, crops one item and
centres it on a black 28 × 28 input. Picture settings include a manual crop,
rotation and an optional setting for shadows. Inspect the prepared picture before
predicting. Unclear or multiple foreground regions may require a better picture.
Turn off **Remove a plain background** to use the previous grayscale, resize,
brightness, contrast and inversion controls. For background removal, transparent areas receive a contrasting fill based on
visible pixels. Resize-only mode uses the explicitly selected background colour. Exact benchmark mode preserves every pixel.
Changing the input or models hides stale answers. Ordinary-photo accuracy is
unmeasured; some photographs still produce wrong answers. See
[photo preparation and actual checks](PHOTO_INPUTS.md).

The **Prediction model** selector is visible on the main page. The validation-selected model is marked **Recommended**. Open **Compare models** to add alternatives or **Picture settings** to adjust an ordinary photo, then click **Predict**. Original published models are fetched on demand if they are not already on disk. Reproduction folders still need their own locally built models.
Each uses its own recorded pixel/HOG representation and the existing `extract()`
function. Any fitted scaler remains inside the original saved model pipeline.

Model files download and load only on prediction requests. Downloads are limited to the fixed project release and must match their published size and original frozen SHA-256 before being saved. Their SHA-256 hashes must match the
corresponding frozen records, including after a file is replaced. Four loaded
models are cached at most. Use trusted experiment folders; matching a manifest
establishes consistency, and joblib files are still Python serialized objects.

Prediction time includes feature extraction and `predict()` using two CPU threads.
It excludes decoding, model loading and the separate alternative-class call.
The displayed live time is separate from the original repeated timing benchmark.
Probability estimates appear only for models with `predict_proba()` and are not
claimed to be calibrated. SVM decision values are labelled ranking scores,
never probabilities. Values are not comparable across models.

## Compare results and mistakes

Use plain-language filters to select image measurements and learning methods.
The category balance score is Macro F1; the practice-set score is validation
Macro F1. These terms are explained under **How It Works → Names used in the
project report**. Saved CSV fields and scientific definitions are unchanged.

The error table uses actual categories in rows and predicted categories in
columns. Percentages are normalized within each actual category. Image browsing
requires matching local test archives; their saved indices and true labels must
match the original verified loader. The app never repeats test prediction.

## Data and missing files

Optional training-picture browsing under About uses only the persisted training IDs. It uses an isolated
instance of the existing verified loader, without changing the pipeline's shared
root or creating a new split. It shows bundled category examples and saved figures
when the full training files are absent. CSV/data caches are bounded and only the
active page executes.

`reports/test/results.csv` and `reports/frozen.json` are required to open a run.
For a reproduction folder, missing model files disable new predictions while retaining previews. The original results folder can fetch its published matching models. Missing
confusion counts fall back to the chosen model's saved image when available.
Missing per-image answers or raw images retain saved figures and useful guidance.
The app does not download datasets, train models or change frozen evidence. Prediction details, picture preparation steps and additional charts are collapsed by default.

## Hosted app

Open [Predict](https://inventory-image-tagging.streamlit.app/?view=predict),
[Results](https://inventory-image-tagging.streamlit.app/?view=results) or
[Mistakes](https://inventory-image-tagging.streamlit.app/?view=mistakes). Visitors can predict without
signing in or installing Python. The first prediction for a model downloads its
published file; later predictions reuse it. An idle free-hosted app may take time
to wake up.

The deployment uses Python 3.12 and the committed `uv.lock`. The server binds
explicitly to `0.0.0.0`. All eight active model downloads must match their frozen records. The hosted app uses the validation-selected pixel Random Forest by default.

## Local Phase 2 page

Both phases are on main. Set up with `bash scripts/setup_mac.sh --phase2` and open **Neural model** in the sidebar. The page predicts first, with three-seed scores collapsed under **Saved evaluation results**. It predicts an uploaded image or the included sample using a checksum-verified CNN checkpoint. Build the frozen hybrid classifier once with `cli export` to enable that option locally. If PyTorch or optional model files are absent, the page explains how to enable them. Ordinary photos use the existing labelled preparation mode and have no measured accuracy. The hosted installation uses the smaller Phase 1 dependency set; local Phase 2 prediction requires PyTorch.

## Interface check, 10 November 2026

The first page keeps image selection, the model and Predict visible, with the
preview and answer alongside. The layout stacks on small screens. Optional
comparisons, photo settings, feature views and detailed scores stay collapsed.
Shareable `view` links open the requested page; invalid links fall back to Predict.
The neural page appears only when PyTorch is installed. Its photo mode detects
ordinary uploads, clears stale answers on a new image and keeps results across
reruns. Photo settings also survive navigation. CNN and hybrid scores can still
be read under Results without installing PyTorch. No trained artifacts or official
scores were changed.
