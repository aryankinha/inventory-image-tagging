# Exploring the saved experiment

Start from the repository root:

```bash
uv sync --locked
bash scripts/uv.sh run --locked python3 -m streamlit run app.py
```

For pip, install `requirements.txt` in your activated Python 3.11/3.12 environment,
then run `python3 -m streamlit run app.py`. The sidebar's **Experiment folder**
accepts the original `.` or another completed run such as `runs/reproduction`.
Every selected folder uses the same `reports/`, `models/` and `data/` layout.
The dashboard never trains models or downloads data.

## Sections

- **Overview:** saved official test scores for the model selected on validation
  data, dataset protocol and recorded feature/classifier combinations.
- **Prediction Lab:** select any locally available frozen model, compare models
  on one input, inspect actual inference latency and view the feature vector.
- **Model Comparison:** filter features and classifiers, choose a recorded metric,
  sort the table and inspect interactive score, latency and size comparisons.
- **Error Analysis:** explore saved confusion counts, row-normalized fractions,
  per-class recall and correct/incorrect predictions by true/predicted category.
- **Dataset Explorer:** browse only IDs in the persisted 48,000-image training
  split, with training-category counts, mean images and intensity distributions.
- **How It Works:** follow the feature/classifier pipeline. CNN work is planned;
  this dashboard contains no CNN results.

## Image input and prediction

**Exact benchmark input** requires a 28×28 grayscale PNG in Pillow mode `L`.
It passes the original uint8 values to the existing `extract()` function, which
normalizes once. HOG settings come from the existing feature module, including
its fine-cell representation. The saved model applies its own fitted scaler
where the original pipeline includes one.

**Experimental photo preparation** accepts PNG/JPEG, corrects EXIF orientation,
converts to grayscale and offers padding, centered cropping or stretching to
28×28. Brightness, contrast and inversion are explicit controls. The preview
shows each step before prediction. Photos have not been benchmarked; background,
pose and subject scale can make their predictions unreliable. No changes are
made to the original image or experiment data.

Models load lazily after SHA-256 verification against the corresponding frozen
record. File size and modification time participate in the resource cache key;
a replaced artifact is checked again. Up to four loaded models are cached.
Use trusted experiment folders: checksums establish consistency with the chosen
manifest, and joblib files remain Python serialized objects.

Live latency includes feature extraction and `predict()` with two CPU threads.
It excludes decoding, model loading and the separate alternative-class call.
It is a single measurement, separate from the saved repeated evaluation timing.

Probabilities appear only for estimators implementing `predict_proba()` and are
labelled as model probabilities without an independent calibration study. SVM
`decision_function()` values are labelled decision scores, never confidence
percentages. Alternative-class values are not comparable across models.

## Artifact availability and integrity

`reports/test/results.csv` and `reports/frozen.json` are needed to open a run.
The app handles missing optional files as follows:

| Missing artifact | Available experience |
|---|---|
| `models/*.joblib` | All saved-result sections; prediction gives reproduction guidance |
| `*_confusion.csv` | Selected model's saved confusion image when present |
| `predictions.npz` | Confusion/per-class reports and saved illustrative errors when present |
| Local test archives | Saved labels/indices, plus illustrative errors when present |
| Training archives or `data/split.npz` | Saved class distribution and training figures when present |

Error-image browsing uses the original verified test loader and checks saved
indices/true labels against the local archive. It only displays saved predictions,
without running another test evaluation. Use it to describe frozen results, not
for tuning. Dataset Explorer uses an isolated instance of the existing data
loader and validates the saved split; it never changes the pipeline's shared
root or creates a new split. No test images are used in Dataset Explorer.

Dataset/CSV operations use bounded data caches. Only the active section runs.
Image preparation and prediction state persist in the current Streamlit session;
results are hidden when the input, model selection, artifact or experiment changes.
