# Additional comparison of the saved Phase 1 answers

This analysis was added after the original experiment. It reads saved predictions
only. It does not train models, choose settings, reopen test images or replace the
original final evaluation. The two model names come from the frozen validation
records: Fine HOG + RBF SVM is the overall choice, and pixel Random Forest is the
choice among the eight core combinations.

| Outcome on the same 10,000 final-test images | Images |
|---|---:|
| Both models correct | 8,462 |
| Only Fine HOG + RBF SVM correct | 625 |
| Only pixel Random Forest correct | 299 |
| Both models incorrect | 614 |

The accuracy difference is **3.26 percentage points** in favour of Fine HOG + RBF
SVM. A paired image bootstrap with 2,000 resamples and seed 42 gives a 95% interval
of **2.64 to 3.84 percentage points**. Image pairs stay together in each resample.
This describes variation in sampled test images for these fixed models. It does
not include retraining or training-seed variation, and it does not isolate the
separate effects of features, model family or tuning.

The machine-readable JSON records hashes of the frozen manifest and prediction
file so the inputs can be checked. Regenerate it without changing either input:

```bash
bash scripts/uv.sh run --locked python3 scripts/compare_saved_models.py
```

The dashboard also allows descriptive comparisons of other saved pairs. None of
these comparisons is used to select or retune a model.
