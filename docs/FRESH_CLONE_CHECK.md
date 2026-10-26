# Phase 1 verification

The active project contains exactly eight combinations: pixels and HOG, each with
Logistic Regression, k-NN, Random Forest and linear SVM.

A fresh full run was completed in a separate folder using the official Fashion-MNIST
archives and the locked Python 3.12 environment on an Apple silicon Mac. All models
used 48,000 training and 12,000 validation images. The validation-selected model
was pixel Random Forest, fixed before the final 10,000-image test.

Its test accuracy is 87.61% and Macro F1 is 0.874729334929437. All eight model files
and prediction arrays exactly match the previous eight baseline models. Training
and prediction timings were measured again for this run. The current reports,
figures, notebook and presentation use this eight-model run.

Model checksums, saved predictions, category scores, confusion matrices, timing
repetitions, source hashes and the saved split are checked by the verification
scripts. The hosted app uses only the eight matching published model files.

The previous repository snapshot is preserved in local archives outside this
repository, and earlier commits remain in Git history. Removed experiments are
not part of the active Phase 1 presentation or reproduction commands.

A new clone can start the dashboard without training data. Predictions fetch a
requested published model and verify its checksum. To train the required eight
models yourself, use `bash scripts/build_models.sh`; the default output folder is
`runs/phase1`. The original saved results are protected from retraining in place.

## Checks completed

- Python 3.12: 41 tests passed.
- Python 3.11: 41 tests passed in an independent environment.
- Dashboard: all four pages and all eight saved predictors passed on both versions.
- Notebook: all 18 cells executed successfully in memory.
- Report: three PDF pages compiled and visually checked.
- Presentation: all 13 slides visually checked; editable tables and chart passed package checks.

A clean GitHub clone in a folder containing spaces and a colon passed the documented
Mac setup with its own new environment. Before any dataset or model download, its
suite passed 40 tests with one expected skip for optional training archives, and
all four dashboard pages opened. Its first prediction then downloaded only pixel
Random Forest from the eight-file release, checked the frozen checksum and
predicted successfully. The public dashboard was also checked: exactly eight
model choices, a working Random Forest prediction, and summary scores of 87.61%
and 0.8747 matching the report. GitHub's macOS compatibility checks passed for the
eight-model implementation.
