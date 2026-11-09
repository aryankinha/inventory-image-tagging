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
- Presentation: the supplied nine-slide presentation was checked against the frozen results and visually reviewed. Editable charts retain their original workbook data. A matching nine-page PDF is included.

A clean GitHub clone in a folder containing spaces and a colon passed the documented
Mac setup with its own new environment. Before any dataset or model download, its
suite passed 40 tests with one expected skip for optional training archives, and
all four dashboard pages opened. Its first prediction then downloaded only pixel
Random Forest from the eight-file release, checked the frozen checksum and
predicted successfully. The public dashboard was also checked: exactly eight
model choices, a working Random Forest prediction, and summary scores of 87.61%
and 0.8747 matching the report. GitHub's macOS compatibility checks passed for the
eight-model implementation.

## Phase 2 completion check, 9 November 2026

The `phase2` branch was compared with the remote before this audit. The PDFs require a two- or three-block CNN, pooling, dropout, early stopping, ablations, a DL literature review, technical validation and comparison with the classical models. The existing study has seven configurations and three seeds each. All 21 local checkpoints reproduced their recorded validation accuracy and macro F1 exactly on the CPU. All three included CNN checkpoints reproduced the 10,000 recorded test predictions exactly.

The missing inference/export and setup steps were added. Both CNN and hybrid correctly predicted the included Shirt example through the command line and the running dashboard. Hybrid forests rebuilt on CPU features agree with 99.94% to 99.98% of the original GPU-feature test predictions; this is a replay check and does not replace the official scores. Forest weights and their generated inference record remain local.

A separate clone in a path containing spaces and a colon used a new Python 3.12 environment with no training/test archives or hybrid weights. The `--phase2` uv setup succeeded, CNN inference worked without downloading data, and the suite passed 59 tests with three dataset-dependent skips. The pip setup also succeeded using Python 3.11 and its own temporary environment. The full local suite passed 62 tests on both Python 3.11 and 3.12. Both notebooks executed using the project interpreter. The four-page IEEE report and seven-slide presentation were rendered and reviewed.

The published Phase 1 dashboard was not redeployed. The Phase 2 dashboard runs locally. Team contributions and formal affiliation remain for the team to fill in; they were not invented.

## Ordinary-photo preparation check, 9 November 2026

The current Python 3.12 suite passed 73 tests after adding background preparation,
crop and rotation. The new photo and existing dashboard tests also passed in the
independent Python 3.11 environment (18 tests). The saved Phase 1 result check
passed for eight models, and all 109 protected evidence files matched their
original hashes. The Phase 2 verification also passed, including three checkpoint
checksums. The running Phase 1 dashboard accepted an internet backpack photograph
and returned Bag. No new dependency was added for this change. The photo checks
are development examples, not an independent accuracy benchmark; failures are
recorded in `docs/PHOTO_INPUTS.md` and local `runs/photo-preparation-fix/`.

A separate source copy, without local model files, training archives or the project
environment, also passed all 18 photo/dashboard tests using Python 3.11. This checks
the new adapter independently of local model/data caches. The earlier complete
setup checks above remain separate from this targeted copy check.

## Second photo check, 9 November 2026

Transparency, small-item scaling and new-picture state fixes passed the full
85-test suite on Python 3.12 and Python 3.11. A separate source copy passed the
30 photo/dashboard tests without project model or data caches. All eight original
Phase 1 score/model checks and 109 protected evidence hashes remained valid.
Eight additional internet photographs were checked; the failures and refusals
are documented in `PHOTO_TEST_ROUND2.md` and are not a new accuracy benchmark.
