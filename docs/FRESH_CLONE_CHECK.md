# Fresh-clone verification — 25 October 2026

The Phase 1 additions were checked in separate Git checkouts and new environments.
No virtual environment, fitted model files or raw dataset archives were copied from the working project.
The core training/evaluation modules stayed byte-identical throughout the update.

## Environments actually exercised

Tests ran on **macOS 26.5.1, Apple silicon (arm64)**.

| Check | uv environment | pip environment |
|---|---|---|
| Python | 3.12.13 | 3.11.16 |
| NumPy / SciPy | 2.5.4 / 1.18.1 | 2.4.6 / 1.17.1 |
| scikit-learn | 1.9.1 | 1.9.1 |
| Streamlit / Plotly | 1.66.1 / 6.9.0 | 1.66.1 / 6.9.0 |
| Setup and saved-evidence checks | Passed | Passed |
| Final test suite with prepared training data | **40 passed** | **40 passed** |
| Dashboard sections checked | 6 | 6 |
| Real reproduced models verified and predicted | 11 | 11 |
| Two models compared on one image in the UI | Passed | Passed |

Different NumPy/SciPy versions are the Python-version-specific choices already
pinned by the lockfile and its requirements export, not uncontrolled upgrades.
The environments were newly created; package download caches are not copied
virtual environments. Module paths were checked to point to the independent
checkout and its new site-packages, not the original project.

The principal test checkout was `/private/tmp/inventory-phase1-fresh-check`,
initially cloned from GitHub. The new tracked commits were then fast-forwarded
into it for pre-push testing. A second checkout,
`/private/tmp/inventory clone: path check`, tested spaces and a colon in the
project path with another newly created uv environment.

## Clean clone, before models or full data exist

The second checkout retains no `data/` or `models/` folder. Its final code passed
**39 tests with one expected skip**: the optional test comparing the sample PNGs
against full training archives cannot run when those archives are absent. The
same test passes after the documented data-preparation command.

`check_dashboard.py` passed all six sections and checked all ten gallery
categories with **zero fitted models present**. The app displayed saved results,
previews and sample images while disabling new predictions and explaining how to
enable them. It did not substitute fake or reference model files. The setup
helper passed even from the path containing spaces and a colon.

## Complete real Phase 1 reproduction

The principal clone downloaded the official archives using the existing verified
loader and ran the complete sequence:

1. Training-only preparation and persisted seed-42 split.
2. Training-only EDA.
3. Eight pixel/HOG classical combinations, using all 48,000 training images.
4. Three additional validation-selected RBF combinations.
5. Frozen-model official test evaluation.

Its generated files are under `runs/reproduction/`, separate from the original
reports. The selected model was `hog_fine_rbf_svm`, chosen from validation data.
It scored **90.87%** test accuracy and **0.9083955808458445** Macro F1.

All eleven reproduced models matched the original saved accuracy and Macro F1
values on this machine. Their saved per-image predictions were also identical
for all eleven models. Timings are measurements from the new run and are not
substituted into the original report.

The expanded result verifier passed for the reproduction: scores, per-class
metrics, timing summaries, complexity/size records, source hashes, all eleven
model hashes, the saved split and available archive hashes. Both new environments
loaded and ran these actual models. Browser model selection and comparison also
worked against `runs/reproduction`.

Running `bash scripts/build_models.sh` again printed that the run was already
complete and verified, then exited without repeating training or test evaluation.
A permanent regression test also checks that an interrupted final-test marker is
preserved and that the original project folder cannot be overwritten.

## Notebook and browser checks

The documented training-data preparation command succeeded in the clone and
loaded no official test images into the original-root data folder. All **18
notebook cells** then executed in the newly installed project kernel. The demo
used the clone's own model, verified its checksum and split, and predicted a
training example. Execution was in memory; it did not rewrite the notebook.

The real Streamlit server was started through `start_dashboard.sh` with the fresh
pip environment. Browser checks covered:

- Navigation, known-category gallery selection and image previews without models.
- The saved paired comparison, without needing model or dataset files.
- Transparent-picture preparation and readable blank-picture rejection.
- Choosing `runs/reproduction`, changing the predictor and comparing two models.
- A successful prediction and correctly loaded picture previews.

The numerical helper tests additionally cover exact benchmark pixel preservation,
alpha/background equivalence, very narrow pictures, invalid inputs, real feature
vectors, changed-model checksum rejection, missing optional files and partial
final-result tables.

## Preserving the original experiment

The artifact verifier passed for **128 versioned evidence/sample files**. Its
manifest deliberately excludes ignored caches and presentation build files; a
regression test ensures every required path is Git-versioned. The initial clean
clone caught and led to correction of an earlier manifest that included caches.

A separate before/after hash comparison found **no changes in 162 pre-existing
local report, frozen-pipeline and model files**. The original report PDF, slides,
frozen choices and test results were not rewritten. New paired statistics are
stored separately under `reports/additional_analysis/` and labelled as additional
descriptive analysis, not model selection or a new evaluation.

## Limits of this verification

This establishes the tested Apple silicon Mac paths and the two Python versions
above. An Intel Mac and other macOS releases were not executed locally.
GitHub's fresh-run checks cover its configured macOS runners separately. Model
files and full datasets remain excluded from Git: a new clone needs the documented
model-building step for new predictions. No CNN or other Phase 2 implementation
was added.
