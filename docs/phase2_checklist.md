# Phase 2 requirements and evidence

Requirements come from the Project 11 description (Project Ideas p.12: two or three block CNN with pooling, dropout and early stopping; official split with validation from training only; accuracy, macro F1, per-class recall, parameter count and inference time) and the Phase 2 rubric (Instructions p.2). Marks and approval remain with the faculty.

| Requirement | Evidence |
|---|---|
| Architecture logic | `src/inventory_tagging_dl/cnn.py`; report section IV and `reports/phase2/figures/architecture.png`; two blocks with pooling, why not three (28, 14, 7), ablation tests a third block |
| Hybrid approach with a reason and shown value | `hybrid()` in `experiments.py`; report section VII; CNN's 64 features into Phase 1 classifiers; compared with the CNN's own last layer and an untrained-CNN control (`reports/phase2/hybrid_selection.json`) |
| Ablation studies | 7 versions x 3 seeds, `reports/phase2/ablation_summary.csv`, `ablation_runs.csv`, `figures/ablation.png`; report section VI |
| Deep learning literature review | `docs/dl_literature_review.md` (10 sources, citations checked); report section II |
| Dataset and regularisation | Same 48,000/12,000/10,000 split as Phase 1; batch norm, dropout, augmentation, weight decay, early stopping (patience 5, best epoch restored); report section V |
| Fair comparison | Same official test pictures as Phase 1; choices frozen in `reports/phase2/frozen.json` before the test; test run once (`reports/phase2/test/started.json`); paired bootstrap |
| Accuracy, macro F1, per-class recall | `reports/phase2/test/results.csv`, `per_class_recall.csv`, `comparison.csv` |
| Parameter count and inference time | `reports/phase2/test/summary.json` (`costs`): 429,450 weights; CPU timing with two threads like Phase 1 |
| Shirts, T-shirts, pullovers confusion; error analysis | `test/upper_body_confusions.csv`, `figures/error_examples.png`, `figures/confusion_cnn.png`, `reports/phase2_findings.md` |
| Technical validation | Three seeds per setting; `tests/test_phase2.py` (model, augmentation, early stopping, split); `tests/test_phase2_evidence.py` (scores recomputed from saved predictions, checkpoint hashes, frozen choices), and `test_phase2_completion.py` (verified inference and UI) |
| Theoretical rigor | Report sections IV and V (convolution, pooling, batch norm, dropout, loss, early stopping); `docs/phase2_viva_guide.md` |
| Classifier notebook | `notebooks/phase2.ipynb`, executed; reads saved results, no retraining |
| LaTeX/IEEE report | `reports/phase2_report/main.tex` and compiled `main.pdf` (4 pages) |
| Presentation | `reports/presentation/phase2.pptx` (7 slides with speaker notes) and `phase2.pdf` |
| GitHub and code quality | `src/inventory_tagging_dl/` contains the CNN, experiments, commands and verified inference; PyTorch uses the optional `phase2` uv group and exported `requirements-phase2.txt`; both phases are included on main; hosted prediction depends on installed optional dependencies; one command to rebuild: `bash scripts/build_phase2.sh` |
| Individual viva | `docs/phase2_viva_guide.md` and each member's own practice |
| Five same-section members; contribution record | Names in `docs/team.md`. Section B is recorded; formal affiliation and each member's actual contribution still need confirmation before submission |

## Choices made after seeing validation results

- Light dropout (0.2) was added after dropout 0.4 hurt in the first runs.
- The final CNN is the best version that uses dropout, because the brief asks for it. The best version overall (no dropout) is 0.0018 higher.
- Neither decision used test pictures.

## Not done on purpose

Installing PyTorch on the free hosted deployment, pretrained or residual networks, learning-rate schedules, other augmentations, and a search beyond the seven versions. Each adds machinery beyond what the brief needs.

## Completion checks added on 9 November 2026

- Verified CNN and hybrid image prediction: `src/inventory_tagging_dl/predict.py`, CLI `predict`, and the local dashboard's Phase 2 page. Checksums are checked before deserialization. Photo preparation reuses the Phase 1 interface and is labelled experimental.
- `export` saves the frozen hybrid classifiers for repeated predictions. Large forest files stay local. It never chooses settings from test scores.
- `scripts/verify_phase2.py` independently checks histories, selection, test scores, recall, confusions and checkpoints. `--replay` compares live CPU predictions with recorded test predictions.
- Frozen experiment folders reject further ablation, selection and hybrid training. An interrupted final test cannot be restarted in place.
- Both setup methods support `--phase2`; CI installs PyTorch and exercises the deep-learning tests.
- The original results are preserved. Local CPU hybrid exports agree with the recorded GPU-feature run on 99.94% to 99.98% of test predictions. This replay is separate from the official reported experiment.
