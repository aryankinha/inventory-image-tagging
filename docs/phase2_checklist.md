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
| Technical validation | Three seeds per setting; `tests/test_phase2.py` (model, augmentation, early stopping, split); `tests/test_phase2_evidence.py` (scores recomputed from saved predictions, checkpoint hashes, frozen choices) |
| Theoretical rigor | Report sections IV and V (convolution, pooling, batch norm, dropout, loss, early stopping); `docs/phase2_viva_guide.md` |
| Classifier notebook | `notebooks/phase2.ipynb`, executed; reads saved results, no retraining |
| LaTeX/IEEE report | `reports/phase2_report/main.tex` and compiled `main.pdf` (4 pages) |
| Presentation | `reports/presentation/phase2.pptx` (7 slides with speaker notes) and `phase2.pdf` |
| GitHub and code quality | New package `src/inventory_tagging_dl/` (3 small files); PyTorch kept in `requirements-phase2.txt` so Phase 1 and the hosted dashboard are unchanged; one command to rebuild: `bash scripts/build_phase2.sh` |
| Individual viva | `docs/phase2_viva_guide.md` and each member's own practice |
| Five same-section members; contribution record | Names in `docs/team.md`. Course section, formal affiliation and each member's actual contribution are still to be added before submission |

## Choices made after seeing validation results

- Light dropout (0.2) was added after dropout 0.4 hurt in the first runs.
- The final CNN is the best version that uses dropout, because the brief asks for it. The best version overall (no dropout) is 0.0018 higher.
- Neither decision used test pictures.

## Not done on purpose

CNN in the live dashboard, pretrained or residual networks, learning-rate schedules, other augmentations, and a search beyond the seven versions. Each adds machinery beyond what the brief needs.
