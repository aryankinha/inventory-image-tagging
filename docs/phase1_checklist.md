# Phase 1 requirements and evidence

The requirements come from the supplied project description (Project Ideas p.12) and Phase 1 instructions (Instructions p.2). The table lists where to find the work for each requirement. Marks and approval remain with the faculty.

| Requirement | Evidence |
|---|---|
| Literature review | `docs/literature_review.md`; report references |
| Dataset quality and EDA | `reports/data_audit.json`, `reports/class_counts.csv`, five training EDA figures, notebook |
| Normalize 28x28 images, preserve labels/orientation | `features.py`, `data.py`, normalization/orientation tests |
| Pixel and HOG feature engineering | `features.py`; 784 pixel values and a 324-dimensional HOG descriptor |
| LR, k-NN, RF, SVM for both representations | Eight core experiments; `reports/validation_results.csv` and per-model trial files |
| Theory and model justification | `docs/viva_guide.md`, report methods, notebook |
| Official test split; validation only from training | `data.py`; persisted seed-42 IDs; `reports/frozen.json` |
| Accuracy, macro F1, class recall | `reports/test/results.csv`, `per_class.csv`, per-model classification reports |
| Parameter/model complexity and inference time | Per-model complexity JSON; common CPU timings including HOG; single/batch columns |
| Shirts/T-shirts/pullovers confusion | Confusion matrices, upper-body confusion CSV, error examples |
| Classifier notebook | `notebooks/phase1.ipynb`, executed and checked |
| Concise error analysis | `reports/phase1_findings.md`, notebook, report |
| GitHub/code quality | Main linked checkout, `uv.lock`, tests, source modules, README, attribution, reproducibility instructions |
| LaTeX/IEEE report | `reports/phase1_report/main.tex`, compiled three-page `main.pdf`, and successful native compilation |
| Presentation | `reports/presentation/phase1.pptx`: five slides with speaker notes and a Mermaid prediction flow |
| Individual viva | `docs/viva_guide.md` and each member's own practice |
| Exactly five same-section members; contribution evidence | Five identities are in `docs/team.md`; add section/formal affiliation and truthful roles/commit contributions before submission |

CNN training and experiments on hybrid models and regularization belong to Phase 2. Before submission, each member should practise the viva and record their actual contribution. Attendance and individual understanding remain part of the assessment.
