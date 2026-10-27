# Project team

Aryan is the team lead and owns the core modelling in both phases. Course section: B.

| Enrolment number | Name | Email |
|---|---|---|
| 2401020013 | Aryan (lead) | aryan.kinha2024@nst.rishihood.edu.in |
| 2401010161 | Divyanshu Singh | divyanshu.singh2024@nst.rishihood.edu.in |
| 2401010199 | Jatin Kumar Singh | jatin.singh2024@nst.rishihood.edu.in |
| 2401010200 | Jatin Verma | jatin.verma2024@nst.rishihood.edu.in |
| 2401010247 | Kushagra Bhardwaj | kushagra.bhardwaj2024@nst.rishihood.edu.in |

## Phase 1 responsibilities

| Order | Person | Responsibility | Files to be ready to explain |
|---|---|---|---|
| 1 | Aryan | Core modelling and major implementation: pixels and HOG features, the four classifier builders, training and validation selection, the frozen test evaluation, reproducibility, integration, the hosted dashboard release | `src/inventory_tagging/features.py`, `models.py`, `train.py`, `evaluate.py`, `cli.py`, `scripts/` |
| 2 | Divyanshu Singh | Dataset and EDA | `src/inventory_tagging/data.py`, `eda.py`, `reports/data_audit.json`, `reports/figures/` |
| 3 | Jatin Verma | SVM experiment evidence | linear SVM settings, `reports/training/*_trials.json`, `reports/engineering/solver_note.md` |
| 4 | Kushagra Bhardwaj | Evaluation, report and presentation | `reports/test/`, `reports/phase1_findings.md`, `reports/phase1_report/`, `reports/presentation/phase1.pptx`, `docs/literature_review.md` |
| 5 | Jatin Kumar Singh | Dashboard, notebook and setup | `app.py`, `dashboard/`, `notebooks/phase1.ipynb`, `scripts/setup_mac.sh`, `.github/workflows/` |

## Phase 2 responsibilities (proposed)

| Person | Responsibility | Files to be ready to explain |
|---|---|---|
| Aryan | Core modelling and major implementation: the CNN, training with early stopping, the ablation study, final model selection, the hybrid model, the one-time test, reproducibility | `src/inventory_tagging_dl/cnn.py`, `experiments.py`, `cli.py`, `reports/phase2/frozen.json`, `scripts/build_phase2.sh` |
| Divyanshu Singh | Data protocol and augmentation: reuse of the Phase 1 split, flip and shift augmentation, data and split checks | `augment()` in `cnn.py`, `tests/test_phase2.py` (split and augmentation tests) |
| Jatin Verma | Hybrid classifier experiments: Logistic Regression, linear SVM and Random Forest on the CNN features, and the untrained-CNN control | `reports/phase2/hybrid_validation.csv`, `hybrid_selection.json`, `hybrid()` in `experiments.py` |
| Kushagra Bhardwaj | Results, error analysis, report and presentation | `reports/phase2_findings.md`, `reports/phase2/test/`, `reports/phase2_report/`, `reports/presentation/phase2.pptx`, `docs/dl_literature_review.md` |
| Jatin Kumar Singh | Notebook, setup and reproduction checks | `notebooks/phase2.ipynb`, `requirements-phase2.txt`, `README.md` run instructions, a fresh-clone run of `scripts/build_phase2.sh` |

## Contribution record

These tables are planned responsibilities, not a claim that one person wrote every line. Before submission, each member adds a few lines below about what they personally did: code reviewed or written, experiments run, sections written, problems found. Evidence can be Git history, notes or the files listed above. Write only what you did.

| Person | What I actually did (Phase 1) | What I actually did (Phase 2) |
|---|---|---|
| Aryan | | |
| Divyanshu Singh | | |
| Jatin Verma | | |
| Kushagra Bhardwaj | | |
| Jatin Kumar Singh | | |

The Phase 1 upload order was Aryan, Divyanshu Singh, Jatin Verma, Kushagra Bhardwaj, then Jatin Kumar Singh. Uploading prepared files records publication of shared work, not independent authorship of every file. The viva is individual, so every member should understand all stages, not only their own.
