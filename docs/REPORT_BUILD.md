# Report and presentation files

`reports/phase1_report/main.tex` uses the official IEEEtran class. Its original class file and license notice are kept alongside it, downloaded from https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls on 23 October 2026. The project uses the built-in LaTeX editor for source editing and compilation diagnostics. If you already have TeX installed, compiling `main.tex` from its directory is another option; do not use an unrelated working directory because images/results are relative.

`results.tex` and the report's copied images are generated from actual test evidence by `scripts/make_findings.py`. The narrative, methods, limitations and references are in `main.tex`. If regenerating the notebook, `scripts/make_notebook.py` intentionally writes a clean unexecuted source; execute it in the project environment to refresh output cells.

`reports/presentation/phase1.pptx` contains five editable slides with speaker notes. `phase1.pdf` is the matching reading copy. Its figures and result values are derived from the same experiment tables. PowerPoint can edit the delivered file directly. The deck authoring tools run separately from the Python classifier environment.

Team identities are in `docs/team.md`. Add the course section, formal affiliation and actual member contributions before submission. Each member should also practise explaining the work.

The prediction flow is kept in `reports/presentation/prediction_flow.mmd` and `prediction_flow.svg`. The slide uses editable boxes and arrows with the Mermaid layout. If the dashboard steps change, update the diagram source and the slide together. The presentation covers the task, data and methods, all eight results, common mistakes, and the dashboard workflow. Detailed formulas and settings remain in the report.
