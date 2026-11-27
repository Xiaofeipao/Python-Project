# Building the Single-Column Manuscript

The source uses the standard `article` class with A4 paper, 11-point Times-style text, 25 mm margins and one column. It is a research-paper style, not a named journal's official template.

From `docs/report/latex/`, compile twice:

```text
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Then copy `report.pdf` to `../report.pdf`, the canonical tracked report. On PowerShell:

```powershell
Copy-Item -LiteralPath report.pdf -Destination ../report.pdf -Force
```

Alternatively upload `report.tex` to Overleaf and choose pdfLaTeX. Standard packages used include fontenc, mathptmx, amsmath, amssymb, graphicx, booktabs, array, tabularx, geometry, microtype, fancyhdr, enumitem, titlesec and hyperref. The workflow diagram and references are embedded in the source. No BibTeX, external image, API key or model execution is required.

The examined implementation is pushed commit `c835a0d4318d5d37fc83f380d299a8ee36ad7e00`. Empirical accuracy fields remain blank by request. Complete them only from matching autonomous runs and corresponding prediction/model/log artifacts. Confirm author information separately.

Legacy `generated.tex`, `report_data.json`, figures and `IEEEtran.cls` remain historical assets; this manuscript does not import them. The historical `scripts/reporting/update_report_data.py` helper does not populate the new blank result table and should not be used to overwrite this report's evidence scope.

Paragraph formatting follows the standard article convention: first paragraphs after headings are flush left; later normal paragraphs have a 1.5 em first-line indent. Extra paragraph spacing is zero. The embedded bibliography contains model/algorithm and software sources.
