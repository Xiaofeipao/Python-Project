# English IEEE Transactions-style manuscript

Compile from this directory twice. Generated tables, data and figures are already included, so ordinary compilation does not require access to local runs:

```text
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Alternatively upload this entire directory to Overleaf and select pdfLaTeX. IEEEtran.cls is included with its original license header. Keep generated.tex and figures/current_validation.png. Standard amsmath, graphicx, booktabs, url and hyperref packages are required. No BibTeX run or API key is needed.

To refresh data, run `python scripts/reporting/update_report_data.py` from the repository root with the original local run artifacts available. This reporting-only helper requires matplotlib; it verifies ledger/source hashes and reads existing JSON, without training, inference or API calls. It regenerates generated.tex, report_data.json and the plot. Compile twice afterward.

Copy the compiled PDF to ../report.pdf. Only that canonical PDF is tracked; intermediate and duplicate PDFs are ignored. This development manuscript uses the November 24 same-runtime pair and explicitly separates cross-version frontier results. Confirm actual authorship and deployment eligibility before formal submission.
