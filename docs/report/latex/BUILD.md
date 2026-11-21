# English IEEE Transactions-style manuscript

Compile from this directory twice:

```text
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Alternatively upload this entire directory to Overleaf and select pdfLaTeX. IEEEtran.cls is included with its original license header. Keep figures/validation_curves.png. Standard amsmath, graphicx, booktabs, url and hyperref packages are required. No BibTeX run or API key is needed.

Copy the compiled PDF from this directory to ../report.pdf after building. Only that canonical PDF is tracked; intermediate files and duplicate PDFs are ignored. This manuscript describes the 2026-11-16 frozen development experiment, not every later source change. It is a course-project development draft, not an IEEE publication. Complete authorship and unresolved accounting before formal submission.
