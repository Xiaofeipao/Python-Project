# Report Manuscript

- report.pdf: the only published report PDF.
- latex/report.tex: editable English manuscript.
- latex/IEEEtran.cls: document class with its original license header.
- latex/figures/: report figures.
- latex/BUILD.md: build commands.

The manuscript was updated on 2026-11-24 and describes the current code. Its primary same-runtime pair is coupon-v6-20261124 / headlines-v6-20261124 (78.8727% / 69.7092%). The report generator verifies both fingerprints against the repository runtime. The higher cross-version Coupon development result is explicitly separate from the primary evaluation.

The five-page English IEEE-style report covers architecture, autonomy, data, complete primary trial tables, current results, costs, recovery, reproducibility, limitations and submission readiness. The supplied requirements mandate report.pdf and content coverage, but do not specify a page limit, font or citation style. All current accounting records are summarized: 15 ledgers, 133 calls, CNY 1.319903. The primary pair costs CNY 0.097397; selected-run costs are not added to the inventory again.

Authorship, deployment eligibility, test reuse and post-hoc cross-run comparisons remain explicit limitations. The document is still a development report, not proof of completed formal compliance. No models, logs or ledger entries were changed, and no training/API calls were made to update it. Previous manuscript/PDF copies are retained locally under runs/report_backup_20261124/.

Compile inside latex/ and copy the result to this directory as report.pdf. Intermediate files and duplicate PDFs are ignored. Presentations and pre/ are not published.
