# Single-Column Project Report

The English manuscript documents GitHub commit `c835a0d4318d5d37fc83f380d299a8ee36ad7e00`, verified against `origin/main` on 2026-11-27. It uses an A4 single-column research-paper layout, rather than a journal-branded template.

- `report.pdf`: canonical compiled report (9 pages).
- `latex/report.tex`: editable manuscript and inline workflow figure.
- `latex/BUILD.md`: compilation instructions.

The report covers scope, single-agent architecture, feedback-guided planning, general tabular/text/image methods, reliability, final-only test access, resource/accounting policy, evaluation protocol, reproducibility and limitations. Accuracy and completed-trial fields are intentionally blank. Names, student IDs and individual contributions require confirmation. No new training, inference or planner API calls were performed for this revision; existing accounting records were not changed.

Historical generated tables and plots are not imported by this manuscript. Do not run the historical data updater to fill this report automatically: final values must match complete autonomous runs of the same submitted version, policy, model artifacts and logs. Blank results make this an unfinished empirical draft, not a complete final submission.

Previous manuscript/PDF copies are retained locally in `runs/report_backup_20261127/`. Compile twice inside `latex/` and copy the resulting PDF to this directory. Only this canonical report PDF is published; intermediate files and duplicate PDFs are ignored.

References identify model/algorithm papers, official pretrained model cards and software sources; course handouts and this project repository are not bibliography entries. Paragraphs after headings are flush left, while subsequent normal paragraphs use a 1.5 em first-line indent and no added paragraph gap.
