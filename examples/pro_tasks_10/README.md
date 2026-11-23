# Pro ten-trial development experiment

Both tasks completed ten successful validation trials using the same frozen runtime and policy. `progressive_output=false`; test data was used only after final selection. Reloaded predictions, class probabilities and known token costs were checked. Full models and frozen source remain locally in `runs/_archive_20261123/pro_tasks_20261116/`.

- Coupon: test accuracy 79.0698% (2006/2537), validation best 75.9606%. Validation reached its best by trial five and did not improve through ten. Current-run known cost: CNY 0.2439945.
- Headlines: test accuracy 66.8626% (2161/3232), validation best 69.1489%. Best validation was 69.0426% through eight trials and improved by trial ten. Current-run known cost: CNY 0.2704455.

Each task made nine successful HTTP/API requests to `DeepSeek-V4-Pro` with actual usage returned. Four responses were accepted as structured decisions; five hit the 2048-token output limit (`finish_reason=length`) and triggered deterministic fallback. Do not attribute those fallback decisions to the LLM. There were no API timeouts in this run. A future Pro planning experiment should test a larger output allowance rather than invent missing reasoning.

The official historical off-peak rates are CNY 4.5/M input (cache miss) and CNY 13.5/M output. Costs are accounting figures, not school invoices. The exact school deployment revision is not independently attested.

Historical Flash timeouts were reconciled on 2026-11-23 at their reserved upper bound: each task now carries an explicit CNY 5.24288 conservative charge instead of an unresolved hold, in addition to known cumulative costs (CNY 0.3014195 and CNY 0.2839105). Those charges are upper bounds, not actual usage, so task-wide actual totals remain unknown and accounting_complete stays false; the reconciliation audit is under each task's `.model_usage` directory. Formal released/unseen execution still rejects incomplete accounting, and these results remain ineligible for formal submission without provider-supplied usage for the timeouts.

The reused test sets make these development comparisons, not independent final evaluations. More trials improved the headlines short-run score but did not improve the coupon short-run score. Historical best coupon (79.2274%) and headlines (66.8626%) models are retained separately; no test-based replacement or mixing of versions is claimed as a unified formal submission.

`verification.json` preserves per-task prefix comparisons and accounting status. Logs are real, with absolute workspace paths redacted; they include earlier task attempts and must not be mistaken for this run alone.
