# Pro ten-trial development experiment

Both tasks completed ten successful validation trials using the same frozen runtime and policy. `progressive_output=false`; test data was used only after final selection. Reloaded predictions, class probabilities and known token costs were checked. Full models and frozen source remain locally in `runs/pro_tasks_20261116/`.

- Coupon: test accuracy 79.0698% (2006/2537), validation best 75.9606%. Validation reached its best by trial five and did not improve through ten. Current-run known cost: CNY 0.2439945.
- Headlines: test accuracy 66.8626% (2161/3232), validation best 69.1489%. Best validation was 69.0426% through eight trials and improved by trial ten. Current-run known cost: CNY 0.2704455.

Each task made nine successful HTTP/API requests to `DeepSeek-V4-Pro` with actual usage returned. Four responses were accepted as structured decisions; five hit the 2048-token output limit (`finish_reason=length`) and triggered deterministic fallback. Do not attribute those fallback decisions to the LLM. There were no API timeouts in this run. A future Pro planning experiment should test a larger output allowance rather than invent missing reasoning.

The official historical off-peak rates are CNY 4.5/M input (cache miss) and CNY 13.5/M output. Costs are accounting figures, not school invoices. The exact school deployment revision is not independently attested.

Historical Flash timeouts remain unresolved. Each task still holds CNY 5.24288 in worst-case reservations, in addition to known cumulative costs (CNY 0.3014195 and CNY 0.2839105). Those reservations are not actual charges; actual task-wide totals remain unknown. This explicitly enabled development-only mode conservatively includes unknown reservations when checking the RMB 20 limit. Formal released/unseen execution still rejects unresolved usage. These results are not eligible for formal submission until accounting is reconciled.

The reused test sets make these development comparisons, not independent final evaluations. More trials improved the headlines short-run score but did not improve the coupon short-run score. Historical best coupon (79.2274%) and headlines (66.8626%) models are retained separately; no test-based replacement or mixing of versions is claimed as a unified formal submission.

`verification.json` preserves per-task prefix comparisons and accounting status. Logs are real, with absolute workspace paths redacted; they include earlier task attempts and must not be mistaken for this run alone.
