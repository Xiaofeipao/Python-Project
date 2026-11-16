# Released-task API short run

Real run on 2026-11-16, using the same frozen SuperAgent and policy for both tasks. Each task ran two candidates (a baseline and one API-selected method). This checks the complete API/train/predict/evaluate/accounting workflow; it is not the planned 40-trial search or an independent final evaluation. Test sets were previously reused during development.

- Coupon: 2010/2537 correct, accuracy 0.7922743397713835; cross-boost blend. One real API call in this run, cost CNY 0.002521. Task-wide recorded cumulative cost: CNY 0.018892.
- Headlines: 2135/3232 correct, accuracy 0.6605816831683168; ordered character-feature logistic regression. One real API call in this run, cost CNY 0.002649. Task-wide recorded cumulative cost: CNY 0.002649.

Requested and returned API identity: `DeepSeek-V4.1-Flash`; exact deployment revision remains unverified. Costs use the fixed official off-peak policy: CNY 1/M input tokens and CNY 4/M output tokens, without cache discounts. They are accounting values, not school platform invoices.

`usage.jsonl` includes earlier attempts for the same task; do not count every row as a successful request in this run. A historical headlines DNS failure was reconciled as not sent using its original getaddrinfo error log; the original record and evidence hash remain in the local reconciliation audit. Unknown HTTP failures from separate connectivity probes were not represented as zero-cost successes or merged into task results.

`verification.json` records frozen runtime identity, matching reloaded predictions and accounting totals. Structured API decisions are preserved in `planning_events.jsonl`. Output paths are redacted; full models, original requests/responses and frozen source remain in `runs/flash_short_complete_20261116/` locally. These short-run scores do not supersede historical best models.
