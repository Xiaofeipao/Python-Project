# AI Usage Declaration

OpenAI Codex assisted with requirements interpretation, implementation, review, debugging, testing and documentation. Development tools and runtime planning models are distinct; project API ledgers do not measure Codex token usage or costs.

Runtime planning can use the school platform's DeepSeek-V4.1-Flash or DeepSeek-V4-Pro deployments. The planner receives data summaries and validation feedback, chooses executable candidates and may propose bounded parameter or feature changes. It does not receive test scores or execute generated code. Runs with the LLM disabled or unavailable are not evidence of successful LLM planning.

Registered off-peak accounting rates per million tokens are CNY 1 input / 4 output for Flash, and CNY 4.5 input / 13.5 output for Pro. All input uses the cache-miss rate. These calculated costs are not school-platform invoices; the course must confirm the accepted accounting convention. Registries and hash-checked original evidence are under `configs/`.

Each run retains actual returned usage, requests/responses, failures and unresolved reservations in usage.jsonl, model_calls/, model_usage.json and summary.json. Unknown usage is not zero. The two historical Flash timeouts were reconciled on 2026-11-23 at their reserved upper bound (CNY 5.24288 each) with actual usage unrecoverable, so their per-task accounting remains incomplete; provisional development results are not accounting-complete formal submissions. A successful prediction file does not establish complete accounting.

Experiment 1 (`coupon-v2-20261123`, 2026-11-23) is the first run reported as an experiment: three recorded Flash calls, CNY 0.018845 in total, accounting complete and no unreconciled reservation. Earlier exploratory runs were archived under `runs/_archive_20261123/` and are no longer listed as experiments; their recorded charges and the two conservative timeout reconciliations stay in the ledger as disclosed historical AI usage and are not attributed to experiment 1.

The exact school deployment revisions have not been independently confirmed. A publisher release date does not prove the revision behind a platform alias. Verify eligibility before formal submission.

All members must understand and verify the code and results, add their actual tools and contributions, and confirm their own declaration. This document does not replace individual confirmation.
