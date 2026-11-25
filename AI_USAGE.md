# AI Usage Declaration

OpenAI Codex assisted with interpretation, implementation, code review, debugging, testing and documentation. Development tools are separate from the runtime planner; runtime API ledgers do not measure Codex usage or costs.

The runtime can use configured Qwen/DeepSeek models, subject to registered version and price constraints. Current experiments use DeepSeek-V4.1-Flash or DeepSeek-V4-Pro. The planner receives data summaries and validation feedback, selects executable candidates and may propose bounded changes. It does not receive test scores or execute arbitrary generated code. Local fallback is not an LLM decision.

Registered off-peak rates per million tokens are CNY 1 input / 4 output for Flash and CNY 4.5 input / 13.5 output for Pro. Inputs use the cache-miss rate. Calculated costs are not provider invoices; the course must accept the pricing convention. Publisher originals remain losslessly archived with verifiable decoded hashes.

The accounting record set is the ledger files currently present under `.model_usage/`. Its read-only inventory, scope and file hashes are documented in [Accounting Records](docs/accounting/README.md). A run's snapshots and the shared ledger describe the same calls and must not be double-counted. No token usage or charge is invented when records are missing.

[Current results](docs/development/current_results.json) and the [experiment catalog](docs/development/experiment_log.md) own experiment/model details. The reported Coupon and Headlines models use different frozen runtime hashes; these development frontier results must not be described as a single frozen submission experiment or independent unbiased final evaluation. Model revision eligibility remains to be verified separately from cost arithmetic.

Every member must understand and verify the code and results, declare actual tools and contributions, and confirm their own AI declaration. This document does not substitute for individual confirmation.
