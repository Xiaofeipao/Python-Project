# Accounting Records

This directory explains the complete set of ledger files currently present in this workspace. It does not modify, replace or independently reconstruct a ledger.

## Read in this order

1. `ledger_inventory.json`: read-only inventory of every current `.model_usage/*/ledger.json`, with file hashes, evaluation IDs, call/token counts, costs and statuses.
2. `../development/current_results.json`: informational pointers to the currently reported models and their individual run costs.
3. `../development/experiment_log.md`: catalog of the current experiments.
4. `../guides/api_usage.md`: credential setup, actual usage accounting and operational safeguards.

## Source of truth

`.model_usage/` is the operational shared ledger. A run's usage snapshots and `model_calls/` retain supporting evidence; they are not additional independent charges. Do not add a run snapshot to the same task ledger again. Selected-model costs are a subset of the inventory total, not another amount to add.

All currently present records are included in the inventory. Missing token usage, if encountered in a future snapshot, must remain unknown rather than zero. Accounting completeness describes those included records. Keep the inventory scope and underlying evaluation IDs visible when interpreting totals.

The course budget applies per announced evaluation task. Run-specific IDs and a catalog subtotal are not themselves proof of a complete formal task-budget review. Fixed off-peak token costs are distinct from provider invoices, and model deployment eligibility remains a separate question.

## File lifecycle

Do not move, rename or edit live ledgers to reorganize the repository: recovery and request evidence can contain bound paths. Regenerate an inventory by reading source records; never feed it back as a ledger. This snapshot contains no credentials or raw prompts. Detailed raw accounting files remain local and are ignored by Git.

Accounting code remains in `superagent/governance/usage_ledger.py`; its narrow review/reconciliation command is `scripts/api/reconcile_usage.py`. Training control and model selection stay in separate layers.
