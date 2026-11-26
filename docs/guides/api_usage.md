# API Setup and Accounting

## Credentials and endpoint

The school endpoint is https://aigc-api.hkust-gz.edu.cn/api/v2. The client appends /chat/completions. Remote requests use HTTPS and Bearer authentication. Keep credentials outside Git and outputs.

```powershell
$credentialPath = Read-Host 'Local credential file path'
$env:HKUST_API_KEY = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $credentialPath).Path).Trim()
python scripts/api/probe_hkust_api.py --key-file $credentialPath --output runs/api_probe_new
```

Outputs must be new. The probe is development connectivity evidence, not an evaluation run. If an existing proxy prevents school access, add the school hostname to NO_PROXY for the current process only.

## Enable planning

```powershell
python scripts/workflows/run_released_batch.py --output runs/pro_new --max-trials 10 --llm-registry configs/model_registry.pro.json --llm-model DeepSeek-V4-Pro --key-file $credentialPath --provisional-accounting
```

Flash uses configs/model_registry.flash.json with --llm-model DeepSeek-V4.1-Flash. Raw task/test content is not sent: the planner receives statistics, recent validation feedback and an executable menu. Proposals remain bounded; arbitrary generated code is not executed.

## Complete planning replies

Planning defaults and the two final-policy files now allow 4096 output tokens, following the supplied LLM-fixed package. Providers can spend part of that allowance on reasoning, so the larger limit does not guarantee a usable reply.

The planner rejects missing/invalid choices or messages, empty/non-text content, malformed JSON and JSON values that are not objects. A finish_reason of length is rejected even if the returned text happens to parse as JSON. Existing candidate bounds, duplicate checks and required decision fields still apply. Raw replies and returned usage are recorded before plan validation; rejected replies are not free calls and are not retried automatically.

This integration changes planning response handling only. It does not import the supplied package's fitted classifiers, candidate-ranking results or reported accuracy. Existing best models and their supporting artifacts remain unchanged.

A development check on 2026-11-23 accepted one live Flash plan using the integrated handler: 1017 input tokens, 465 output tokens and CNY 0.002877 at the then-registered off-peak rates. No classifier training occurred. That historical smoke charge is retained for audit. Local evidence is under runs/llm_fix_smoke_20261123/.

## Off-peak accounting convention

Registered user-selected off-peak prices per million tokens: Flash CNY 1 input / 4 output (effective 2026-09-10); Pro CNY 4.5 input / 13.5 output (effective 2026-08-17). Every input token uses the cache-miss rate; cache discounts and free credits are excluded. Historical publisher evidence and hashes are preserved; these calculations are distinct from platform invoices. Exact school deployment revisions remain unverified. Earlier standard/peak ledgers are retained as historical audit records and must not be reported as the final off-peak run.

Task-wide ledgers retain calls across attempts. Known costs, unresolved worst-case reservations and the next reservation must fit the CNY 20 ceiling. Actual costs use returned tokens, including reasoning tokens counted by the provider. Missing usage is not assumed zero; reservations are not charges.

--provisional-accounting is development-only. It preserves historical unknowns and marks accounting_provisional=true and eligible_for_formal_submission=false. Formal released/unseen runs require reconciled accounting and one persistent absolute SUPERAGENT_USAGE_ROOT. Do not create a new ledger to evade historical usage.

Complete accounting alone never certifies formal eligibility. Current summaries
also distinguish formal_runtime_checks_passed from submission_review_required.
There is one runtime planner identity. Fitting workers receive no planner API
credential and cannot make calls on behalf of another runtime agent.

After two consecutive invalid structured plans, the next decision may make one shorter repair request using a listed seed and empty overrides. It consumes one separately reserved and recorded API call. If repair fails, planning stays local. Success resets the consecutive failure counter, but the repair allowance is never renewed within the run. Recovery preserves that allowance, the original ledger and the local first restored decision. Unknown-usage timeouts are never automatically replayed. Local fallback is logged and must not be presented as an LLM decision.

## Reconciling unrecoverable usage

A read timeout proves the request reached the provider but not that it generated nothing, so the missing token usage cannot be recovered locally. `scripts/api/reconcile_usage.py` closes such a call without inventing token counts or a zero-cost success.

```powershell
# Dry run: list unrecorded calls and the current summary
python scripts/api/reconcile_usage.py --ledger-dir .model_usage/<task-hash> --evaluation-id <task-id>

# Apply: book the worst-case reservation as a stated upper bound
python scripts/api/reconcile_usage.py --ledger-dir .model_usage/<task-hash> --evaluation-id <task-id> `
  --call-id <call-id> --basis "Why usage is unrecoverable" --evidence "<original request path>" --apply
```

`--apply` requires a basis and evidence; an evidence path must exist. The record books the call's worst-case reservation (CNY 1.06496 for a 4096-output Flash request under off-peak 1/4 pricing) as an explicit upper-bound charge, keeps input/output token counts absent, and stores the basis, evidence path, author and timestamp in `ledger.json`, `calls.jsonl` and `<ledger>/reconciliation.json`. `unreconciled_reserved_rmb` drops to zero while `conservative_reconciled_cost_rmb` and `charged_total_cost_rmb` report the bounded charge. Older reservations remain at the amount recorded when the call was made.

`accounting_complete` deliberately stays false: the charge is bounded, the actual usage is not known. A reconciled task still refuses new model requests under the default policy and remains ineligible for formal submission; only provider-supplied usage can close the gap. This is a description of the tool, not a statement that any current ledger contains such a record. The current record set and totals are indexed in ../accounting/ledger_inventory.json.

Outputs include usage.jsonl, model_calls/, model_calls.jsonl, task_model_calls.json, model_usage.json and summary.json. Inspect current-run IDs when separating new calls from cumulative totals.

Official evidence archives under configs/evidence/ contain Base64-encoded original bytes. The registry SHA-256 applies to decoded bytes, not to the JSON envelope. They are provenance records, not translations or new official publications.

```powershell
Remove-Item Env:HKUST_API_KEY
```

For current record counts, hashes and totals, use ../accounting/ledger_inventory.json. Per-run usage snapshots are views of those calls, not additional charges. Do not relocate active ledgers or use the documentation inventory as a spending ledger.
