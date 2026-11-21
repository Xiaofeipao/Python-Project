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

## Fixed accounting convention

Registered off-peak prices per million tokens: Flash CNY 1 input / 4 output (effective 2026-09-10); Pro CNY 4.5 input / 13.5 output (effective 2026-08-17). Every input token uses the cache-miss rate. Historical publisher evidence and hashes are preserved; these calculations are distinct from platform invoices. Exact school deployment revisions remain unverified.

Task-wide ledgers retain calls across attempts. Known costs, unresolved worst-case reservations and the next reservation must fit the CNY 20 ceiling. Actual costs use returned tokens, including reasoning tokens counted by the provider. Missing usage is not assumed zero; reservations are not charges.

--provisional-accounting is development-only. It preserves historical unknowns and marks accounting_provisional=true and eligible_for_formal_submission=false. Formal released/unseen runs require reconciled accounting and one persistent absolute SUPERAGENT_USAGE_ROOT. Do not create a new ledger to evade historical usage.

Two consecutive invalid structured plans disable further API planning for that run. Accepted plans reset that counter. Local fallback remains logged and must not be presented as an LLM decision. Ambiguous requests are not blindly retried. Recovery preserves the original ledger and uses local planning for its first restored decision.

Outputs include usage.jsonl, model_calls/, model_calls.jsonl, task_model_calls.json, model_usage.json and summary.json. Inspect current-run IDs when separating new calls from cumulative totals.

Official evidence archives under configs/evidence/ contain Base64-encoded original bytes. The registry SHA-256 applies to decoded bytes, not to the JSON envelope. They are provenance records, not translations or new official publications.

```powershell
Remove-Item Env:HKUST_API_KEY
```
