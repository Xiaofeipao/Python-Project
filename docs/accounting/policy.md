# Course Accounting Policy

Source: the instructor clarification supplied by the project owner on
26 November 2026. This document records that supplied interpretation; it does
not invent provider usage or replace original call evidence.

## Budget scope

The RMB 20 cap applies to each final task evaluation, including its resumed
attempts. Development experiments do not consume that course budget. Known
development costs are still recorded for transparency. Existing pricing entries
are unchanged. Platform invoices and quotas are separate from course accounting.

New runtime ledger directories use SHA256(scope + ':' + evaluation_id), where
scope is development or final_evaluation. Released and unseen formal execution
use the final_evaluation scope. The ledger also verifies its scope. Restarting
or changing output directories does not reset an evaluation's budget.

Historical ledgers lack some newer scope metadata. They are retained verbatim,
not silently assigned to final evaluation or migrated into new scoped ledgers.
Historical development totals must not be presented as final-task spending.

## Failed requests without token usage

Record the local call ID, request time, completion/failure time, error, HTTP
status and provider request ID when available. Preserve returned error evidence.
Token counts and actual cost remain null with status usage_unknown. A qualifying
network failure or timeout has budget_excluded_reason failed_request_without_usage
and course_budget_cost_rmb 0. This is an explicit budget exclusion, not zero
actual usage. No upper-bound failure charge or write-back reconciliation is used.

Pre-request reservations are temporary concurrency/budget safety checks. They
are released for these failed requests and do not become estimated failure fees.
The planner continues with local fallback rather than automatically replaying
the failed request. Later decisions can make separately logged calls.

If an HTTP failure returns valid token counts, those known tokens are priced
normally. A successful response with absent, malformed or inconsistent usage is
not automatically exempt. Pending requests are not treated as failed requests.

## Summary fields

- actual_recorded_cost_rmb: sum of calls with actual recorded usage.
- course_budget_cost_rmb: qualifying known costs for final evaluation, zero for
  development; not a provider invoice.
- token_usage_complete: false while any actual usage remains unknown.
- accounting_complete: whether records are accounted for under the course rule;
  a documented exempt failure can satisfy this even with unknown tokens.
- actual_total_cost_rmb: null when actual usage is incomplete.
- excluded_failed_call_count: documented failed requests excluded from the cap.

Successful but unusable planning replies are still charged when usage is known.
The cap and pending reservations still protect final evaluation requests.

## Historical preservation

No original ledger, call log, request, response or historical inventory is
rewritten by this policy update. Old conservative records remain identifiable as
historical evidence; read-only views can apply the clarified exclusion without
altering their bytes. scripts/api/reconcile_usage.py is inspection-only and
requires an explicit evaluation stage. --apply is rejected.

Local verification uses synthetic responses and temporary ledger directories.
It must not create calls in the retained .model_usage record set.
