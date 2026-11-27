# Requirements and Submission Status

This checkout is under development and has not been frozen. This document maps
the local assignment requirements to implementation and missing evidence. It is
not an assertion of complete formal eligibility.

## One general SuperAgent

main.py is the unified entry point. One AutonomousPlanner owns each run, with
llm_agent_id fixed to planner. Fit workers execute candidates and do not inherit
the planner API credential. They are not additional decision-making agents.
Batch tasks run serially using the same source and algorithm policy. Input
schema and task IDs are supplied for loading and accounting, not model dispatch.
Ensemble members, repeated seeds and nested validation are estimators and
evaluation procedures inside the one agent.

## Autonomous search and held-out evaluation

The agent analyzes train/validation data, proposes candidates, records diagnosis
and outcomes, checks proposal bounds and duplicates, and selects by the specified
validation metric. Feature transformers fit only the appropriate training fold.
Final refitting can combine train and validation. If refits fail, verified trial
models can provide a partial solution. Selection uses no test statistic.

The designated test path stays guarded in both controller and workers throughout
search and refitting. File aliases are checked at Python open calls. Even with
progressive_output enabled, checkpoints do not predict test inputs. The final
selection event precedes test reads. Explicit predict/evaluate commands are for
the final evaluation phase only. The guards do not undo historical test reuse or
stop an operator from manually conducting prohibited post-hoc comparisons.

## Resource limits and reproducibility

The registry restricts language-model families and declared release/pricing
metadata. Actual token usage, reservations and per-task costs are accounted for
without inventing missing usage. Formal runs require a persistent shared ledger.
Unknown-usage timeouts are not replayed; documented failed requests without usage are excluded from the final course budget. The short format-repair request has a
single allowance and consumes a separate reservation. Local fallback is logged.

All runtime calls belong to the one planner. The ledger supports historical
records from other identifiers for auditing; this does not launch other agents.
Current requests use the planner identifier only.

Dependencies, observed versions, source snapshots, data hashes, seeds and device
reports support reproduction. Exact pretrained image weights and their source
URL/hash are recorded. device=auto selects CUDA when available, otherwise CPU.
Environment instructions are in reproducible_environment.md. Search/refit worker
timeouts and checkpoint recovery protect unattended runs. The official unseen
window remains supported, but no unseen evaluation has been performed here.

## Evidence still required beyond source cleanup

- Confirm that the deployed language-model revision meets the release cutoff.
- Confirm that all new accounting uses the user-selected off-peak prices (Flash 1/4, Pro 4.5/13.5), with earlier standard/peak records retained as historical.
  The local assignment text specifies official standard pricing as of the cutoff.
- Generate the final reported task results through autonomous runs of one final
  source version and preserve matching models, predictions and logs. Existing
  development frontier models have different source hashes. No fresh scores are
  claimed for the current unfrozen source.
- Retain the disclosure that released tests were reused during development.
  Future unseen evaluation must not repeat that procedure.
- Confirm authors, student IDs, member contributions and individual AI usage.
- Update report.pdf from the actual final experiment records. The current report
  describes its earlier experiment, not all current source features.
- Assemble models, runnable preprocessing/prediction code, results and execution
  logs with the source, README, dependencies, report and AI declaration.

Runtime summary fields formal_runtime_checks_passed and
submission_review_required distinguish automated checks from human evidence
review. eligible_for_formal_submission is never inferred from complete accounting
alone. Freezing is deliberately deferred and must be an explicit later action.

## Local verification policy

Use offline tests and synthetic HTTP responses. Do not run connectivity probes,
real model requests, ledger reconciliation or released-task searches merely to
verify code organization. Synthetic ledger fixtures use temporary test folders,
not the official .model_usage directory. New models and scores must not overwrite
the retained development artifacts. No code cleanup can guarantee unchanged
accuracy after a new stochastic search.

## Accounting clarification supplied on 26 November 2026

The newer course clarification supersedes earlier conservative unknown-cost holds: failed requests without usage do not consume the final-task budget and need no upper-bound reconciliation. Development costs are excluded from the course cap. See ../accounting/policy.md. All historical source records remain unchanged.
