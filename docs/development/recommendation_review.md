# Review of Every Proposed Improvement

Every suggestion was reviewed. Reviewed does not mean fully implemented or proven effective. Implementation paths below are relative to superagent/ unless scripts/ is specified.

## Agent and planner: suggestions 1-9

1. **Trial hashes: implemented.** planning/memory.py records current-run family/parameter combinations. Menus exclude tried candidates; proposals.py checks again before execution. Independent reproducibility runs may intentionally repeat an experiment.
2. **Single-variable proposals: implemented as one change category.** A proposal changes features or hyperparameters, not both. Multiple fields within one category are possible, so this is not always a strict one-parameter causal ablation.
3. **Family statistics/allocation: heuristic implemented.** Best/mean/std, failures and elapsed time guide allocation and retirement, with exploration rounds. No optimality claim is made.
4. **Plateau versus seed fluctuation: descriptive measurements implemented.** stability_repeats records seeds, scores, means and standard deviations. This does not measure split uncertainty; ensemble seeds can be correlated. No significance claim is made.
5. **Compact context: implemented.** Recent five trials, best candidate, family summaries and rejection feedback replace full histories. Oversized contexts reduce the current executable menu.
6. **Few-shot examples: two implemented.** Parsing and validation remain authoritative. Offline/local-fallback runs do not establish improved real-model output reliability.
7. **Candidate list: current decision menu implemented.** Choices must reference it and overrides must obey rules. This does not enumerate an infinite parameter space or allow arbitrary model code.
8. **Rejected-proposal feedback: implemented.** Reasons enter logs and subsequent context. Accepted plans clear stale rejection state.
9. **Fallback reflection: bounded implementation.** Repeated invalid JSON disables API planning for the run. Network failures are recorded separately. No extra paid reflection-model branch was added.

## Stability and generalization: suggestions 10-13

10. **Multi-seed stability:** as in item 4. Enabled selection uses validation means, not the best seed.
11. **Generalization risk: partial diagnostic.** evaluation/split_overlap.py reports exact feature-row and normalized-title overlap. Reliable time/source metadata is unavailable, so no chronology is invented and official partitions remain unchanged. Zero exact overlap does not exclude semantic near-duplicates or other leakage.
12. **Variance penalty: no arbitrary fixed penalty.** Standard deviations are provided to planning; validation means remain the selection criterion. Two seeds cannot justify a generalization confidence interval.
13. **Nested CV: bounded implementation.** evaluation/nested_cv.py performs inner selection between two C values and outer evaluation on a training subsample. This is not nested evaluation of the complete AutoML search and does not replace official validation/test metrics.

## Feature engineering: suggestions 14-15

14. **Configurable components: implemented.** preprocessing/transformers.py supports imputation, one-hot/ordinal encoding, n-grams, min_df, max_features, IDF, binary features and optional stopwords. Proposals validate bounds and modality applicability.
15. **Filtering: variance and MI implemented.** Filters fit training data only, with a dense-memory bound. Multi-column text MI candidates can use training-fitted SVD. Failures are logged; improvement is not guaranteed.

## Test isolation: original section five

16. **Read guard: bounded implementation.** governance/test_access.py blocks the configured test_csv in the current Python training interpreter before selection. It is not an OS sandbox for native code, other processes or arbitrary file aliases.
17. **Timing evidence: scoped implementation.** Configured final prediction logs selection_completed and test_read_started timestamps. Batch conversion follows completed training, but not every OS file access is globally logged.
18. **Train-only fitting: implemented and tested.** Scaling, vocabularies, encoding, selection and SVD fit each appropriate training partition. Nested folds rebuild models. Final train-plus-validation refitting remains distinct from test statistics.

## Accounting and recovery: original section six

19. **API exponential-backoff retries: not implemented.** Unknown usage retains its reservation and is never silently replayed. The first restored decision is local. Bounded training-process restarts are not HTTP-request retries.
20. **Version binding: strengthened.** Environment records include Python, main libraries and installed distributions. Trials reference source/environment evidence. Recovery verifies code, policy, data, dependencies and the original ledger.
21. **Recovery: implemented.** runtime/checkpoint.py and scripts/workflows/train_resilient.py restore to new audit directories, rebuild trial/family memory, retain structured reflections and parent evidence, and count downtime. Only unfinished same-version runs can resume. Controlled process termination verified automatic completion and unchanged ledger paths.

## Stopping: original section seven

22. **Patience: optional implementation.** After at least eight trials, a configured interval without validation-mean improvement can stop search. The real Headlines stability run stopped at trial nine. This is a heuristic, not proof of an optimum.
23. **Family retirement/exhaustion: implemented.** Failed families leave the current menu and exhausted eligible search ends. Algorithms are not permanently banned across unrelated tasks.
24. **Cost/time gates: implemented.** Historical unknown holds and the next worst-case request count against the task budget. Final refitting receives reserved time; restarting never resets the unseen window. Unreconciled runs remain development artifacts.

## Current evidence and limits

Use current_results.json for the retained model pointers, experiment_log.md for the current run catalog, and ../accounting/ledger_inventory.json for every current source ledger. Previously written implementation tests remain available in tests/; this organization pass did not run them again.

Capabilities described above are separate from measured improvement. Current reported models use different frozen runtime hashes, and the released test set was reused in development. The report still refers to its stated earlier frozen experiment. Current indexes do not authorize spending or replace original run artifacts.
