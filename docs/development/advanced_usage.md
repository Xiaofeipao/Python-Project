# Advanced Features and Development Evidence

## Pairwise text candidate integration (not executed)

The supplied WordModel share reports 70.00% validation accuracy on 940 examples, compared with a rechecked 69.68% baseline. This is a three-example validation gain, not a 70% independent test result. Its saved model was refitted on training plus validation and was not imported into this repository.

For any input with exactly two text columns, search now offers word-boundary character TF-IDF (3-6 grams), signed left-minus-right features and logistic regression without an intercept. A second candidate adds shared word unigram/bigram differences, capped at 15,000 features. Existing FeatureUnion combines both branches, avoiding a separate duplicated feature pipeline. Existing candidates remain available and validation still determines selection. No task or column name dispatch was added; tabular candidates are unchanged.

Proposals can select char_wb, fit_intercept and pairwise_word_features within existing change-category rules. Boolean flags and pairwise input shape are checked. Old serialized feature objects retain default behavior. Regression cases were added but not run, as requested. No training, inference, API calls or new accuracy measurements were performed for this integration, and existing best models remain unchanged.

## Search and feedback

The same runner supports tabular, text and image classification. Candidates are selected from data properties and measured validation feedback, not task identity. Available methods include linear models, trees, boosting, neural models and bounded blends. Ordered text columns preserve their identities; signed differences are available for exactly two columns. Preprocessing and vocabulary statistics fit training data only.

The planner records diagnosis, next_action and bounded proposals. proposals.py rejects unsupported families, paths, code, resources and parameters. It permits changes to one feature/hyperparameter category at a time. Recent trial summaries and family statistics support decisions. Repeated invalid structured output triggers local fallback. Actual API calls and accepted plans are distinct counts.

trial_memory.json stores canonical candidate hashes and family state. planner_checkpoint.json retains recent structured feedback and rejection state. Family allocation is heuristic: repeated failures or sustained poor results may retire a family within a run; exploration is retained. Family dispersion across configurations is distinct from seed variation.

## Stability and stopping

stability_repeats accepts 1-3, default 1. Repeats use different training seeds on the same supplied validation partition. score is the mean; validation_std, seed_scores and seeds preserve details. Final selection uses means and refits with the base seed; repeats are not automatically combined into an ensemble.

Seed variation does not estimate split uncertainty, provide a confidence interval or prove generalization. Deterministic methods may produce identical scores. No arbitrary variance penalty is hard-coded.

search_patience defaults to 0 (off). Once at least eight trials have completed, the configured interval without mean improvement may end search. Trial count, wall-clock and cost limits remain hard gates, with time reserved for final fitting.

## Test isolation

Formal policies disable progressive_output. Search reads training and validation; final test inference follows selection/refitting. Explicit development progressive output may predict features after checkpoint selection and must be labeled separately. Python audit hooks guard configured test_csv access in the training interpreter; this is not a cross-process native-code sandbox. Batch test arrays are converted after training ends.

## Recovery

```bash
python scripts/workflows/train_resilient.py --config configs/your_task.json --output runs/resilient_new --attempts 2
python main.py train --config configs/your_task.json --resume-from runs/interrupted --output runs/resumed_new
```

The supervisor permits at most three attempts in separate directories. Explicit interruption is not automatically restarted. Recovery requires unchanged source, policy, train/validation inputs, Python/distributions and original task ledger. It counts downtime and orphan-worker waiting against the original budget, preserving the official unseen deadline.

Completed runs cannot resume. Completed trials are restored; uncommitted worker results can be recovered. An interrupted candidate without a result is retained as interrupted and is not repeated. Unsafe concurrency or unverifiable worker state is refused. Keep parent directories for original requests, responses and source evidence. Final refitting may require new computation.

Recovery rebuilds family/hash memory and retains structured reflections/rejections. Its first new decision is local, avoiding replay of an interrupted API call. Unknown usage is never zeroed. New code cannot directly resume historical runs from different source versions.

## Optional feature components

Enable enable_feature_search in policy, or --feature-search in the batch script. Additional generic linear candidates compare no filtering, variance filtering and mutual information. Dense MI input is bounded to 128 MiB; multi-column text MI candidates use training-fitted 64-dimensional SVD first.

Bounded proposals can adjust imputation, one-hot/ordinal encoding, feature filters, selection percentile and vectorizer_options. Text options include n-gram range, minimum document frequency, feature cap, IDF, binary features and optional English stopwords. Inapplicable modality settings and mixed feature/hyperparameter changes are rejected.

MiniLM remains optional with --enable-minilm. It uses a fixed publisher revision, frozen features and automatic first-use downloading without remote code execution. Preserve the cache for offline reruns. It is not the planning LLM.

## Training-only nested diagnostics

nested_cv_folds accepts 0 (off), 2 or 3. Supported logistic/SVM candidates perform inner two-fold selection between two C values and separate outer evaluation. The default stratified training subsample has at most 600 rows; nested_cv_max_rows controls it. Insufficient class counts are explicitly skipped.

This is a bounded regularization diagnostic, not nested evaluation of the entire AutoML system. It never uses the official validation or test partition. Smaller-sample diagnostic scores are not directly comparable to full-data scores.

split_risk.json reports exact feature-row and normalized-text overlap between training and validation, without exporting text. Shared titles may be legitimate task structure. No time/source groups are invented and the official split is not silently changed.

## Experiments and reproducibility

```bash
python scripts/workflows/run_released_batch.py --output runs/features_new --max-trials 4 --feature-search --nested-cv-folds 2 --stability-repeats 2 --trial-timeout 1200
python scripts/reporting/decision_timeline.py runs/task/events.jsonl --output Timeline.md
python -m unittest discover -s tests -v
```

The original unified Pro snapshot achieved 79.07% / 66.86% on Coupon / Headlines. The later CPU stability run achieved 78.99% / 66.06%; Headlines stopped after nine trials. The four-trial linear feature study achieved 66.93% / 65.75%, without covering the original full search space. MiniLM trials also did not establish an improvement. Existing submitted models were retained.

These released tests were reused during development. Hardware and search allocations differ, so changes are not causal ablations. API unavailability caused local planning; do not attribute those decisions to a model. Historical accounting/revision evidence remains incomplete.

A controlled process-kill probe verified recovery and unchanged ledger paths. The latest pre-translation checks contained 53 cases: 51 passed and two CUDA checks were skipped. Original artifacts are local under runs/. Keep source, model, environment and logs from the same version. Only trusted serialized models should be loaded.

New runtime functionality uses existing dependencies and Python's standard library. Windows PowerShell and taskkill were used only for the controlled fault-injection probe, not required for normal training. See the API guide for genuine calls, credentials and accounting.
