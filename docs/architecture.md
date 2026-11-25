# Architecture and Directory Responsibilities

Read the entry point and runtime first, followed by planning, preprocessing/models, evaluation and governance.

```text
main.py                             Unified entry point
superagent/
  cli.py                            Command parsing and dispatch
  config.py                         Typed configuration and resource limits
  runtime/
    orchestrator.py                 Experiment loop, selection and refitting
    worker.py                       Isolated fitting, seeds and failure handling
    checkpoint.py                   Locks, verified recovery and ledger binding
  planning/
    search_space.py                 Generic candidates and model construction
    llm.py                          Context, API calls, parsing and fallback
    proposals.py                    Proposal categories, bounds and deduplication
    memory.py                       Candidate hashes and family statistics
  preprocessing/
    datasets.py                     Data loading, partitions, labels and hashes
    transformers.py                 Imputation, encoding, text, SVD and images
    feature_selection.py            Variance/MI filters and memory limits
  models/
    pairwise.py                     Selected ordered-text overlap classifier
    tabular.py                      Boosting, blending and member caching
    neural.py                       Neural training, devices and serialization
    text_encoder.py                 Fixed-revision MiniLM features
  evaluation/
    metrics.py                      Metrics and optimization direction
    inference.py                    Prediction and final evaluation
    nested_cv.py                    Bounded training-only nested diagnostics
    split_overlap.py                Descriptive train/validation overlap
  governance/
    usage_ledger.py                 Tokens, prices, reservations and budgets
    frozen_bundle.py                Source freezing, hashes and unseen deadline
    test_access.py                  Configured test-file Python audit guard
  common/json_io.py                 Atomic JSON and timestamped events
scripts/
  workflows/                        Batch tasks and bounded automatic recovery
  data/                             NumPy task conversion
  api/                              Connectivity probes
  validation/                       Benchmarks, artifact and freeze checks
  reporting/                        Timelines derived from actual events
 tests/                             Contracts, planning, models and workflows
 configs/                           Examples, policies, registries and evidence
 docs/guides/                       Operational instructions
 docs/accounting/                   Current ledger inventory and accounting scope
 docs/development/                  Improvement review and experiment limits
 docs/report/                       Canonical report PDF and LaTeX
 examples/                          Small real execution examples
```

## Execution flow

1. Parse configuration and verify frozen policy, budget and deadline.
2. Read training/validation only; record hashes, environment and overlap.
3. Propose a candidate from feedback and validate its bounds and novelty.
4. Fit in an isolated worker, measure validation and update memory/usage.
5. Select automatically, refit on training plus validation, then predict test.
6. Recover through the parent evidence chain with verified inputs and the original remaining budget.

## Compatibility and local artifacts

The root features.py, tabular_models.py, torch_models.py, embeddings.py and selection.py modules preserve historical pickle imports. Their implementations live in subpackages; do not add business logic to these compatibility files. Root script wrappers preserve old command paths.

Local data/, runs/, .model_usage/, .pretrained_cache/, releases/, submission_*/ and pre/ hold inputs, evidence, caches or outputs and are ignored by Git. Course data directories retain their original paths to preserve existing configurations.
