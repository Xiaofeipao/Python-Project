# Architecture and Directory Responsibilities

The code follows the execution flow: interface/configuration, runtime, planning,
feature preparation, estimators, evaluation and governance. Read `main.py`,
`superagent/cli.py` and `runtime/orchestrator.py` first.

```text
main.py                               Unified public command-line entry point
superagent/
  cli.py                              Parse commands and dispatch operations
  config.py                           Validate task schemas and resource limits
  runtime/
    orchestrator.py                   Own the experiment loop and final selection
    worker.py                         Execute isolated fits, seeds and failures
    checkpoint.py                     Save models, recover, lock outputs and resume
  planning/
    llm.py                            One planner: summaries, API replies and fallback
    proposals.py                      Validate proposal categories, bounds and novelty
    memory.py                         Hash configurations and summarize model families
    search/
      candidates.py                   Generic candidate catalog and local refinement
      allocation.py                   Metric variants, candidate identity and timing
    search_space.py                   Compatibility facade for previous imports
  preprocessing/
    datasets.py                       Load data, inspect schemas, split and hash inputs
    pipelines.py                      Compose modality-specific and mixed feature paths
    feature_selection.py              Optional variance/MI filtering and memory limits
    components/
      base.py                         Constant baseline features and bounded SVD
      tabular.py                      Frame schemas, dates and category frequencies
      text.py                         TF-IDF, text columns and ordered-pair features
      image.py                        Load images, resize pixels and form histograms
    transformers.py                   Compatibility facade for old serialized models
  models/
    cache.py                          Resolve shared project-local pretrained resources
    factory.py                        Turn a validated candidate into an estimator
    decision.py                       Fixed thresholds and train-only calibration
    tabular.py                        Boosting, native categories and cached blends
    neural.py                         Shared neural fitting and frozen image encoding
    text/
      encoders.py                     Pinned pretrained encoders and feature extraction
      networks.py                     Trainable EmbeddingBag, TextCNN and GRU models
      finetuning.py                   Optional Transformer sequence-classifier fitting
      pairwise.py                     Compose ordered-pair overlap classifiers
    text_encoder.py, text_neural.py,
    text_finetune.py, pairwise.py      Compatibility facades, not duplicate estimators
  evaluation/
    metrics.py                        Scores and optimization direction
    inference.py                      Final prediction and labeled evaluation
    nested_cv.py                      Bounded training-only nested diagnostics
    split_overlap.py                  Describe train/validation overlap
  governance/
    usage_ledger.py                   Tokens, prices, reservations and budget scopes
    frozen_bundle.py                  Source bundles, hashes and unseen deadline
    test_access.py                    Final-only test-file Python audit guard
  common/json_io.py                   Atomic JSON writes and timestamped events
  features.py, embeddings.py,
  selection.py, tabular_models.py,
  torch_models.py                     Earlier serialized-model compatibility paths
scripts/
  setup/                              Prepare pinned pretrained weights and tokenizers
  workflows/                          Batch tasks and bounded automatic recovery
  data/                               Convert trusted NumPy task inputs
  api/                                Explicit connectivity probes and reconciliation
  validation/                         Benchmarks, artifact and frozen-workflow checks
  reporting/                          Timelines and historical report-data helpers
  *.py                                Earlier command-path compatibility launchers
 tests/                               Contracts, planning, preprocessing, models, workflows
 configs/                             Task examples, shared policies, registries and evidence
 docs/
  guides/                             Setup, API, execution and submission instructions
  accounting/                         Retained ledger inventory and accounting policy
  development/                        Recorded results, suggestions and limitations
  report/                             Canonical report PDF and editable LaTeX
 examples/                            Small recorded execution examples
```

## Layer boundaries

Planning chooses *what to try*. `models/factory.py` specifies *how to build it*.
The factory composes preprocessing and estimators; it does not choose a task's
winner. Feature components learn only from their fitting partition, while
`preprocessing/pipelines.py` defines how components are combined. Optional model
packages remain lazily imported in the relevant fitting paths.

`planning/search/candidates.py` contains the complete existing candidate order
and refinement rules. `allocation.py` retains resource estimates and metric
variants. This organization does not introduce another agent, alter search
priorities, change configurations or add any model family.

## Execution flow

1. Parse configuration; verify frozen policy only for explicit formal execution.
2. Read training/validation only; record hashes, environment and overlap.
3. Propose a candidate from feedback and validate its bounds and novelty.
4. Build the feature/estimator pipeline in an isolated worker, measure validation
   and update the same planner's memory and scoped usage.
5. Select automatically, refit on training plus validation with bounded attempts,
   or recover a verified train-only model marked partial. Only then predict test.
6. Resume through the evidence chain with verified inputs and the original
   remaining budget.

The manifest separates `requested_config` from the effective configuration
derived from data and hardware. The caller's configuration is copied. A
fresh-process resume verifies the original request, re-derives runtime choices
and checks them against the recorded effective configuration. Source/data
bindings are not weakened.

Initial exploration covers different model families before repeated tuning.
Observed family runtimes and candidate training sizes constrain later proposals.
Binary F1/balanced-accuracy tasks can compare fixed thresholds on validation;
probability objectives can compare sigmoid calibration. Complete pipelines are
refitted inside calibration folds. Selected decision settings remain fixed for
final refitting and test evaluation. Accuracy does not automatically add these
variants. Frozen image encoding provides a CPU-compatible alternative to full
fine-tuning, with streamed batches and cached pipeline features where enabled.

There is one `AutonomousPlanner` in each run. The controller owns all planning and
API calls. Fit workers execute candidates serially and receive no planner API
credential. Ensemble members are estimators, not independent agents. Local
fallback belongs to the same decision loop. No task identity selects a prepared
winner, and external test evaluations have no return path into selection.

The configured test file is guarded in the controller and fitting workers.
Canonical paths and file-identity aliases are checked. This is a Python-level
backstop rather than an OS security sandbox. Checkpoint saving never opens test.

## Compatibility, evidence and local artifacts

Compatibility facades re-export the canonical implementations. They preserve
previous imports and historical pickle class lookups; do not add business logic
to them. `tests/contracts/test_module_layout.py` checks these identities and old
pickle globals. No copies of the estimator implementations are maintained in
the facades.

Historical fitted models still require their recorded compatible dependencies
and resources. Frozen artifacts and interrupted runs remain bound to their
original source snapshot. A directory refactor changes the source fingerprint:
resume old runs from their matching original snapshot, rather than modifying
old manifests or weakening verification.

Local `data/`, `runs/`, `.model_usage/`, `.pretrained_cache/`, `releases/`,
`submission_*/` and `pre/` hold inputs, evidence, caches or outputs and are ignored
by Git. Course data directories retain their original paths to preserve
existing configurations. Model settings, task inputs, accounting records and
historical source snapshots are not reorganized or rewritten by this refactor.
