# AIAA 3102 General Classification SuperAgent

One system selects classification methods using training data, resources and validation feedback. It supports tabular, text and image inputs. Task names organize artifacts; they do not dispatch to manually prepared winners.

## Start here

- [Documentation index](docs/README.md) and [architecture](docs/architecture.md)
- [Reproducible environment and pinned pretrained weights](docs/guides/reproducible_environment.md)
- [Current accounting policy](docs/accounting/policy.md), [retained records](docs/accounting/README.md) and [model pointers](docs/development/current_results.json)
- [Commands for running SuperAgent](docs/guides/quickstart.md)
- [Review of every proposed improvement](docs/development/recommendation_review.md)
- [Experiment log](docs/development/experiment_log.md)
- [Advanced features and experiment history](docs/development/advanced_usage.md)
- [API setup](docs/guides/api_usage.md) and [formal evaluation workflow](docs/guides/full_assignment_workflow.md)
- [English report and LaTeX source](docs/report/README.md)

## Installation and configuration

Use Python 3.10+ and run from the project root in an activated environment:

```bash
python -m pip install -r requirements.txt
python main.py doctor
```

Optional GPU, XGBoost and MiniLM dependencies are in requirements-gpu.txt. Choose a compatible PyTorch build. `device: "auto"` supports CPU fallback. MiniLM is an optional frozen feature extractor, disabled by default; it does not replace the Qwen/DeepSeek planner.

Training CSV files include the target column. Relative paths are resolved against the configuration directory. Follow the task announcement for modality, columns, IDs and metric. Examples are under configs/; final_policy.json is a proposed shared algorithm policy. It does not freeze the current checkout.

## Unified entry point

```bash
python main.py train --config configs/tabular.example.json --output runs/task_new
python main.py predict --model runs/task_new/model.joblib --input data/test.csv --output runs/task_new/predictions_new.csv
python main.py evaluate --model runs/task_new/model.joblib --input data/test.csv --output runs/task_new/evaluation_new.json
```

Outputs must not already exist. Exactly one planner owns decisions and API calls per run; isolated fit workers receive no planner API credential. Search uses training/validation only. Test prediction follows final selection. Evaluation requires labels and is final-only. Vocabulary, encoding, scaling, SVD and filtering fit the appropriate training partition. Final refitting may combine training and validation. progressive_output saves checkpoints only and never predicts test inputs during search.

## Batch tasks and recovery

```bash
python scripts/workflows/run_released_batch.py --output runs/released_new --max-trials 24
python scripts/workflows/train_resilient.py --config configs/your_task.json --output runs/resilient_new --attempts 2
python main.py train --config configs/your_task.json --resume-from runs/interrupted --output runs/resumed_new
```

Batch inputs are the local in_vehicle_coupon_recommendation/ and Released_Task2-Headlines_CTR/ directories. Original data is not published. Batch execution is serial development execution of one shared policy and does not freeze the project. Test arrays are converted only after final model selection. The batch command defaults to local planning; enable genuine LLM calls only when authorized, using the registry/model/credential options in the API guide. Formal released/unseen execution is a separate explicit workflow.

Optional flags include --stability-repeats 2, --search-patience 6, --feature-search, --nested-cv-folds 2 and --enable-minilm. They change cost and search allocation, without guaranteeing an accuracy gain. Nested diagnostics remain inside training and do not replace the course validation metric.

Recovery binds the original source, configuration, data, environment and ledger. Downtime counts against the original budget. Only unfinished runs from the same source version can resume. Unknown API usage is never changed to zero or blindly replayed.

## Artifacts and checks

Runs retain models, predictions, trials/events/summary, environment, source snapshots, search memory and usage snapshots. Prediction needs no planner API key. Load only trusted models with their corresponding source.

```bash
python -X utf8 -m unittest discover -s tests -t . -v
python scripts/validation/check_frozen_workflow.py --help
```

Implementation is grouped by responsibility. Compatibility modules directly under superagent/ preserve historical pickle imports. Model serialization, verified fallback and resume bindings share runtime/checkpoint.py.

## Results and submission boundaries

See [requirements status](docs/guides/requirements_status.md) for implemented safeguards and unresolved submission evidence. Passing tests is not a certification of course compliance. This checkout remains unfrozen; the figures below are development results from the recorded runs, not a frozen formal submission.

The current off-peak development pair reaches 79.15% Coupon accuracy (`coupon-recover3-20261125-coupon`, 2008/2537) and 69.80% Headlines accuracy (`headlines-recover2-20261125-headlines`, 2256/3232), using 24 trials each and accounting-complete Flash 1/4 ledgers. The Coupon snapshot predates the generic two-text-column anchor, so the two source fingerprints differ only in planner anchor ordering, not model or training code. Within each run, model selection used only the official validation split; test was read once after final selection. Released test sets were reused during development, so these are not independent unbiased final estimates.

See docs/accounting/ledger_inventory.json for the retained ledger sources and their hashes. Selected-run costs are already included and must not be added again.

The reported models have different frozen runtime hashes. They form a development frontier, not a single-version formal experiment. Deployment revision eligibility remains separate from accounting arithmetic. The published report describes an earlier frozen experiment and remains a draft; do not attribute later features or current frontier scores to it. Members must confirm AI_USAGE.md.

Only docs/report/report.pdf and its LaTeX sources are published as report material. Other PDFs, presentations, pre/, datasets, credentials, runs and local submission bundles are ignored. Official source evidence is stored as lossless Base64 archives with English metadata; decoded original bytes retain their registered hashes.
