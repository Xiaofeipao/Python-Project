# AIAA 3102 General Classification SuperAgent

One system selects classification methods using training data, resources and validation feedback. It supports tabular, text and image inputs. Task names organize artifacts; they do not dispatch to manually prepared winners.

## Start here

- [Architecture and directory responsibilities](docs/architecture.md)
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

Training CSV files include the target column. Relative paths are resolved against the configuration directory. Follow the task announcement for modality, columns, IDs and metric. Examples are under configs/; final_policy.json contains the frozen algorithm policy.

## Unified entry point

```bash
python main.py train --config configs/tabular.example.json --output runs/task_new
python main.py predict --model runs/task_new/model.joblib --input data/test.csv --output runs/task_new/predictions_new.csv
python main.py evaluate --model runs/task_new/model.joblib --input data/test.csv --output runs/task_new/evaluation_new.json
```

Outputs must not already exist. Search uses training/validation only. Test prediction follows final selection. Evaluation requires labels and is final-only. Vocabulary, encoding, scaling, SVD and filtering fit the appropriate training partition. Final refitting may combine training and validation.

## Batch tasks and recovery

```bash
python scripts/workflows/run_released_batch.py --output runs/released_new --max-trials 10
python scripts/workflows/train_resilient.py --config configs/your_task.json --output runs/resilient_new --attempts 2
python main.py train --config configs/your_task.json --resume-from runs/interrupted --output runs/resumed_new
```

Batch inputs are the local in_vehicle_coupon_recommendation/ and Released_Task2-Headlines_CTR/ directories. Original data is not published. Test arrays are converted only after selection and refitting. The batch command defaults to local planning; enable genuine LLM calls with the registry/model/credential options in the API guide.

Optional flags include --stability-repeats 2, --search-patience 6, --feature-search, --nested-cv-folds 2 and --enable-minilm. They change cost and search allocation, without guaranteeing an accuracy gain. Nested diagnostics remain inside training and do not replace the course validation metric.

Recovery binds the original source, configuration, data, environment and ledger. Downtime counts against the original budget. Only unfinished runs from the same source version can resume. Unknown API usage is never changed to zero or blindly replayed.

## Artifacts and checks

Runs retain models, predictions, trials/events/summary, environment, source snapshots, search memory and usage snapshots. Prediction needs no planner API key. Load only trusted models with their corresponding source.

```bash
python -m unittest discover -s tests -v
python scripts/validation/check_frozen_workflow.py --help
```

Implementation is grouped by responsibility. A few root modules preserve historical pickle imports; old script commands forward to the organized implementations.

## Results and submission boundaries

The original unified Pro development snapshot achieved 79.07% Coupon accuracy and 66.86% Headlines accuracy. Later stability and feature studies did not replace those models. Released test sets were reused during development, so these are not independent unbiased final estimates. Performance marks depend on group rankings.

Historical timeout usage and deployment revision evidence remain incomplete. The report retains its development-draft status. Source, policy, models, logs and reported results must refer to the same version. New capabilities must not be attributed to old experiments. Members must confirm AI_USAGE.md.

Only docs/report/report.pdf and its LaTeX sources are published as report material. Other PDFs, presentations, pre/, datasets, credentials, runs and local submission bundles are ignored. Official source evidence is stored as lossless Base64 archives with English metadata; decoded original bytes retain their registered hashes.
