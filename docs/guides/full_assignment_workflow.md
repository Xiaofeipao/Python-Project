# Formal Evaluation Workflow

## 1. Prepare inputs

Use the announced train/validation/test partitions, target, modality columns, metric and evaluation ID. One SuperAgent handles all tasks. Test data is final-only; task names must not select manually prepared solutions.

## 2. Freeze source and policy

```bash
python main.py freeze --policy configs/final_policy.json --output releases/final
python releases/final/main.py verify-freeze --manifest releases/final/frozen_manifest.json
```

The output must be new. Formal execution uses this frozen entry point. Do not modify frozen source, policy or examples. Keep task inputs outside the bundle.

## 3. Released tasks

Complete configs/released.example.json with real paths, metric, unique evaluation_id and frozen_manifest. Configure one persistent ledger shared by all model-calling processes:

```powershell
$env:SUPERAGENT_USAGE_ROOT = Join-Path (Get-Location) '.model_usage'
python releases/final/main.py train --config configs/released.example.json --output runs/released_task_new
```

Replace placeholders first. The agent compares candidates and refits automatically. Configured test_csv is predicted only after selection. Evaluate separately when test labels are officially available; never feed final scores back into search.

## 4. Unseen task

After the official release, configure configs/unseen.example.json with the timezone-aware unseen_released_at timestamp:

```bash
python releases/final/main.py train --config configs/unseen.example.json --output runs/unseen_task_new
```

The 24-hour window starts at official release, not process launch or restart. Use the submitted frozen system without manual tuning, trial scheduling or final-model selection. Submit predictions and logs before the deadline; no additional unseen report is required under the supplied assignment.

## 5. Submission

Include source, unified entry point, dependencies, README, released-task models/results, execution logs, environment/source evidence, report.pdf and confirmed AI declarations. All reported results must come from the same final version and complete autonomous runs. Explain failures and valid partial results. The stated deadline is 23:59 on the night before the first class in the section's final week; confirm the actual date from the course timetable.

See [API setup](api_usage.md) and the original assignment requirements at the repository root. Recheck course announcements before submission.
