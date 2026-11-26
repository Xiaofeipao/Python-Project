# Running SuperAgent

SuperAgent currently has a command-line interface, not a desktop or web GUI. Run commands from the repository root in an activated Python environment. See requirements.txt and requirements-gpu.txt for dependencies.

## Train both released tasks with Flash planning

The expected local input folders are in_vehicle_coupon_recommendation/ and Released_Task2-Headlines_CTR/. Keep your credential file outside the repository.

```powershell
$credentialPath = Read-Host 'Local API credential file path'
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
python scripts/workflows/run_released_batch.py --output "runs/flash_$runStamp" --max-trials 24 --llm-registry configs/model_registry.flash.json --llm-model DeepSeek-V4.1-Flash --key-file $credentialPath --llm-output-tokens 4096 --provisional-accounting
```

This performs new training and automatic selection for both tasks, using GPU where supported and available, with CPU fallback. It writes a new run directory and does not overwrite existing best models. Batch console output can remain quiet while subprocess logs are captured; the final task summary is printed after completion.

This example explicitly uses development mode. Provisional accounting does not itself establish formal eligibility. Current ledger totals and completeness are documented in ../accounting/ledger_inventory.json. Formal execution must follow full_assignment_workflow.md, verify task-wide ledger identity and use the submitted frozen policy.

## Run a separate connectivity probe

```powershell
python scripts/api/probe_hkust_api.py --model DeepSeek-V4.1-Flash --key-file $credentialPath --output "runs/api_probe_$runStamp"
```

The probe incurs a separate API call and tests connectivity only. It does not prove valid autonomous planning. A real integrated Flash plan was already accepted in the development check described in api_usage.md.

## Run another task

```bash
python main.py train --config configs/your_task.json --output runs/your_task_new
python main.py predict --model runs/your_task_new/model.joblib --input data/test.csv --output runs/your_task_new/predictions_new.csv
```

Provide a real task configuration based on the tabular/text/image examples. Every output must be new. Configured LLM training reads the API key from HKUST_API_KEY; configure it as described in api_usage.md. Prediction loads a trusted saved model and does not need an API key.

## Inspect outputs

For each batch task, agent/ contains model.joblib, summary.json, trials.jsonl, events.jsonl, model_calls/ and usage snapshots. Predictions are saved under the corresponding task's agent/ directory; final evaluations are in test_evaluation.json. The batch root contains result.json and the frozen source used for that run.

Use python main.py --help or the relevant script's --help for options. GUI support is not currently implemented.
