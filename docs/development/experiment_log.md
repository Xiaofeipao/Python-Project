# Experiment log

## Experiment 1 — coupon-v2-20261123 (development)

The first experiment kept for the report is `runs/coupon-v2-20261123/`, created after the LLM plan-selection path was fixed (read timeout raised to 180 s, malformed-plan parsing hardened, unrecoverable usage reconciled explicitly).

| Item | Value |
| --- | --- |
| evaluation_id | `coupon-v2-20261123` |
| stage | development (test set used only after final selection) |
| runtime | frozen bundle `runtime_sha256 = 98bf60ef…48a68c` |
| compute | `D:\Anaconda\envs\gpu-pytorch` (torch 2.12.1+cu132), RTX 5060 Ti; xgboost members trained on CUDA |
| data | in-vehicle coupon: train 8117 / val 2030 / test 2537, 21 features |
| search | 4 trials; every non-baseline choice decided by the LLM planner |
| selected | trial 2, `cross_boost_blend`, validation accuracy 0.758128078817734 |
| held-out test | **accuracy 0.7922743397713835** (macro F1 0.785772, weighted F1 0.790877) |
| LLM calls | 3 recorded calls, 3933 input / 3728 output tokens |
| cost | **CNY 0.018845** (Flash, CNY 1/M input + 4/M output), accounting complete, no unreconciled reservation |

Artifacts: `RESULT.md`, `agent/evaluation.json`, `agent/model_usage.json`, `agent/trials.jsonl`, `agent/model_calls/` and `agent/predictions.csv` inside the run directory. The ledger lives in `.model_usage/a5fbbeb4cb1dcf940f857803856e11695308cbbfc616b6140420f09286c89b31/`.

## Task 2 — headlines CTR (reference numbers)

The second development task, `Released_Task2-Headlines_CTR` (pairwise headline preference, 3232 test rows), was trained with the same runtime and policy family. Its best held-out accuracy is **0.6686262376237624** (validation 0.6914893617021277), from the ten-trial API run. Its recorded cumulative cost is CNY 1.3649275, plus one historical read timeout accounted at the CNY 5.24288 upper bound, so its per-task accounting stays incomplete.

## Archive policy

Earlier exploratory runs are no longer listed as experiments. They were moved to `runs/_archive_20261123/` on 2026-11-23; the accounting ledger was left untouched and its evidence pointers were rewritten to the archive path, so every historical call in `.model_usage/` still resolves. Historical charges (about CNY 13.3 in total, including the two conservative timeout reconciliations) remain in the ledger and are not part of experiment 1's cost.
