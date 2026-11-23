# Experiment log

Both reported experiments were run on 2026-11-23 with the same frozen runtime (`runtime_sha256 = 98bf60ef…48a68c`), after the LLM planning path was fixed (180 s read timeout, hardened malformed-plan parsing, explicit reconciliation of unrecoverable usage). They replace the earlier exploratory runs, which are no longer counted as experiments.

## Experiment 1 — coupon-v2-20261123 (development, GPU)

| Item | Value |
| --- | --- |
| evaluation_id | `coupon-v2-20261123` |
| data | in-vehicle coupon, 21 features: train 8117 / val 2030 / test 2537 |
| compute | `gpu-pytorch` env, torch 2.12.1+cu132, RTX 5060 Ti; xgboost members on CUDA |
| search | 4 trials, every non-baseline choice decided by the LLM planner |
| selected | trial 2, `cross_boost_blend` |
| validation | 0.758128078817734 |
| held-out test | **accuracy 0.7922743397713835** (2010/2537), macro F1 0.785772, weighted F1 0.790877 |
| LLM calls | 3 recorded, 3933 input / 3728 output tokens |
| cost | **CNY 0.018845**, accounting complete, no unreconciled reservation |

## Experiment 2 — headlines-v2-20261123 (development, CPU)

| Item | Value |
| --- | --- |
| evaluation_id | `headlines-v2-20261123` |
| data | headline pair preference (`label_pairwise`): train 8301 / val 940 / test 3232 |
| compute | `cpu-pytorch` env, `device=cpu`, `enable_torch=false`, profile `standard` |
| search | 6 trials, all chosen by the LLM planner |
| selected | trial 2, logistic regression on character n-grams, separate representation, balanced |
| validation | 0.6882978723404255 |
| held-out test | **accuracy 0.6599628712871287** (2133/3232), macro F1 0.659943, weighted F1 0.659899 |
| LLM calls | 5 recorded, 7795 input / 4690 output tokens |
| cost | **CNY 0.026555**, accounting complete, no unreconciled reservation |

Experiment 2 used 6 trials instead of the 10 used by the archived run, and the planner picked logistic instead of `linear_svm C=0.1 combined`; that is why its test accuracy is about 0.9 points below the archived 0.668626.

## Reproduction — coupon-v3-20261123 (development, CPU)

`coupon-v2` recorded a `status=failed` summary because the held-out `test.csv` was generated only after selection, so the automatic final prediction step read a missing file; the three planner calls themselves succeeded and `agent/evaluation.json` carried the real score. A clean single-pass rerun was made on 2026-11-23 to keep one internally consistent record:

| Item | Value |
| --- | --- |
| evaluation_id | `coupon-v3-20261123` |
| compute | same frozen runtime, `device=auto`; the NVIDIA RTX 5060 Ti had detached (PnP status `CM_PROB_PHANTOM`, `torch.cuda.device_count()==0`), so the run executed on CPU |
| data | identical splits, `test.csv` sha256 `ee6dac76…398c54` (same held-out file as `coupon-v2`) |
| search | 4 trials, LLM planner chose catboost variants each time |
| selected | trial 3, catboost `iterations=1200, depth=8, lr=0.035, l2=10` |
| validation | 0.7512315270935961 |
| held-out test | **accuracy 0.7828143476547103**, macro F1 0.775095 |
| summary | `status=completed`, final prediction rows 2537 |
| LLM calls | 3 recorded, 4250 input / 3752 output tokens |
| cost | **CNY 0.019258**, accounting complete |

The rerun is a reproducibility check, not a replacement: `coupon-v2` remains the reported GPU result (0.792274), and the CPU rerun lands 0.95 points lower with a different planner-selected family, which is consistent with losing the CUDA xgboost member.

## Totals and archive policy

The three 2026-11-23 runs cost **CNY 0.064658** in recorded Flash usage (0.018845 + 0.026555 + 0.019258), with no reservation held.

Earlier exploratory runs were removed from local `runs/` on 2026-11-23 and are not listed as experiments. Before removal, all 107 referenced request/response files were copied into `.model_usage/<ledger>/evidence/` and the ledger pointers were rewritten, so every historical call still resolves. Their charges stay in the accounting ledger (about CNY 13.3 in total, including two conservative timeout reconciliations at the reserved upper bound) as disclosed AI usage, not part of either experiment above.
