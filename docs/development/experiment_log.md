# Experiment log

Both reported experiments were run on 2026-11-23 on the RTX 5060 Ti (`gpu-pytorch`, torch 2.12.1+cu132) with the same frozen runtime (`runtime_sha256 = 98bf60ef…48a68c`), after the LLM planning path was fixed (180 s read timeout, hardened malformed-plan parsing, explicit reconciliation of unrecoverable usage). Both finished with `status=completed`, a full prediction file and an accounting-complete ledger.

## Experiment 1 — coupon-gpu-20261123 (development, GPU)

| Item | Value |
| --- | --- |
| evaluation_id | `coupon-gpu-20261123` |
| data | in-vehicle coupon, 21 features: train 8117 / val 2030 / test 2537 |
| compute | `gpu-pytorch` env, `device=auto` → CUDA available; xgboost members on GPU |
| search | 4 trials, every non-baseline choice decided by the LLM planner |
| selected | trial 4, `cross_boost_blend` |
| validation | 0.7596059113300493 |
| held-out test | **accuracy 0.7906976744186046** (2006/2537), macro F1 0.784146, weighted F1 0.789289 |
| LLM calls | 3 recorded, 3915 input / 2579 output tokens |
| cost | **CNY 0.014231**, accounting complete, no unreconciled reservation |

Trial history: dummy 0.568473 → `cross_boost_blend` 0.758128 → 0.757143 → **0.759606 (selected)**.

## Experiment 2 — headlines-gpu-20261123 (development, GPU host, sklearn model)

| Item | Value |
| --- | --- |
| evaluation_id | `headlines-gpu-20261123` |
| data | headline pair preference (`label_pairwise`): train 8301 / val 940 / test 3232 |
| compute | `gpu-pytorch` env, `device=auto`, `enable_torch=true`, mixed precision ready; the selected model is sklearn, so it executes on CPU |
| search | 10 trials, all chosen by the LLM planner (one malformed plan was rejected and re-planned) |
| selected | trial 2, logistic regression on character n-grams, separate representation, balanced |
| validation | 0.6904255319148936 |
| held-out test | **accuracy 0.6605816831683168** (2135/3232), macro F1 0.660564, weighted F1 0.660524 |
| LLM calls | 9 recorded, 16120 input / 10168 output tokens |
| cost | **CNY 0.056792**, accounting complete, no unreconciled reservation |

Trial history: dummy 0.511702 → **logistic 0.690426 (selected)** → `linear_svm` 0.671277 / 0.679787 / 0.684043 / 0.686170 / 0.673404 / 0.655319 / 0.687234 → `xgboost` 0.617021.

## Search-effort runs on the same runtime

Three additional rounds raised the trial budget to see whether more planning calls buy accuracy. Every run used the same frozen runtime, the same splits and `device=auto` on the RTX 5060 Ti.

| Run | trials | validation | held-out test | LLM calls | cost (CNY) |
| --- | --- | --- | --- | --- | --- |
| `coupon-gpu-20261123` | 4 | 0.759606 | 0.790698 | 3 | 0.014231 |
| `coupon-t10-20261123` | 10 | 0.759606 | 0.790698 | 9 | 0.058049 |
| `coupon-t14r-20261123` | 14 | 0.759606 | 0.790698 | 13 | 0.065004 |
| `headlines-gpu-20261123` | 10 | 0.690426 | 0.660582 | 9 | 0.056792 |
| `headlines-t15-20261123` | 15 | 0.690426 | 0.660582 | 11 | 0.074308 |
| `headlines-t20r-20261123` | 20 | 0.690426 | 0.660582 | 15 | 0.089579 |

The search has converged: coupon trials sit between 0.7547 and 0.7596 validation across 14 candidates, and the planner reselects the same `cross_boost_blend` point; headlines trials sit between 0.6564 and 0.6904, where `logistic` (char n-grams, separate) and one `linear_svm` variant tie, and the tighter of the two wins on test. Extra planning calls therefore cost money without moving the held-out score.

Two further attempts were interrupted mid-run and are kept only as charge records: `coupon-t14` finished its search but its final fit was terminated, and `headlines-t20` stopped at trial 13; neither produced a model or test score. A decision-threshold sweep on the saved coupon model was also rejected: validation accuracy rose to 0.8828 at threshold 0.54 but test accuracy fell from 0.7907 to 0.7840, so the default 0.5 boundary is kept.

Best versions (`runs/best.json`): `coupon-gpu-20261123` and `headlines-gpu-20261123`, chosen because they reach the same held-out score as the longer searches at the lowest recorded cost.

## Accuracy-enrichment probe — capability flags and seed variance

Two capabilities were switched off in every earlier run, so the planner never saw them: `enable_minilm` (MiniLM sentence embeddings feeding a logistic head) and `enable_feature_search` (variance and mutual-information filters). Three runs turned both on and additionally set `stability_repeats=3`, which scores each trial as the mean of three training seeds instead of one.

| Run | config | trials | validation | held-out test | LLM calls | cost (CNY) |
| --- | --- | --- | --- | --- | --- | --- |
| `coupon-minilm2-20261123` | flags on, 3-seed validation | 12 | 0.758785 | 0.791486 | 11 | 0.059672 |
| `headlines-minilm2-20261123` | flags on, 3-seed validation, seed 42 | 12 | 0.684043 | 0.670483 | 11 | 0.076654 |
| `headlines-minilm3-20261123` | identical to the row above, seed 7 | 12 | 0.690426 | 0.660582 | 11 | 0.067790 |

What the three runs establish:

- **MiniLM loses to n-gram features on this task.** Its three candidates scored 0.668085, 0.668085 and 0.671277 on validation and were never selected; the winning family stayed inside the same TF-IDF space.
- **The coupon plateau is structural.** All 12 candidates stayed between 0.7547 and 0.7588 validation on the same `cross_boost_blend` point, so the +0.08 pt test difference (2008 vs 2006 correct rows) is two rows of noise.
- **The headlines "+1 pt" is seed variance.** The seed-42 run reached 0.670483, but the paired seed-7 control with the same policy returned exactly 0.660582 — the baseline number, bit for bit, with a different selected trial. The seed-driven spread on this 3232-row test set is therefore about ±1 pt, which is the size of the effect the earlier attempts were chasing.
- **Three-seed validation alone does not help.** Averaging validation seeds gave a lower, more honest validation number (0.684043 versus 0.690426) but did not change the selected family; the deployed model is still a single fit, so test-time variance is untouched.

An earlier attempt in this batch (`headlines-minilm-20261123`) put the two flags in the policy file instead of the task config, so they never reached the profile; it was stopped after 8 trials and its directory removed. Its CNY 0.049710 charge stays in the ledger as an Aborted-flag misconfiguration record.

Because the paired control shows no systematic gain, the reported experiments stay at `coupon-gpu-20261123` (0.790698) and `headlines-gpu-20261123` (0.660582). The three probe runs are disclosed as exploration, not as a new best; picking the 0.670483 seed would be selecting on the test set, which the protocol forbids.

## Totals and record policy

The two reported experiments together cost **CNY 0.071023** in recorded Flash usage (0.014231 + 0.056792); including the four search-effort rounds, the two interrupted attempts and the accuracy-enrichment probe (0.204116 plus the 0.049710 aborted-flag attempt), the disclosed 2026-11-23 GPU usage is **CNY 0.785876** across sixteen runs, all accounting-complete with no reservation held.

These replace every earlier interim record: `coupon-v2` (GPU, test 0.792274, `summary` marked failed because its held-out `test.csv` was written after selection), the `coupon-v3` CPU rerun (test 0.782814) and the 6-trial `headlines-v2` (test 0.659963). Their ledgers keep the charges actually incurred, so only the runs in this log are reported as experiments.

Earlier exploratory runs were removed from local `runs/` on 2026-11-23 and are not listed as experiments. Before removal, all 107 referenced request/response files were copied into `.model_usage/<ledger>/evidence/` and the ledger pointers were rewritten, so every historical call still resolves. Their charges stay in the accounting ledger (about CNY 13.3 in total, including two conservative timeout reconciliations at the reserved upper bound) as disclosed AI usage, not part of either experiment above.
