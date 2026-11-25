# Current Experiment Catalog

This catalog includes the runs represented by the current ledger inventory. Original runs and ledgers remain unchanged. Current model pointers are in current_results.json; accounting definitions and totals are in ../accounting/README.md.

The reported pair is a development frontier across different frozen runtimes, not a single-version formal experiment. Released test scores were reused in development; cross-run promotion is distinct from within-run validation selection.

## coupon-ens-20261123

- Source: `runs/coupon-ens-20261123/` and `.model_usage/d1af299253efec2650ed3ee86b277be127a486d8801dcb7fd9306759cbffcc8b/ledger.json`.
- Recorded calls: 5; input tokens: 7434; output tokens: 4483.
- Recorded token cost: CNY 0.025366.
- Status counts: {"recorded": 5}.
- Run status: completed; validation score: 0.7596059113300493; trials: 6.
- Selected candidate: `{"family": "cross_boost_blend", "gpu_seeds": 3, "gpu_weight": 0.25, "leaf_weight": 0.0}`.

## coupon-gpu-20261123

- Source: `runs/coupon-gpu-20261123/` and `.model_usage/3244fbf72cf3ca566dd72baabf34614681a9ca88903377285296b0184875693a/ledger.json`.
- Recorded calls: 3; input tokens: 3915; output tokens: 2579.
- Recorded token cost: CNY 0.014231.
- Status counts: {"recorded": 3}.
- Run status: completed; validation score: 0.7596059113300493; trials: 4.
- Selected candidate: `{"family": "cross_boost_blend", "gpu_seeds": 3, "gpu_weight": 0.25, "leaf_weight": 0.0}`.

## coupon-minilm2-20261123

- Source: `runs/coupon-minilm2-20261123/` and `.model_usage/a8b40c719608e202ff90685c9976c6ef42048b1bebe2be531be04cd432e0e8b3/ledger.json`.
- Recorded calls: 11; input tokens: 21356; output tokens: 9579.
- Recorded token cost: CNY 0.059672.
- Status counts: {"recorded": 11}.
- Run status: completed; validation score: 0.7587848932676519; trials: 12.
- Selected candidate: `{"family": "cross_boost_blend", "gpu_seeds": 3, "gpu_weight": 0.275, "leaf_weight": 0.0}`.

## coupon-pro-20261123

- Source: `runs/coupon-pro-20261123/` and `.model_usage/b9f8b7191ccd06c4cbd8148c0e120782c3ec1ef3e2379029b72d54c4919c3b96/ledger.json`.
- Recorded calls: 2; input tokens: 2595; output tokens: 8192.
- Recorded token cost: CNY 0.1222695.
- Status counts: {"recorded": 2}.
- Run status: completed; validation score: 0.7596059113300493; trials: 6.
- Selected candidate: `{"family": "cross_boost_blend", "gpu_seeds": 3, "gpu_weight": 0.25, "leaf_weight": 0.0}`.

## coupon-t10-20261123

- Source: `runs/coupon-t10-20261123/` and `.model_usage/09226f2bed50ef5528ea0e122af7b3a21ad9538ff4bc1457f5ee2e3485db7967/ledger.json`.
- Recorded calls: 9; input tokens: 14541; output tokens: 10877.
- Recorded token cost: CNY 0.058049.
- Status counts: {"recorded": 9}.
- Run status: completed; validation score: 0.7596059113300493; trials: 10.
- Selected candidate: `{"family": "cross_boost_blend", "gpu_seeds": 3, "gpu_weight": 0.25, "leaf_weight": 0.0}`.

## coupon-t14r-20261123

- Source: `runs/coupon-t14r-20261123/` and `.model_usage/7dee1dfeaf6b8738c075c26290e1688cb36ef4306260258343731cc273d65ea2/ledger.json`.
- Recorded calls: 13; input tokens: 22316; output tokens: 10672.
- Recorded token cost: CNY 0.065004.
- Status counts: {"recorded": 13}.
- Run status: completed; validation score: 0.7596059113300493; trials: 14.
- Selected candidate: `{"family": "cross_boost_blend", "gpu_seeds": 3, "gpu_weight": 0.25, "leaf_weight": 0.0}`.

## coupon-v6-20261124

- Source: `runs/coupon-v6-20261124/` and `.model_usage/50a0627d3ba05cbcfb564e468cb30a66ee5f024da1ca33f120a94ac0dfeb0a6e/ledger.json`.
- Recorded calls: 11; input tokens: 20352; output tokens: 12371.
- Recorded token cost: CNY 0.069836.
- Status counts: {"recorded": 11}.
- Run status: completed; validation score: 0.7581280788177339; trials: 12.
- Selected candidate: `{"family": "cross_boost_blend", "gpu_seeds": 3, "gpu_weight": 0.55, "leaf_weight": 0.0}`.

## headlines-ens-20261123

- Source: `runs/headlines-ens-20261123/` and `.model_usage/7546d324c436d8f498b4e62af2221ee9ed1877fe33c3a31271cf7fa16fc7f620/ledger.json`.
- Recorded calls: 9; input tokens: 16829; output tokens: 10243.
- Recorded token cost: CNY 0.057801.
- Status counts: {"recorded": 9}.
- Run status: completed; validation score: 0.6904255319148936; trials: 10.
- Selected candidate: `{"C": 1.0, "analyzer": "char", "class_weight": "balanced", "family": "logistic", "representation": "separate"}`.

## headlines-gpu-20261123

- Source: `runs/headlines-gpu-20261123/` and `.model_usage/2b56df25aa65671771a8b04f88b6448b739bbd0e7991264a58344f0c7f12efe5/ledger.json`.
- Recorded calls: 9; input tokens: 16120; output tokens: 10168.
- Recorded token cost: CNY 0.056792.
- Status counts: {"recorded": 9}.
- Run status: completed; validation score: 0.6904255319148936; trials: 10.
- Selected candidate: `{"C": 1.0, "analyzer": "char", "class_weight": "balanced", "family": "logistic", "representation": "separate"}`.

## headlines-minilm2-20261123

- Source: `runs/headlines-minilm2-20261123/` and `.model_usage/7a8dab993a7322afc0ede8c02cdc2372c7afc44aa2f3a55def136098620304b8/ledger.json`.
- Recorded calls: 11; input tokens: 22446; output tokens: 13552.
- Recorded token cost: CNY 0.076654.
- Status counts: {"recorded": 11}.
- Run status: completed; validation score: 0.6840425531914893; trials: 12.
- Selected candidate: `{"C": 0.1, "family": "text_blend", "linear_weight": 0.5}`.

## headlines-minilm3-20261123

- Source: `runs/headlines-minilm3-20261123/` and `.model_usage/7931f8dabeb1848d09f70d15b7312c6619a7e1343ae2d06d594cf04a93665df5/ledger.json`.
- Recorded calls: 11; input tokens: 21534; output tokens: 11564.
- Recorded token cost: CNY 0.067790.
- Status counts: {"recorded": 11}.
- Run status: completed; validation score: 0.6904255319148936; trials: 12.
- Selected candidate: `{"C": 1.0, "analyzer": "char", "class_weight": "balanced", "family": "logistic", "representation": "separate"}`.

## headlines-pro2-20261123

- Source: `runs/headlines-pro2-20261123/` and `.model_usage/4da5a876483f86711e7bec787983192bf24752a77858423da8f53b68861666e3/ledger.json`.
- Recorded calls: 8; input tokens: 15471; output tokens: 28546.
- Recorded token cost: CNY 0.4549905.
- Status counts: {"recorded": 8}.
- Run status: completed; validation score: 0.6904255319148936; trials: 10.
- Selected candidate: `{"C": 1.0, "analyzer": "char", "class_weight": "balanced", "family": "logistic", "representation": "separate"}`.

## headlines-t15-20261123

- Source: `runs/headlines-t15-20261123/` and `.model_usage/b5c09f1afee7a90002f34d62f4aa014e37ac33610e1c1f36861273f317f4dcc5/ledger.json`.
- Recorded calls: 11; input tokens: 20576; output tokens: 13433.
- Recorded token cost: CNY 0.074308.
- Status counts: {"recorded": 11}.
- Run status: completed; validation score: 0.6904255319148936; trials: 15.
- Selected candidate: `{"C": 1.0, "analyzer": "char", "class_weight": "balanced", "family": "logistic", "representation": "separate"}`.

## headlines-t20r-20261123

- Source: `runs/headlines-t20r-20261123/` and `.model_usage/4d87c75a053ceee6d4fb5c20cf7d07fe67c63516dc604a04420759d8303934aa/ledger.json`.
- Recorded calls: 15; input tokens: 28783; output tokens: 15199.
- Recorded token cost: CNY 0.089579.
- Status counts: {"recorded": 15}.
- Run status: completed; validation score: 0.6904255319148936; trials: 20.
- Selected candidate: `{"C": 1.0, "analyzer": "char", "class_weight": "balanced", "family": "logistic", "representation": "separate"}`.

## headlines-v6-20261124

- Source: `runs/headlines-v6-20261124/` and `.model_usage/d96e60cf3ed5a01894bdfad73698ec0d89a4238521677a1d7d3d57ad07db7651/ledger.json`.
- Recorded calls: 5; input tokens: 8749; output tokens: 4703.
- Recorded token cost: CNY 0.027561.
- Status counts: {"recorded": 5}.
- Run status: completed; validation score: 0.6999999999999998; trials: 12.
- Selected candidate: `{"C": 1.0, "analyzer": "char_wb", "family": "logistic", "fit_intercept": false, "representation": "difference", "vectorizer_options": {"ngram_range": [3, 6]}}`.

## headlines-overlap-20261125

- Source: `runs/headlines-overlap-20261125/` and `.model_usage/689781efd26488f7e8b4ff30d046d2b0277a1c08f92b99140d03d8bd49f8ddc9/ledger.json`.
- Recorded calls: 11; input tokens: 21787; output tokens: 12917.
- Recorded token cost: CNY 0.073455.
- Status counts: {"recorded": 11}.
- Run status: completed; validation score: 0.7031914893617022; trials: 12; test score 0.698019801980198 (2256/3232); macro F1 0.698000.
- Selected candidate: `{"family": "pairwise_overlap", "C": 1.0, "fit_intercept": true, "vectorizer_options": {"ngram_range": [3, 6]}}`.
- Code additions in this run: `pairwise_overlap` (signed char_wb TF-IDF difference plus seven standardized word-level overlap statistics), `pairwise_blend` (char_wb/difference logistic blended with word/difference linear SVM; scored at or below the single model and removed from the kept tree), `fit_intercept`/`class_weight` as generated logistic variants, and a `C` rule for `minilm_logistic`. The winning pipeline now lives in `superagent/models/pairwise.py`; 67 unit tests passed after the extraction.
- Reproducibility: the run directory keeps `source_snapshot/`. A standalone refit returns 2255/3232 instead of 2256/3232 because L-BFGS stops at the same tolerance under a different BLAS thread count; the stored `model.joblib` and `predictions.csv` are the reported artifacts.
- Offline screens (validation only, train-fitted preprocessing, no API calls): overlap block 0.705319 vs 0.700000 baseline; overlap blended with baseline 0.701064; OOF stack 0.692553; word blend 0.700000; fine-tuned MiniLM cross-encoder 0.671277 validation / 0.656869 test; ngram and 60k-feature neighbours all 0.697-0.704255. The [3,6]/30k/C=1 configuration stayed best.

## Coupon untried-family screen (2026-11-25)

Six families absent from all earlier coupon runs were fitted on the official validation split with train-only preprocessing. Reference: `coupon-minilm2-20261123` at 0.758785.

| family | validation |
| --- | --- |
| lightgbm (800 iters) | 0.746305 |
| lightgbm (1200 iters, 63 leaves) | 0.755665 |
| catboost (1200, depth 8) | 0.753695 |
| catboost (1800, depth 10) | 0.758128 |
| catboost_ensemble | 0.758128 |
| tabular_blend | 0.751724 |
| extra_trees (best depth) | 0.753695 |
| random_forest (best depth) | 0.752709 |
| cosine kNN (k=30) | 0.698522 |

No family beats 0.758785. CatBoost already applies ordered target statistics to the 21 categorical columns internally, so explicit target encoding is covered; class crossings and kNN similarity do not move a platform whose validation label noise is about 0.9% and whose five leading schemes sit within two to three rows of each other.

## Current snapshot totals

16 ledgers, 144 calls, CNY 1.3933580 recorded cost, 0 missing-cost records, CNY 0 unresolved reservation. These totals cover the source set identified by the inventory; selected-run costs are already included.

No new training, inference, API calls or tests were performed to produce this catalog.
