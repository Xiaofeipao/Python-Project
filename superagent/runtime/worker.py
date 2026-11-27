"""Isolated worker. Launched only by an explicit train command."""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path


def fit_with_fallback(model, X, y):
    """Retry only recognizable allocation failures, without reusing fitted weights."""
    from sklearn.base import clone
    recoveries = []
    for attempt in range(3):
        try:
            from joblib import parallel_config
            with parallel_config(prefer="threads"):
                model.fit(X, y)
            return model, recoveries
        except Exception as exc:
            if 'out of memory' not in str(exc).lower() or attempt == 2:
                raise
            params = model.get_params()
            updates = {k: max(1, v // 2) for k, v in params.items()
                       if k.endswith('batch_size') and isinstance(v, int)} if attempt == 0 else {}
            if not updates:
                updates = {k: 'cpu' for k in params if k.endswith('device')}
            if not updates:
                raise
            recoveries.append({'error': 'out of memory', 'changes': updates})
            model = clone(model).set_params(**updates)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result_path = request["result_path"]
    started = time.monotonic()
    try:
        # Agent sets native thread environment variables before this process starts.
        import joblib
        import pandas as pd

        from superagent.config import TaskConfig
        from superagent.preprocessing.datasets import labels
        from superagent.common.json_io import write_json
        from superagent.evaluation.metrics import score_model
        from superagent.models.factory import build_model
        from superagent.runtime.checkpoint import save_solution

        config = TaskConfig(**request["config"])
        from superagent.governance.test_access import install_test_guard
        install_test_guard(config.test_csv)
        data = joblib.load(request["data_path"])
        train, validation = data["train"], data["validation"]
        candidate = request["candidate"]
        model = build_model(request["profile"], candidate, config)
        model.memory = request.get("feature_cache")
        estimator = model.named_steps['classifier']
        if hasattr(estimator, 'cache_dir') and request.get('feature_cache'):
            estimator.set_params(cache_dir=str(Path(request['feature_cache']) / 'members'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if request["mode"] == "trial":
                model, recoveries = fit_with_fallback(model, train, labels(train, config.target_column))
                score = score_model(model, validation, labels(validation, config.target_column),
                                    config.metric, config.positive_label)
                result = {"status": "success", "score": score}
                import numpy as np
                from dataclasses import replace
                scores = [score]
                for index in range(1, config.stability_repeats):
                    repeated_config = replace(config, seed=config.seed + 101 * index)
                    repeated = build_model(request['profile'], candidate, repeated_config)
                    repeated.memory = request.get('feature_cache')
                    if hasattr(repeated.named_steps['classifier'], 'cache_dir') and request.get('feature_cache'):
                        repeated.named_steps['classifier'].set_params(cache_dir=str(Path(request['feature_cache']) / 'members'))
                    repeated, _ = fit_with_fallback(repeated, train, labels(train, config.target_column))
                    scores.append(score_model(repeated, validation, labels(validation, config.target_column), config.metric, config.positive_label))
                result.update(score=float(np.mean(scores)), validation_std=float(np.std(scores)),
                              seed_scores=scores, seeds=[config.seed + 101*i for i in range(len(scores))],
                              uncertainty_scope='training seeds on the same official validation split; not unseen generalization')
                if config.nested_cv_folds:
                    from superagent.evaluation.nested_cv import nested_diagnostic
                    result['nested_validation']=nested_diagnostic(train,request['profile'],candidate,config)
                # Keep the first-seed fitted pipeline as a last-resort solution.
                # Only train rows were fitted; no test input is accessed here.
                destination = Path(result_path).parent / 'validated_model.joblib'
                metadata = dict(request['metadata'], partial=True, fit_scope='training_split_only',
                                checkpoint_seed=config.seed, validation_seed_score=score)
                save_solution(model, metadata, destination)
                from superagent.preprocessing.datasets import file_hash
                result.update(checkpoint_path=str(destination), checkpoint_sha256=file_hash(destination))
            else:
                all_data = pd.concat([train, validation], ignore_index=True)
                selected_resources = request["metadata"].get("selected_training_resources")
                if selected_resources and "best_epoch" in selected_resources:
                    # Refit on every allowed row for the train-only stopping
                    # epoch(s), without consulting the designated test set.
                    model.named_steps["classifier"].set_params(refit_epochs=selected_resources["best_epoch"])
                model, recoveries = fit_with_fallback(model, all_data, labels(all_data, config.target_column))
                destination = Path(request["model_path"])
                save_solution(model, dict(request['metadata'], partial=False,
                                         fit_scope='training_and_validation'), destination)
                result = {"status": "success", "model_path": str(destination)}
        estimator = model.named_steps["classifier"]
        result['hardware_recoveries'] = recoveries
        result['feature_resources'] = {name: step.resource_report_ for name, step in model.steps
                                       if hasattr(step, 'resource_report_')}
        if hasattr(estimator, "training_report_"):
            result["training_resources"] = estimator.training_report_
        elif hasattr(estimator, "resource_report_"):
            result["training_resources"] = estimator.resource_report_
        result["warnings"] = list(dict.fromkeys(str(item.message) for item in caught))[:30]
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(result_path, result)
        return 0
    except Exception as exc:
        from superagent.common.json_io import write_json

        write_json(result_path, {"status": "failed", "error": str(exc),
                                 "traceback": traceback.format_exc(),
                                 "elapsed_seconds": time.monotonic() - started})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
