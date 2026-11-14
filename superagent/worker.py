"""Isolated worker. Launched only by an explicit train command."""

import json
import os
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

        from .config import TaskConfig
        from .data import labels
        from .io import write_json
        from .metrics import score_model
        from .search import build_model

        config = TaskConfig(**request["config"])
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
            else:
                all_data = pd.concat([train, validation], ignore_index=True)
                selected_resources = request["metadata"].get("selected_training_resources")
                if selected_resources and "best_epoch" in selected_resources:
                    # Refit on every allowed row for the train-only stopping
                    # epoch(s), without consulting the designated test set.
                    model.named_steps["classifier"].set_params(refit_epochs=selected_resources["best_epoch"])
                model, recoveries = fit_with_fallback(model, all_data, labels(all_data, config.target_column))
                # Prediction has no dependency on the training disk cache.
                model.memory = None
                if hasattr(model.named_steps['classifier'], 'cache_dir'):
                    model.named_steps['classifier'].set_params(cache_dir=None)
                destination = Path(request["model_path"])
                temporary = destination.with_name(destination.name + ".tmp")
                joblib.dump({"model": model, "metadata": request["metadata"]}, temporary)
                os.replace(temporary, destination)
                result = {"status": "success", "model_path": str(destination)}
        estimator = model.named_steps["classifier"]
        result['hardware_recoveries'] = recoveries
        if hasattr(estimator, "training_report_"):
            result["training_resources"] = estimator.training_report_
        elif hasattr(estimator, "resource_report_"):
            result["training_resources"] = estimator.resource_report_
        result["warnings"] = list(dict.fromkeys(str(item.message) for item in caught))[:30]
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(result_path, result)
        return 0
    except Exception as exc:
        from .io import write_json

        write_json(result_path, {"status": "failed", "error": str(exc),
                                 "traceback": traceback.format_exc(),
                                 "elapsed_seconds": time.monotonic() - started})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
