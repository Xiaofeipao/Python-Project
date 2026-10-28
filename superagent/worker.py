"""Isolated worker. Launched only by an explicit train command."""

import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path


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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if request["mode"] == "trial":
                model.fit(train, labels(train, config.target_column))
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
                model.fit(all_data, labels(all_data, config.target_column))
                # Prediction has no dependency on the training disk cache.
                model.memory = None
                destination = Path(request["model_path"])
                temporary = destination.with_name(destination.name + ".tmp")
                joblib.dump({"model": model, "metadata": request["metadata"]}, temporary)
                os.replace(temporary, destination)
                result = {"status": "success", "model_path": str(destination)}
        estimator = model.named_steps["classifier"]
        if hasattr(estimator, "training_report_"):
            result["training_resources"] = estimator.training_report_
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
