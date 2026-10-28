"""Verify final development artifacts without changing fitted models."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from superagent.agent import run_worker
from superagent.data import read_csv
from superagent.prediction import evaluate, load_solution, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="development_final", help="Directory name under runs/")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in {".", ".."}:
        raise ValueError("--run must be a directory name")
    directory = ROOT / "runs" / args.run
    benchmark = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    source_versions = set()
    report = {"status": "passed", "checks": {}, "gpu_trials": []}
    for modality in ("tabular", "text", "image"):
        run = directory / modality
        artifact_path = run / "model.joblib"
        model, metadata = load_solution(artifact_path)
        source_versions.add(metadata["source_sha256"])
        source = run / "source_snapshot"
        for path in source.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".txt"}:
                assert path.read_bytes() == (ROOT / path.relative_to(source)).read_bytes(), f"Changed source: {path}"
        data = read_csv(metadata["config"]["test_csv"], "id", "label")
        saved = pd.read_csv(run / "predictions.csv", dtype={"id": "string", "prediction": "string"})
        np.testing.assert_array_equal(saved["prediction"], model.predict(data).astype(str))
        np.testing.assert_array_equal(saved["id"], data["id"])
        np.testing.assert_allclose(saved.filter(like="probability_").to_numpy().sum(axis=1), 1, atol=1e-6)
        trials = [json.loads(line) for line in (run / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(trial["status"] == "success" for trial in trials), f"Failed trial in {modality}"
        for trial in trials:
            if trial["candidate"]["family"].startswith("torch_"):
                resources = trial["training_resources"]
                assert resources["device"] == "cuda"
                assert resources["peak_cuda_allocated_bytes"] > 0
                assert not trial["warnings"], f"Unexpected CUDA warnings: {trial['warnings']}"
                report["gpu_trials"].append({"modality": modality, "trial": trial["trial"],
                                             "score": trial["score"], "resources": resources})
        estimator = model.named_steps["classifier"]
        cpu_gpu_difference = None
        strict_cpu_gpu_difference = None
        if hasattr(estimator, "training_report_"):
            import torch

            gpu_probabilities = model.predict_proba(data)
            estimator.set_params(device="cpu")
            cpu_probabilities = model.predict_proba(data)
            cpu_gpu_difference = float(np.max(np.abs(cpu_probabilities - gpu_probabilities)))
            # CUDA convolution may use TF32 for float32 inference. Preserve
            # its fast deployment path, but separately verify strict math.
            np.testing.assert_array_equal(cpu_probabilities.argmax(1), gpu_probabilities.argmax(1))
            np.testing.assert_allclose(cpu_probabilities, gpu_probabilities, atol=1e-3, rtol=1e-3)
            estimator.set_params(device="cuda")
            original_tf32 = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = False
                strict_probabilities = model.predict_proba(data)
            finally:
                torch.backends.cudnn.allow_tf32 = original_tf32
            strict_cpu_gpu_difference = float(np.max(np.abs(cpu_probabilities - strict_probabilities)))
            np.testing.assert_allclose(cpu_probabilities, strict_probabilities, atol=1e-4, rtol=1e-4)
            resources = estimator.training_report_
            for member in resources.get("members", [resources]):
                assert member["internal_monitor_rows"] == 0
                assert member["training_rows"] == metadata["profile"]["rows"] + len(
                    read_csv(metadata["config"]["validation_csv"], "id", "label"))
        for action, kwargs in (
            (evaluate, {"model_path": artifact_path, "input_path": metadata["config"]["train_csv"],
                        "output_path": run / "forbidden_train_evaluation.json"}),
            (predict, {"model_path": artifact_path, "input_path": metadata["config"]["test_csv"],
                       "output_path": artifact_path}),
        ):
            try:
                action(**kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError("Expected leakage/overwrite guard to reject operation")
        report["checks"][modality] = {
            "trials_succeeded": len(trials), "prediction_rows": len(saved),
            "reloaded_predictions_match": True, "source_snapshot_matches_workspace": True,
            "leakage_and_overwrite_guards": True,
            "selected_model_cpu_portability_checked": hasattr(estimator, "training_report_"),
            "cpu_gpu_max_probability_difference": cpu_gpu_difference,
            "cpu_gpu_strict_max_probability_difference": strict_cpu_gpu_difference,
            "validation_accuracy": benchmark["tasks"][modality]["training"]["validation_score"],
            "development_holdout_accuracy": benchmark["tasks"][modality]["evaluation"]["score"],
        }
    assert len(source_versions) == 1, "Final tasks used different source versions"
    report["source_sha256"] = next(iter(source_versions))
    request = {"config": {"seed": 42}, "code_root": str(directory / "image" / "source_snapshot")}
    timeout = run_worker(request, directory / "timeout_check", timeout=0.001, threads=1)
    assert timeout["status"] == "timeout", timeout
    report["timeout_process_terminated"] = True
    (directory / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "gpu_trial_count": len(report["gpu_trials"]),
                      "timeout_process_terminated": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
