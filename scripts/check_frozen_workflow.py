"""Exercise released/unseen orchestration on local development data, without LLM calls."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from superagent.submission import freeze_submission, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy = output / "smoke_policy.json"
    policy.write_text(json.dumps({"max_trials": 3, "time_budget_seconds": 180,
                                  "trial_timeout_seconds": 60, "threads": 2,
                                  "device": "cuda", "compute_profile": "gpu_large",
                                  "llm_mode": "disabled"}), encoding="utf-8")
    bundle = output / "frozen_development_copy"
    frozen = freeze_submission(policy, bundle)
    verify_bundle(frozen["manifest"])
    data = ROOT / "data" / "development"
    # Remove labels from the held-out input to check genuine prediction input.
    prediction_input = output / "unlabelled_test.csv"
    expected = pd.read_csv(data / "tabular_test.csv", dtype={"id": "string"})
    expected.drop(columns="label").to_csv(prediction_input, index=False)
    environment = os.environ.copy()
    environment["SUPERAGENT_USAGE_ROOT"] = str(output / "shared_usage")
    report = {"purpose": "Development workflow verification, not course results",
              "runtime_sha256": frozen["runtime_sha256"], "stages": {}}
    for stage in ("released", "unseen"):
        config = {"evaluation_stage": stage, "evaluation_id": f"development-smoke-{stage}",
                  "frozen_manifest": frozen["manifest"],
                  "train_csv": str(data / "tabular_train.csv"),
                  "validation_csv": str(data / "tabular_validation.csv"),
                  "test_csv": str(prediction_input), "target_column": "label",
                  "id_column": "id", "data_type": "tabular", "metric": "accuracy"}
        if stage == "unseen":
            # Simulated release for development only, never an official timestamp.
            config["unseen_released_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        config_path = output / f"{stage}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        run = output / stage
        process = subprocess.run([sys.executable, str(bundle / "main.py"), "train",
                                  "--config", str(config_path), "--output", str(run)],
                                 cwd=bundle, env=environment, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace")
        (output / f"{stage}.console.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"{stage} failed; inspect {output / (stage + '.console.log')}")
        summary = json.loads(process.stdout)
        assert summary["status"] == "completed"
        assert summary["submission"]["frozen_runtime_sha256"] == frozen["runtime_sha256"]
        predictions = pd.read_csv(run / "predictions.csv", dtype={"id": "string"})
        assert predictions["id"].tolist() == expected["id"].tolist()
        assert predictions["prediction"].notna().all()
        usage = summary["model_usage"]
        assert usage["call_count"] == 0 and usage["actual_total_cost_rmb"] == "0"
        assert usage["within_budget"] and usage["accounting_complete"]
        assert summary["submission"]["evaluation_stage"] == stage
        if stage == "unseen":
            assert "unseen_deadline_utc" in summary["submission"]
        report["stages"][stage] = {"status": "passed", "prediction_rows": len(predictions),
                                    "automatic_prediction": True, "input_has_no_labels": True,
                                    "llm_calls": 0, "accounting_complete": True}
        print(f"{stage}: passed; {len(predictions)} automatic predictions", flush=True)
    verify_bundle(frozen["manifest"])
    report["status"] = "passed"
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
