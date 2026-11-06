"""Run a generic N_*/C_*/y_* NumPy classification task using its existing splits.

Only use allow_pickle=True with a trusted task package. Test arrays are opened
after autonomous model selection and refit, never while preparing training.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def convert_split(source, split, destination):
    columns = {}
    for prefix in ("N", "C"):
        path = source / f"{prefix}_{split}.npy"
        if not path.exists():
            continue
        values = np.load(path, allow_pickle=prefix == "C")
        if values.ndim != 2:
            raise ValueError("Feature arrays must be two-dimensional")
        for index in range(values.shape[1]):
            column = values[:, index]
            if prefix == "C":
                # Preserve categorical type even when every value looks numeric.
                column = [np.nan if pd.isna(v) else "c:" + str(v) for v in column]
            columns[f"{prefix}_{index}"] = column
    if not columns:
        raise ValueError("No N_ or C_ feature array found")
    frame = pd.DataFrame(columns)
    target = np.load(source / f"y_{split}.npy", allow_pickle=False)
    if target.ndim != 1 or len(frame) != len(target):
        raise ValueError("Feature and target rows must align")
    frame.insert(0, "id", [f"{split}_{index}" for index in range(len(frame))])
    frame["label"] = target
    frame.to_csv(destination, index=False)
    return {"rows": len(frame), "features": len(columns), "csv": str(destination)}


def command(output, *arguments):
    process = subprocess.run([sys.executable, str(ROOT / "main.py"), *map(str, arguments)],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / f"{arguments[0]}.console.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{arguments[0]} failed; inspect console log")
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--skip-test", action="store_true", help="Compare validation only; do not open test arrays")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "converted"
    data.mkdir()
    report = {"split_policy": "Use supplied splits unchanged; test opened only after final refit", "splits": {}}
    for split in ("train", "val"):
        report["splits"][split] = convert_split(source, split, data / f"{split}.csv")
    config = {"train_csv": str(data / "train.csv"), "validation_csv": str(data / "val.csv"),
              "target_column": "label", "id_column": "id", "data_type": "tabular", "metric": args.metric,
              "max_trials": 16, "time_budget_seconds": 600, "trial_timeout_seconds": 120,
              "threads": 4, "device": "auto", "compute_profile": "auto", "llm_mode": "disabled"}
    config_path = output / "task.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report["training"] = command(output, "train", "--config", config_path, "--output", output / "agent")
    if args.skip_test:
        report["test_evaluated"] = False
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "completed", "validation_score": report["training"]["validation_score"],
                          "test_evaluated": False}, indent=2))
        return
    print("Training completed; now opening designated test arrays", flush=True)
    report["splits"]["test"] = convert_split(source, "test", data / "test.csv")
    report["prediction"] = command(output, "predict", "--model", output / "agent/model.joblib",
                                    "--input", data / "test.csv", "--output", output / "predictions.csv")
    report["evaluation"] = command(output, "evaluate", "--model", output / "agent/model.joblib",
                                    "--input", data / "test.csv", "--output", output / "evaluation.json")
    report["original_array_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.npy")}
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "validation_score": report["training"]["validation_score"],
                      "test_score": report["evaluation"]["score"], "rows": report["evaluation"]["rows"]}, indent=2))


if __name__ == "__main__":
    main()
