"""Local development benchmark; no downloads and no course task results.

Baseline runs inspect validation scores only. Final runs additionally evaluate
a separate development holdout after autonomous model selection.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    cancer = load_breast_cancer(as_frame=True)
    tabular = cancer.data.copy()
    tabular["label"] = cancer.target.astype(str)
    # Exercise categorical, missing-value, and leading-zero ID handling too.
    tabular["bucket"] = np.where(np.arange(len(tabular)) % 3, "group_A", "group_B")
    tabular.loc[::19, tabular.columns[0]] = np.nan
    paths["tabular"] = tabular
    rng = np.random.default_rng(41)
    vocab = {
        "science": ["physics", "atom", "lab", "科学", "实验", "研究"],
        "sport": ["match", "team", "goal", "足球", "比赛", "运动"],
        "food": ["recipe", "cook", "taste", "烹饪", "美食", "厨房"],
    }
    texts = []
    for index in range(900):
        label = list(vocab)[index % 3]
        tokens = rng.choice(vocab[label], 5).tolist() + rng.choice(["today", "news", "讨论", "分享"], 4).tolist()
        rng.shuffle(tokens)
        texts.append({"text": " ".join(tokens), "label": label})
    paths["text"] = pd.DataFrame(texts)
    digits = load_digits()
    image_root = destination / "images"
    image_root.mkdir(exist_ok=True)
    for index, pixels in enumerate(digits.images):
        Image.fromarray(np.uint8(pixels / 16 * 255)).save(image_root / f"{index:04d}.png")
    paths["image"] = pd.DataFrame({"path": [f"{i:04d}.png" for i in range(len(digits.target))],
                                   "label": digits.target.astype(str)})
    for modality, frame in paths.items():
        frame.insert(0, "id", [f"{index:06d}" for index in range(len(frame))])
        fitting, test = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.label)
        train, validation = train_test_split(fitting, test_size=0.25, random_state=42, stratify=fitting.label)
        for name, data in (("train", train), ("validation", validation), ("test", test)):
            data.to_csv(destination / f"{modality}_{name}.csv", index=False)
        config = {
            "train_csv": f"{modality}_train.csv", "validation_csv": f"{modality}_validation.csv",
            "test_csv": f"{modality}_test.csv", "target_column": "label", "id_column": "id",
            "data_type": modality, "metric": "accuracy", "max_trials": 8,
            "time_budget_seconds": 300, "trial_timeout_seconds": 55, "threads": 2, "seed": 42,
        }
        if modality == "text":
            config["text_columns"] = ["text"]
        elif modality == "image":
            config.update(image_column="path", image_root="images")
        (destination / f"{modality}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def command(*args):
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), *map(str, args)], cwd=ROOT,
                            text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args}\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "optimized"], required=True)
    parser.add_argument("--name", help="New output directory name under runs/")
    parser.add_argument("--compute-profile", choices=["standard", "gpu_large"], default="standard")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()
    data = ROOT / "data" / "development"
    if not (data / "image.json").exists():
        prepare(data)
    run_name = args.name or f"development_{args.phase}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--name must be a single directory name")
    output = ROOT / "runs" / run_name
    output.mkdir(parents=True, exist_ok=False)
    results = {"purpose": "Development checks only, not course results", "phase": args.phase, "tasks": {}}
    for modality in ("tabular", "text", "image"):
        run = output / modality
        config = json.loads((data / f"{modality}.json").read_text(encoding="utf-8"))
        config["enable_torch"] = args.phase == "optimized"
        config["compute_profile"] = args.compute_profile
        if args.compute_profile == "gpu_large":
            config.update(max_trials=16, time_budget_seconds=1800, trial_timeout_seconds=300,
                          neural_batch_size=256, mixed_precision=True, device="cuda")
        if args.max_trials is not None:
            config["max_trials"] = args.max_trials
        if args.budget is not None:
            config["time_budget_seconds"] = args.budget
        config_path = data / f"{modality}.{args.phase}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        trained = command("train", "--config", config_path, "--output", run)
        result = {"training": trained}
        if args.phase == "optimized":
            result["prediction"] = command("predict", "--model", run / "model.joblib", "--input",
                                           data / f"{modality}_test.csv", "--output", run / "predictions.csv")
            result["evaluation"] = command("evaluate", "--model", run / "model.joblib", "--input",
                                           data / f"{modality}_test.csv", "--output", run / "evaluation.json")
        results["tasks"][modality] = result
        print(f"{modality}: validation={trained['validation_score']:.6f}; selected={trained['selected_candidate']}", flush=True)
        (output / "benchmark.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
