"""CLI imports ML modules lazily, so --help does not start data processing."""

import argparse
import importlib
import importlib.metadata
import json
import sys


def parser():
    root = argparse.ArgumentParser(description="AIAA 3102 General Algorithm SuperAgent")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Autonomously search and refit a classification solution")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True, help="New directory for this autonomous run")
    freeze = commands.add_parser("freeze", help="Create an immutable submission bundle without training")
    freeze.add_argument("--policy", required=True, help="JSON algorithm policy; no task-specific input data")
    freeze.add_argument("--output", required=True, help="New frozen bundle directory")
    verify = commands.add_parser("verify-freeze", help="Check the submission bundle and current runtime hashes")
    verify.add_argument("--manifest", required=True)
    for name in ("predict", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True, help="Trusted local model.joblib")
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--image-root", default=None)
    commands.add_parser("doctor", help="Import dependencies and report the selected Python environment")
    return root


def doctor():
    report = {"executable": sys.executable, "python": sys.version, "packages": {}}
    success = True
    for distribution, module in (("numpy", "numpy"), ("pandas", "pandas"),
                                 ("scikit-learn", "sklearn"), ("scipy", "scipy"),
                                 ("joblib", "joblib"), ("Pillow", "PIL")):
        try:
            imported = importlib.import_module(module)
            report["packages"][distribution] = {
                "version": importlib.metadata.version(distribution),
                "module_version": getattr(imported, "__version__", None),
                "path": str(imported.__file__),
            }
        except Exception as exc:
            report["packages"][distribution] = {"error": str(exc)}
            success = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "freeze":
            from .submission import freeze_submission

            result = freeze_submission(args.policy, args.output)
        elif args.command == "verify-freeze":
            from .submission import verify_bundle

            manifest = verify_bundle(args.manifest)
            result = {"status": "verified", "runtime_sha256": manifest["runtime_sha256"]}
        elif args.command == "train":
            from .agent import train_agent
            from .config import TaskConfig

            result = train_agent(TaskConfig.load(args.config), args.output)
        else:
            from .prediction import evaluate, predict

            action = predict if args.command == "predict" else evaluate
            result = action(args.model, args.input, args.output, args.image_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; inspect the run directory for partial records.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
