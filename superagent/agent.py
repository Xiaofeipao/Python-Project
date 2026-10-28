"""Experiment orchestration, resource deadlines and autonomous final selection."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .data import image_hashes, inspect_data
from .io import append_json, event, write_json
from .metrics import utility
from .llm_planner import AutonomousPlanner
from .submission import program_files, stage_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def environment():
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": versions}
    try:
        import torch

        result["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0}
    except Exception as exc:
        result["torch"] = {"error": str(exc)}
    return result


def snapshot_source(output):
    files = program_files(PROJECT_ROOT)
    digest = hashlib.sha256()
    for source in files:
        relative = source.relative_to(PROJECT_ROOT)
        content = source.read_bytes()
        digest.update(str(relative).replace("\\", "/").encode("utf-8") + b"\0" + content)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return digest.hexdigest()


def run_worker(request, folder, timeout, threads):
    folder.mkdir(parents=True, exist_ok=False)
    request = dict(request, result_path=str(folder / "result.json"))
    request_path = folder / "request.json"
    write_json(request_path, request)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = str(threads)
    env["PYTHONHASHSEED"] = str(request["config"]["seed"])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.monotonic()
    with open(folder / "stdout.log", "w", encoding="utf-8") as stdout, open(folder / "stderr.log", "w", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, "-m", "superagent.worker", str(request_path)],
                                   cwd=request.get("code_root", PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr)
        try:
            returncode = process.wait(timeout=max(0.001, timeout))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {"status": "timeout", "error": "Worker exceeded its wall-clock allocation",
                    "elapsed_seconds": time.monotonic() - started}
        except BaseException:
            process.kill()
            process.wait()
            raise
    result_path = folder / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if returncode and result.get("status") == "success":
            result = {"status": "failed", "error": f"Worker exit code {returncode} after writing success"}
    else:
        result = {"status": "failed", "error": f"Worker exited with code {returncode}; see stderr.log"}
    result["wall_seconds_including_startup"] = time.monotonic() - started
    return result


def train_agent(config, output):
    started = time.monotonic()
    deadline = started + config.time_budget_seconds
    search_deadline = started + config.time_budget_seconds * (1 - config.final_fit_reserve_fraction)
    output = Path(output).resolve()
    if output.is_relative_to(PROJECT_ROOT / "superagent"):
        raise ValueError("Run output cannot be placed inside the Python source package")
    output.mkdir(parents=True, exist_ok=False)
    history = []
    planner = None
    summary = {"status": "running", "metric": config.metric, "trials": 0}
    event(output, "run_started", config=config.to_dict())
    try:
        config.validate()
        submission = stage_record(config)
        summary["submission"] = submission
        if submission.get("unseen_deadline_utc"):
            window = datetime.fromisoformat(submission["unseen_deadline_utc"]) - datetime.now(timezone.utc)
            deadline = min(deadline, time.monotonic() + window.total_seconds())
            search_deadline = min(search_deadline, started + (deadline - started) * (1 - config.final_fit_reserve_fraction))
        source_hash = snapshot_source(output)
        if submission.get("frozen_runtime_sha256") and source_hash != submission["frozen_runtime_sha256"]:
            raise ValueError("Runtime snapshot does not match the frozen submission")
        env = environment()
        write_json(output / "environment.json", env)
        train, validation, profile, split, hashes = inspect_data(config)
        evaluation_id = config.evaluation_id or "development-data-" + hashes["train_csv"]
        planner = AutonomousPlanner(config, output, evaluation_id)
        profile["torch_available"] = config.enable_torch and "version" in env["torch"]
        cuda_enabled = profile["torch_available"] and env["torch"].get("cuda_available") and config.device != "cpu"
        if config.compute_profile == "gpu_large" and not cuda_enabled:
            raise ValueError("gpu_large requires enabled PyTorch and an available CUDA device")
        profile["compute_profile"] = ("gpu_large" if cuda_enabled else "standard") if config.compute_profile == "auto" else config.compute_profile
        if profile["data_type"] == "image":
            hashes["images"] = image_hashes([train, validation], profile["image_column"], config.image_root)
        write_json(output / "profile.json", profile)
        manifest = {"config": config.to_dict(), "source_sha256": source_hash,
                    "data_sha256": hashes, "split": split,
                    "test_data_read_during_training": False,
                    "test_access_policy": "Formal test inference only after final solution selection; never used by search",
                    "submission": submission, "model_usage": "model_usage.json",
                    "resource_accounting": "wall-clock, configured threads, CUDA training memory and official-standard-price LLM token costs"}
        write_json(output / "manifest.json", manifest)
        cache = output / "_split.joblib"
        joblib.dump({"train": train, "validation": validation}, cache)
        event(output, "data_analyzed", profile=profile)
        base_request = {"config": config.to_dict(), "profile": profile, "data_path": str(cache),
                        "code_root": str(output / "source_snapshot"),
                        "feature_cache": str(output / "preprocessing_cache") if config.cache_preprocessing else None}
        for trial_number in range(1, config.max_trials + 1):
            remaining = search_deadline - time.monotonic()
            if remaining <= 0:
                event(output, "search_stopped", reason="Search allocation exhausted; preserve final-fit time")
                break
            candidate, reason = planner.choose(profile, history, config.metric, search_deadline)
            if candidate is None:
                event(output, "search_stopped", reason=reason)
                break
            remaining = search_deadline - time.monotonic()
            if remaining <= 0:
                event(output, "search_stopped", reason="Planning used the remaining search allocation")
                break
            folder = output / "experiments" / f"trial_{trial_number:03d}"
            event(output, "trial_started", trial=trial_number, candidate=candidate, reason=reason)
            result = run_worker(dict(base_request, candidate=candidate, mode="trial"), folder,
                                min(config.trial_timeout_seconds, remaining), config.threads)
            record = {"trial": trial_number, "candidate": candidate, "reason": reason, **result}
            history.append(record)
            append_json(output / "trials.jsonl", record)
            event(output, "trial_finished", **record)
        successful = [h for h in history if h["status"] == "success"]
        successful.sort(key=lambda h: utility(h["score"], config.metric), reverse=True)
        if not successful:
            raise RuntimeError("No successful validation trial; inspect experiment diagnostics")
        usage = planner.export_usage()
        if not usage["accounting_complete"] or not usage["within_budget"]:
            raise RuntimeError("Model usage is unreconciled or over budget; cannot publish a final solution")
        selected = None
        for rank, record in enumerate(successful, 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            metadata = {"schema_version": 1, "config": config.to_dict(), "profile": profile,
                        "candidate": record["candidate"], "validation_score": record["score"],
                        "source_sha256": source_hash, "environment": env, "data_sha256": hashes,
                        "validation_score_is_not_final_test_score": True,
                        "submission": submission, "model_usage": usage,
                        "selected_training_resources": record.get("training_resources")}
            event(output, "final_fit_started", rank=rank, trial=record["trial"],
                  candidate=record["candidate"], validation_score=record["score"])
            result = run_worker(dict(base_request, candidate=record["candidate"], mode="final",
                                     metadata=metadata, model_path=str(output / "model.joblib")),
                                output / "experiments" / f"final_{rank:03d}", remaining, config.threads)
            event(output, "final_fit_finished", rank=rank, **result)
            if result["status"] == "success":
                selected = record
                break
            # A timeout can occur just after a worker saves its model; failed
            # final fits must never leave a model that looks ready for use.
            for name in ("model.joblib", "model.joblib.tmp"):
                abandoned = output / name
                if abandoned.exists():
                    abandoned.unlink()
        if selected is None:
            raise RuntimeError("Final refit failed or time budget expired; validation records remain available")
        summary.update(status="completed", selected_trial=selected["trial"],
                       selected_candidate=selected["candidate"], validation_score=selected["score"],
                       model_path=str(output / "model.joblib"), final_test_score=None)
        if config.evaluation_stage != "development" and config.test_csv:
            from .prediction import predict

            event(output, "final_prediction_started", model_path=str(output / "model.joblib"))
            summary["prediction"] = predict(output / "model.joblib", config.test_csv, output / "predictions.csv")
            event(output, "final_prediction_finished", prediction=summary["prediction"])
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=str(exc))
        event(output, "run_failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        accounting_error = None
        if planner is not None:
            try:
                summary["model_usage"] = planner.export_usage()
                if summary["status"] == "completed" and (
                    not summary["model_usage"]["accounting_complete"] or not summary["model_usage"]["within_budget"]
                ):
                    raise RuntimeError("Final shared model-usage accounting is incomplete or over budget")
            except Exception as exc:
                if summary["status"] == "completed":
                    summary.update(status="failed", error=str(exc))
                    accounting_error = exc
                else:
                    summary["accounting_error"] = str(exc)
        if summary["status"] != "completed" and (output / "model.joblib").exists():
            accounting = summary.get("model_usage", {})
            name = "partial_model.joblib" if accounting.get("accounting_complete") and accounting.get("within_budget") else "unverified_model.joblib"
            (output / "model.joblib").replace(output / name)
            summary["partial_model_path"] = str(output / name)
            summary.pop("model_path", None)
        summary.update(trials=len(history), elapsed_seconds=time.monotonic() - started)
        write_json(output / "summary.json", summary)
        event(output, "run_finished", **summary)
        # Keep split indices/hashes in manifest; avoid retaining a second full dataset.
        cache = output / "_split.joblib"
        if cache.exists():
            cache.unlink()
        if accounting_error is not None:
            raise accounting_error
