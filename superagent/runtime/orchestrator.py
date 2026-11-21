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

from superagent.preprocessing.datasets import image_hashes, inspect_data
from superagent.common.json_io import append_json, event, write_json
from superagent.evaluation.metrics import utility
from superagent.planning.llm import AutonomousPlanner
from superagent.governance.frozen_bundle import program_files, stage_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def environment():
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "Pillow", "catboost", "lightgbm", "xgboost", "sentence-transformers", "transformers", "huggingface-hub", "tokenizers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": versions}
    result['installed_distributions'] = {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
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
        process = subprocess.Popen([sys.executable, "-m", "superagent.runtime.worker", str(request_path)],
                                   cwd=request.get("code_root", PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr)
        write_json(folder / 'worker_pid.json', {'pid':process.pid})
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


def train_agent(config, output, resume_from=None):
    from contextlib import ExitStack
    from superagent.runtime.checkpoint import run_lock
    with ExitStack() as locks:
        if resume_from:
            locks.enter_context(run_lock(resume_from))
        locks.enter_context(run_lock(output))
        return _train_agent(config,output,resume_from)


def _train_agent(config, output, resume_from=None):
    from superagent.runtime.checkpoint import load_resume
    resumed = load_resume(resume_from, config) if resume_from else None
    wall_started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    if resumed:
        started -= resumed['elapsed']
    deadline = started + config.time_budget_seconds
    search_deadline = started + config.time_budget_seconds * (1 - config.final_fit_reserve_fraction)
    output = Path(output).resolve()
    if output.is_relative_to(PROJECT_ROOT / "superagent"):
        raise ValueError("Run output cannot be placed inside the Python source package")
    output.mkdir(parents=True, exist_ok=False)
    history = list(resumed['history']) if resumed else []
    planner = None
    summary = {"status": "running", "metric": config.metric, "trials": 0,
               "test_set_used_for_selection": False, "partial": True}
    event(output, "run_started", config=config.to_dict())
    from superagent.governance.test_access import install_test_guard
    test_access = install_test_guard(config.test_csv if not config.progressive_output else None)
    try:
        config.validate()
        submission = stage_record(config)
        summary["submission"] = submission
        if submission.get("unseen_deadline_utc"):
            window = datetime.fromisoformat(submission["unseen_deadline_utc"]) - datetime.now(timezone.utc)
            deadline = min(deadline, time.monotonic() + window.total_seconds())
            search_deadline = min(search_deadline, started + (deadline - started) * (1 - config.final_fit_reserve_fraction))
        source_hash = snapshot_source(output)
        if resumed and source_hash != resumed['manifest']['source_sha256']:
            raise ValueError('Resume source hash differs from original run')
        if submission.get("frozen_runtime_sha256") and source_hash != submission["frozen_runtime_sha256"]:
            raise ValueError("Runtime snapshot does not match the frozen submission")
        env = environment()
        write_json(output / "environment.json", env)
        train, validation, profile, split, hashes = inspect_data(config)
        from superagent.evaluation.split_overlap import split_risk
        profile['split_risk']=split_risk(train,validation,profile)
        write_json(output / 'split_risk.json',profile['split_risk'])
        if profile['data_type']=='image':
            hashes['images']=image_hashes([train,validation],profile['image_column'],config.image_root)
        if resumed and hashes != resumed['manifest']['data_sha256']:
            raise ValueError('Resume training data or validation split changed')
        if resumed:
            if env['packages'] != resumed['environment']['packages'] or env['python'] != resumed['environment']['python']:
                raise ValueError('Resume Python/dependency environment changed')
            if env.get('installed_distributions') != resumed['environment'].get('installed_distributions') or env.get('torch',{}).get('version') != resumed['environment'].get('torch',{}).get('version'):
                raise ValueError('Resume installed distributions changed')
            for previous in history:
                append_json(output / 'trials.jsonl', previous)
        evaluation_id = config.evaluation_id or "development-data-" + hashes["train_csv"]
        planner = AutonomousPlanner(config, output, evaluation_id, resumed['ledger_root'] if resumed else None)
        if resumed and str(planner.ledger.path.resolve()) != resumed['ledger_path']:
            raise ValueError('Resume evaluation identity differs from original ledger')
        if resumed:
            planner._invalid_plans = resumed['planner'].get('consecutive_invalid_plans', 0)
            planner._last_rejection = resumed['planner'].get('last_rejection')
            planner._restored_reflections = resumed['reflections']
            planner._resume_skip_api_once = config.llm_mode=='required'
            event(output, 'resume_verified', parent=str(resumed['folder']), restored_trials=len(history), elapsed_including_downtime=resumed['elapsed'],restored_invalid_plans=planner._invalid_plans)
        profile["torch_available"] = config.enable_torch and "version" in env["torch"]
        profile['minilm_available'] = config.enable_minilm and bool(env['packages'].get('sentence-transformers'))
        profile['feature_search_enabled'] = config.enable_feature_search
        cuda_enabled = profile["torch_available"] and env["torch"].get("cuda_available") and config.device != "cpu"
        if config.compute_profile == "gpu_large" and not cuda_enabled:
            event(output, 'hardware_fallback', reason='CUDA unavailable; use CPU search')
            config.compute_profile = 'auto'
            config.device = 'auto'
        profile["compute_profile"] = ("gpu_large" if cuda_enabled else "standard") if config.compute_profile == "auto" else config.compute_profile
        write_json(output / "profile.json", profile)
        manifest = {"config": config.to_dict(), "source_sha256": source_hash,
                    "data_sha256": hashes, "split": split,
                    "test_data_read_during_training": bool(config.test_csv and config.progressive_output),
                    "test_access_policy": "Partial feature-only inference may follow checkpoint selection; test labels/scores never enter search. Final inference follows final selection.",
                    "submission": submission, "model_usage": "model_usage.json",
                    "resource_accounting": "wall-clock, configured threads, CUDA training memory and official-standard-price LLM token costs"}
        write_json(output / "manifest.json", manifest)
        manifest['original_started_at'] = resumed['origin'] if resumed else wall_started_at
        write_json(output / 'manifest.json', manifest)
        cache = output / "_split.joblib"
        joblib.dump({"train": train, "validation": validation}, cache)
        event(output, "data_analyzed", profile=profile)
        event(output, 'modality_inferred', requested=config.data_type, inferred=profile['data_type'],
              evidence=profile.get('modality_evidence', 'Explicit task configuration'))
        base_request = {"config": config.to_dict(), "profile": profile, "data_path": str(cache),
                        "code_root": str(output / "source_snapshot"),
                        "feature_cache": str(output / "preprocessing_cache") if config.cache_preprocessing else None}
        for trial_number in range(len(history) + 1, config.max_trials + 1):
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
            record['environment_ref'] = 'environment.json'
            record['source_sha256'] = source_hash
            history.append(record)
            from superagent.planning.memory import state
            write_json(output / 'trial_memory.json', state(history, config.metric))
            if config.search_patience and len(history) >= max(8, config.search_patience + 1):
                old = [h for h in history[:-config.search_patience] if h['status'] == 'success']
                recent = [h for h in history[-config.search_patience:] if h['status'] == 'success']
                if old and recent and max(utility(h['score'],config.metric) for h in recent) <= max(utility(h['score'],config.metric) for h in old):
                    event(output, 'search_stopped', reason='No improvement in validation seed means within patience; not proof of exhausted generalization', patience=config.search_patience)
                    append_json(output / 'trials.jsonl', record)
                    event(output, 'trial_finished', **record)
                    break
            append_json(output / "trials.jsonl", record)
            event(output, "trial_finished", **record)
            earlier = [h for h in history[:-1] if h['status'] == 'success']
            before = max((utility(h['score'], config.metric) for h in earlier), default=None)
            after = max((utility(h['score'], config.metric) for h in history if h['status'] == 'success'), default=None)
            event(output, 'feedback_assessed', trial=trial_number, candidate=candidate,
                  metric=config.metric, status=record['status'], validation_score=record.get('score'),
                  best_utility_before=before, best_utility_after=after,
                  utility_gain=None if before is None or after is None else after-before,
                  outcome='failed' if record['status'] != 'success' else
                          ('new_best' if before is None or after > before else 'no_improvement'),
                  evidence_source='trials.jsonl; validation only')
            previous = [h for h in history[:-1] if h['status'] == 'success']
            if (config.progressive_output and record['status'] == 'success' and
                (not previous or utility(record['score'], config.metric) > max(utility(h['score'], config.metric) for h in previous))):
                checkpoint = output / 'partial' / f'v{trial_number:03d}'
                partial_metadata = {'schema_version': 1, 'config': config.to_dict(), 'profile': profile,
                    'candidate': candidate, 'source_sha256': source_hash, 'submission': submission,
                    'partial': True, 'validation_score': record['score'], 'model_usage': planner.export_usage()}
                event(output, 'partial_selection_completed', trial=trial_number, candidate=candidate)
                allowance = min(config.trial_timeout_seconds, max(0, search_deadline-time.monotonic()) * .2)
                if allowance > 1:
                    result = run_worker(dict(base_request, candidate=candidate, mode='final',
                        metadata=partial_metadata, model_path=str(checkpoint / 'model.joblib')),
                        checkpoint, allowance, config.threads)
                    if result['status'] == 'success':
                        summary['partial_model_path'] = str(checkpoint / 'model.joblib')
                        summary['partial_version'] = trial_number
                        if config.test_csv:
                            from superagent.evaluation.inference import predict
                            event(output, 'partial_test_read_started', version=trial_number)
                            summary['partial_predictions'] = predict(checkpoint / 'model.joblib', config.test_csv,
                                                                     checkpoint / 'predictions.csv')
                        write_json(checkpoint / 'summary.json', dict(summary, status='partial'))
                        write_json(output / 'summary.json', summary)
                    event(output, 'partial_checkpoint_finished', version=trial_number, **result)
        successful = [h for h in history if h["status"] == "success"]
        successful.sort(key=lambda h: utility(h["score"], config.metric), reverse=True)
        if not successful:
            raise RuntimeError("No successful validation trial; inspect experiment diagnostics")
        usage = planner.export_usage()
        if (not usage["accounting_complete"] and not config.allow_provisional_accounting) or not usage["within_budget"]:
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
        event(output, 'selection_completed', trial=selected['trial'], candidate=selected['candidate'])
        test_access[0] = False
        summary.update(status="completed", selected_trial=selected["trial"],
                       partial=False,
                       selected_candidate=selected["candidate"], validation_score=selected["score"],
                       model_path=str(output / "model.joblib"), final_test_score=None)
        if config.test_csv:
            from superagent.evaluation.inference import predict

            event(output, "final_prediction_started", model_path=str(output / "model.joblib"))
            event(output, 'test_read_started', phase='final')
            summary["prediction"] = predict(output / "model.joblib", config.test_csv, output / "predictions.csv")
            event(output, "final_prediction_finished", prediction=summary["prediction"])
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=str(exc))
        event(output, "run_failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        test_access[0] = False
        accounting_error = None
        if planner is not None:
            try:
                summary["model_usage"] = planner.export_usage()
                if summary["status"] == "completed" and (
                    (not summary["model_usage"]["accounting_complete"] and not config.allow_provisional_accounting) or not summary["model_usage"]["within_budget"]
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
        summary['accounting_provisional'] = not summary.get('model_usage', {}).get('accounting_complete', False)
        summary['eligible_for_formal_submission'] = not summary['accounting_provisional']
        summary.update(trials=len(history), elapsed_seconds=time.monotonic() - started)
        write_json(output / "summary.json", summary)
        event(output, "run_finished", **summary)
        # Keep split indices/hashes in manifest; avoid retaining a second full dataset.
        cache = output / "_split.joblib"
        if cache.exists():
            cache.unlink()
        if accounting_error is not None:
            raise accounting_error
