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


def _minilm_available(config, profile, env):
    """Optional encoder: explicit opt-in, or automatic for any text task."""
    auto = profile.get('data_type') == 'text'
    return bool((config.enable_minilm or auto)
                and env.get('packages', {}).get('sentence-transformers'))


def environment():
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "Pillow", "catboost", "lightgbm", "xgboost", "torchvision", "sentence-transformers", "transformers", "huggingface-hub", "tokenizers", "safetensors"):
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
    # Workers execute fits only. The one planner in this controller owns calls.
    env.pop(request['config'].get('llm_api_key_env', 'HKUST_API_KEY'), None)
    env.pop('HKUST_API_KEY', None)
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
    from dataclasses import replace
    from superagent.runtime.checkpoint import run_lock
    with ExitStack() as locks:
        if resume_from:
            locks.enter_context(run_lock(resume_from))
        locks.enter_context(run_lock(output))
        return _train_agent(replace(config),output,resume_from)


def _train_agent(config, output, resume_from=None):
    from superagent.runtime.checkpoint import load_resume
    requested_config = config.to_dict()
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
    test_access = install_test_guard(config.test_csv)
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
        (output / 'requirements.snapshot.txt').write_text(''.join(
            f'{name}=={version}\n' for name, version in sorted(env['installed_distributions'].items())), encoding='utf-8')
        train, validation, profile, split, hashes = inspect_data(config)
        profile['metric'] = config.metric
        profile['fitting_class_counts'] = {str(k): int(v) for k, v in train[config.target_column].value_counts().items()}
        from superagent.evaluation.split_overlap import split_risk
        profile['split_risk']=split_risk(train,validation,profile)
        write_json(output / 'split_risk.json',profile['split_risk'])
        # A small official validation split is noisy across 24 trials. Use two
        # deterministic training seeds, and a small nested diagnostic when the
        # training set itself is tiny; both use only training rows and never
        # touch the designated test set.
        validation_rows = len(validation)
        if validation_rows < 300 and config.stability_repeats == 1:
            config.stability_repeats = 2
            event(output, 'stability_auto_enabled', validation_rows=validation_rows,
                  stability_repeats=2)
        if (validation_rows < 100 and len(train) <= config.nested_cv_max_rows
                and config.nested_cv_folds == 0
                and config.evaluation_stage in {'released', 'unseen'}):
            config.nested_cv_folds = 2
            event(output, 'nested_cv_auto_enabled', validation_rows=validation_rows,
                  training_rows=len(train), folds=2)

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
            planner._repair_used = resumed['planner'].get('repair_used', False)
            planner._restored_reflections = resumed['reflections']
            planner._resume_skip_api_once = config.llm_mode=='required'
            event(output, 'resume_verified', parent=str(resumed['folder']), restored_trials=len(history), elapsed_including_downtime=resumed['elapsed'],restored_invalid_plans=planner._invalid_plans)
        profile["torch_available"] = config.enable_torch and "version" in env["torch"]
        # A hidden text task gets the optional frozen sentence encoder without
        # requiring a task-specific config flag. Explicit enable_minilm stays
        # valid for development experiments; tabular tasks do not pay this cost.
        profile['minilm_available'] = _minilm_available(config, profile, env)
        # The broader frozen-encoder registry shares the same optional
        # infrastructure.  Keep it behind an explicit package check so a
        # minimal deployment does not advertise unavailable backbones.
        profile['text_encoder_available'] = bool(
            profile['minilm_available'] and env.get('packages', {}).get('transformers'))
        profile['feature_search_enabled'] = config.enable_feature_search
        cuda_enabled = profile["torch_available"] and env["torch"].get("cuda_available") and config.device != "cpu"
        if config.compute_profile == "gpu_large" and not cuda_enabled:
            event(output, 'hardware_fallback', reason='CUDA unavailable; use CPU search')
            config.compute_profile = 'auto'
            config.device = 'auto'
        profile["compute_profile"] = ("gpu_large" if cuda_enabled else "standard") if config.compute_profile == "auto" else config.compute_profile
        # The verified tabular/text frontier used three training seeds per
        # candidate. A single seed makes validation selection noticeably
        # noisier on GPU; restore that generic stability without touching test.
        if (config.stability_repeats == 1 and cuda_enabled
                and profile['data_type'] in {'tabular', 'text'}):
            config.stability_repeats = 3
            event(output, 'gpu_stability_enabled', stability_repeats=3)
        if resumed and config.to_dict() != resumed['manifest']['config']:
            raise ValueError('Effective runtime configuration changed after verified data analysis')
        write_json(output / "profile.json", profile)
        manifest = {"config": config.to_dict(), "requested_config": requested_config, "source_sha256": source_hash,
                    "data_sha256": hashes, "split": split,
                    "test_data_read_during_training": False,
                    "test_access_policy": "Test input is blocked throughout search and refitting. Only final prediction/evaluation follows selection.",
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
            profile['remaining_search_seconds'] = remaining
            profile['trial_timeout_seconds'] = config.trial_timeout_seconds
            candidate, reason = planner.choose(profile, history, config.metric, search_deadline)
            if candidate is None:
                event(output, "search_stopped", reason=reason)
                break
            remaining = search_deadline - time.monotonic()
            if remaining <= 0:
                event(output, "search_stopped", reason="Planning used the remaining search allocation")
                break
            folder = output / "experiments" / f"trial_{trial_number:03d}"
            trial_metadata = {"schema_version": 1, "config": config.to_dict(), "profile": profile,
                              "candidate": candidate, "source_sha256": source_hash,
                              "environment": env, "data_sha256": hashes, "submission": submission}
            event(output, "trial_started", trial=trial_number, candidate=candidate, reason=reason)
            result = run_worker(dict(base_request, candidate=candidate, mode="trial", metadata=trial_metadata), folder,
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
            if config.progressive_output and record['status'] == 'success':
                event(output, 'partial_checkpoint_saved', trial=trial_number,
                      model_path=record.get('checkpoint_path'), test_input_read=False)
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
                        "selected_training_resources": record.get("training_resources"),
                        "selected_feature_resources": record.get('feature_resources', {})}
            event(output, "final_fit_started", rank=rank, trial=record["trial"],
                  candidate=record["candidate"], validation_score=record["score"])
            result = run_worker(dict(base_request, candidate=record["candidate"], mode="final",
                                     metadata=metadata, model_path=str(output / "model.joblib")),
                                output / "experiments" / f"final_{rank:03d}",
                                min(config.trial_timeout_seconds, remaining * .6)
                                if rank < len(successful) else remaining, config.threads)
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
        fallback_used = selected is None
        if fallback_used:
            from superagent.runtime.checkpoint import restore_validated_solution
            for record in successful:
                try:
                    restored = restore_validated_solution(record, source_hash, usage, output / 'model.joblib')
                except (ValueError, OSError) as exc:
                    event(output, 'checkpoint_rejected', trial=record['trial'], reason=str(exc))
                    continue
                if restored:
                    selected = record
                    event(output, 'validated_checkpoint_selected', trial=record['trial'],
                          reason='Final refits failed or allocation expired', test_input_read=False)
                    break
        if selected is None:
            raise RuntimeError("No usable refit or verified validation checkpoint remains")
        event(output, 'selection_completed', trial=selected['trial'], candidate=selected['candidate'])
        test_access[0] = False
        summary.update(status="completed", selected_trial=selected["trial"],
                       partial=fallback_used, final_refit_succeeded=not fallback_used,
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
        summary['formal_runtime_checks_passed'] = (
            summary['status'] == 'completed' and config.evaluation_stage in {'released', 'unseen'}
            and not summary['accounting_provisional']
            and summary.get('model_usage', {}).get('within_budget', False))
        # Runtime checks cannot certify model eligibility, authorship or the
        # historical use of released tests. Never label a run fully compliant.
        summary['eligible_for_formal_submission'] = False
        summary['submission_review_required'] = True
        summary.update(trials=len(history), elapsed_seconds=time.monotonic() - started)
        write_json(output / "summary.json", summary)
        event(output, "run_finished", **summary)
        # Keep split indices/hashes in manifest; avoid retaining a second full dataset.
        cache = output / "_split.joblib"
        if cache.exists():
            cache.unlink()
        if accounting_error is not None:
            raise accounting_error
