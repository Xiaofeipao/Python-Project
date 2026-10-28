"""Frozen submission bundles and the official 24-hour unseen-task window."""

import hashlib
import json
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .io import write_json


ROOT = Path(__file__).resolve().parents[1]
MODEL_CUTOFF = "2026-10-21"
TASK_FIELDS = {
    "train_csv", "validation_csv", "test_csv", "target_column", "data_type",
    "id_column", "feature_columns", "text_columns", "image_column", "image_root",
    "metric", "positive_label", "evaluation_id", "evaluation_stage",
    "frozen_manifest", "unseen_released_at",
}


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def program_files(root=ROOT):
    root = Path(root)
    paths = [root / "main.py", root / "requirements.txt", root / "requirements-gpu.txt"]
    return paths + sorted((root / "superagent").rglob("*.py"))


def runtime_fingerprint(root=ROOT):
    digest = hashlib.sha256()
    for path in program_files(root):
        name = path.relative_to(root).as_posix()
        digest.update(name.encode("utf-8") + b"\0" + path.read_bytes())
    return digest.hexdigest()


def safe_member(root, relative):
    if Path(relative).is_absolute():
        raise ValueError("Bundle/evidence members must use relative paths")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Frozen bundle member escapes bundle directory")
    return path


def verify_bundle(manifest_path, check_runtime=True):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_cutoff") != MODEL_CUTOFF:
        raise ValueError("Unsupported frozen submission manifest")
    if "frozen_policy.json" not in manifest["files"]:
        raise ValueError("Frozen manifest omits its algorithm policy")
    for name, expected in manifest["files"].items():
        if digest_file(safe_member(path.parent, name)) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    if json.loads((path.parent / "frozen_policy.json").read_text(encoding="utf-8")) != manifest["policy"]:
        raise ValueError("Embedded manifest policy differs from its frozen policy file")
    if runtime_fingerprint(path.parent) != manifest["runtime_sha256"]:
        raise ValueError("Frozen bundle runtime changed or has extra Python modules")
    if check_runtime and runtime_fingerprint() != manifest["runtime_sha256"]:
        raise ValueError("Current program differs from the frozen submission; use the frozen entry point")
    return manifest


def frozen_policy(raw, config_path):
    if raw.get("evaluation_stage", "development") == "development":
        return raw
    if not raw.get("frozen_manifest"):
        raise ValueError("Released/unseen runs require a frozen_manifest")
    path = (Path(config_path).parent / raw["frozen_manifest"]).resolve()
    manifest = verify_bundle(path)
    policy = manifest["policy"]
    for name, value in raw.items():
        if name not in TASK_FIELDS and name in policy and value != policy[name]:
            raise ValueError(f"Task config overrides frozen algorithm policy: {name}")
    merged = {**policy, **raw, "frozen_manifest": str(path)}
    if merged.get("llm_registry"):
        merged["llm_registry"] = str(safe_member(path.parent, policy["llm_registry"]))
    return merged


def freeze_submission(policy_path, output):
    # This is an explicit CLI action. Importing the module creates no bundle.
    from .config import TaskConfig

    policy_path = Path(policy_path).resolve()
    raw = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    allowed = {field.name for field in fields(TaskConfig)} - TASK_FIELDS
    if set(raw) - allowed:
        raise ValueError(f"Freeze policy must contain algorithm settings only: {sorted(set(raw) - allowed)}")
    config = TaskConfig(train_csv="unused", target_column="unused", **raw)
    config.validate()
    policy = {key: value for key, value in config.to_dict().items() if key in allowed}
    destination = Path(output).resolve()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError("Frozen output cannot replace or contain the current source checkout")
    if any(destination.is_relative_to(ROOT / name) for name in ("superagent", "configs", "docs", "tests", "scripts")):
        raise ValueError("Use a separate releases directory, outside source/resource folders")
    catalog = None
    if config.llm_mode != "disabled":
        from .usage import ModelCatalog

        if not config.llm_registry:
            raise ValueError("LLM-enabled policy needs a verified model registry")
        catalog = ModelCatalog((policy_path.parent / config.llm_registry).resolve())
        catalog.model(config.llm_model_id)
    destination.mkdir(parents=True, exist_ok=False)
    for source in program_files():
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for folder in ("configs", "docs", "tests", "scripts"):
        for source in sorted((ROOT / folder).rglob("*")):
            if source.is_file() and source.suffix in {".py", ".json", ".md", ".txt"}:
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    for name in ("README.md", "AI_USAGE.md", "AIAA3102 - Final Project - Full Assignment Requirements.md"):
        source = ROOT / name
        if source.exists():
            (destination / name).write_bytes(source.read_bytes())
    if catalog is not None:
        catalog.snapshot(destination / "model_policy")
        policy["llm_registry"] = "model_policy/registry.json"
    else:
        policy["llm_registry"] = None
    write_json(destination / "frozen_policy.json", policy)
    manifest = {
        "schema_version": 1, "model_cutoff": MODEL_CUTOFF,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_sha256": runtime_fingerprint(destination), "policy": policy,
        "files": {path.relative_to(destination).as_posix(): digest_file(path)
                  for path in sorted(destination.rglob("*")) if path.is_file()},
    }
    write_json(destination / "frozen_manifest.json", manifest)
    return {"bundle": str(destination), "runtime_sha256": manifest["runtime_sha256"],
            "status": "frozen", "manifest": str(destination / "frozen_manifest.json")}


def unseen_deadline(released_at):
    try:
        released = datetime.fromisoformat(released_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError("unseen_released_at must be an ISO timestamp with timezone") from exc
    if released.tzinfo is None:
        raise ValueError("Unseen release timestamp requires a timezone")
    now = datetime.now(timezone.utc)
    if released > now:
        raise ValueError("Unseen release time is in the future")
    deadline = released + timedelta(hours=24)
    if now >= deadline:
        raise ValueError("The 24-hour Unseen Task window has expired")
    return deadline.astimezone(timezone.utc)


def stage_record(config):
    record = {"evaluation_stage": config.evaluation_stage, "evaluation_id": config.evaluation_id}
    if config.evaluation_stage != "development":
        manifest = verify_bundle(config.frozen_manifest)
        policy = manifest["policy"]
        for name, value in policy.items():
            actual = getattr(config, name)
            if name == "llm_registry" and value is not None:
                expected = safe_member(Path(config.frozen_manifest).parent, value)
                if Path(actual).resolve() != expected:
                    raise ValueError("LLM registry differs from the frozen policy")
            elif actual != value:
                raise ValueError(f"Runtime policy differs from frozen submission: {name}")
        record.update(frozen_runtime_sha256=manifest["runtime_sha256"],
                      frozen_manifest_sha256=digest_file(config.frozen_manifest))
    if config.evaluation_stage == "unseen":
        record["unseen_deadline_utc"] = unseen_deadline(config.unseen_released_at).isoformat()
    return record


def check_solution_stage(metadata):
    record = metadata.get("submission", {})
    expected = record.get("frozen_runtime_sha256")
    if expected and runtime_fingerprint() != expected:
        raise ValueError("Prediction code differs from the frozen submission version")
    if record.get("unseen_deadline_utc"):
        deadline = datetime.fromisoformat(record["unseen_deadline_utc"])
        if datetime.now(timezone.utc) >= deadline:
            raise ValueError("Unseen Task prediction/evaluation deadline has expired")
