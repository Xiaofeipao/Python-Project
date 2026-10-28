"""Strict task configuration. No task-name or dataset-name dispatch."""

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path


METRICS = {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "log_loss"}


@dataclass
class TaskConfig:
    train_csv: str
    target_column: str
    data_type: str = "auto"
    validation_csv: str | None = None
    test_csv: str | None = None
    id_column: str | None = None
    feature_columns: list[str] | None = None
    text_columns: list[str] | None = None
    image_column: str | None = None
    image_root: str = "."
    metric: str = "f1_macro"
    positive_label: str | None = None
    validation_fraction: float = 0.2
    max_trials: int = 16
    time_budget_seconds: float = 1800
    trial_timeout_seconds: float = 300
    final_fit_reserve_fraction: float = 0.3
    threads: int = 1
    seed: int = 42
    enable_torch: bool = True
    device: str = "auto"
    compute_profile: str = "auto"
    neural_batch_size: int = 256
    mixed_precision: bool = True
    gpu_cache_fraction: float = 0.2
    cache_preprocessing: bool = True
    evaluation_stage: str = "development"
    evaluation_id: str | None = None
    frozen_manifest: str | None = None
    unseen_released_at: str | None = None
    llm_mode: str = "disabled"
    llm_registry: str | None = None
    llm_model_id: str | None = None
    llm_base_url: str | None = None
    llm_api_key_env: str = "HKUST_API_KEY"
    llm_agent_id: str = "planner"
    llm_max_output_tokens: int = 512
    llm_timeout_seconds: float = 60
    llm_budget_rmb: str = "20"

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        from .submission import frozen_policy

        raw = frozen_policy(raw, path)
        config = cls(**raw)
        config.validate()
        for name in ("train_csv", "validation_csv", "test_csv", "image_root", "frozen_manifest", "llm_registry"):
            value = getattr(config, name)
            if value is not None:
                setattr(config, name, str((path.parent / value).resolve()))
        paths = [config.train_csv, config.validation_csv, config.test_csv]
        paths = [str(Path(p).resolve()).casefold() for p in paths if p]
        if len(paths) != len(set(paths)):
            raise ValueError("Train, validation and designated test paths must be distinct")
        return config

    def validate(self):
        from .usage import amount

        if self.evaluation_stage not in {"development", "released", "unseen"}:
            raise ValueError("evaluation_stage must be development/released/unseen")
        if self.evaluation_stage != "development" and (not self.frozen_manifest or not self.evaluation_id):
            raise ValueError("Formal evaluation requires frozen_manifest and a shared evaluation_id")
        if self.evaluation_stage == "unseen" and not self.unseen_released_at:
            raise ValueError("Unseen runs require the official release timestamp")
        if self.llm_mode not in {"disabled", "required"}:
            raise ValueError("llm_mode must be disabled or required")
        if not 0 < amount(self.llm_budget_rmb) <= 20:
            raise ValueError("The model budget must be positive and cannot exceed RMB 20 per task")
        self.llm_budget_rmb = str(amount(self.llm_budget_rmb))
        if not isinstance(self.llm_api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.llm_api_key_env):
            raise ValueError("llm_api_key_env must name an environment variable, not contain an API key")
        if not isinstance(self.llm_agent_id, str) or not self.llm_agent_id:
            raise ValueError("llm_agent_id must be a nonempty audit identifier")
        if self.llm_mode == "required" and not all((self.llm_registry, self.llm_model_id, self.llm_base_url)):
            raise ValueError("LLM planning needs llm_registry, llm_model_id and llm_base_url")
        if self.compute_profile not in {"auto", "standard", "gpu_large"}:
            raise ValueError("compute_profile must be auto/standard/gpu_large")
        if type(self.mixed_precision) is not bool or type(self.cache_preprocessing) is not bool:
            raise ValueError("mixed_precision and cache_preprocessing must be boolean")
        if type(self.enable_torch) is not bool or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("enable_torch must be boolean; device must be auto/cpu/cuda")
        if self.data_type not in {"auto", "tabular", "text", "image"}:
            raise ValueError("Unsupported data_type")
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}")
        for name in ("train_csv", "target_column", "image_root"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("validation_csv", "test_csv", "id_column", "image_column", "positive_label",
                     "evaluation_id", "frozen_manifest", "unseen_released_at", "llm_registry", "llm_model_id", "llm_base_url"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or null")
        for name in ("feature_columns", "text_columns"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, list) or not value
                or any(not isinstance(v, str) or not v for v in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(f"{name} must be a nonempty list of unique column names")
        for name in ("max_trials", "threads", "seed", "neural_batch_size", "llm_max_output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid nonnegative/positive integer")
        if self.seed > 4294967295:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        for name in ("time_budget_seconds", "trial_timeout_seconds", "llm_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("validation_fraction", "final_fit_reserve_fraction", "gpu_cache_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if self.id_column == self.target_column:
            raise ValueError("ID and target must be different columns")

    def to_dict(self):
        return asdict(self)
