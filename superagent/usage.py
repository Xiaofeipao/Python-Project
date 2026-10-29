"""Official-price accounting shared by every model-calling agent for a task.

No online calls, model assumptions or price lookup occur on import. Eligibility
depends on configured model dates and official rates, never provider promotions.
"""

import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

from .io import append_json, write_json
from .submission import MODEL_CUTOFF, digest_file, safe_member


class BudgetExhausted(RuntimeError):
    pass


class UsageUncertain(RuntimeError):
    pass


ROLLING_ALIASES = {
    "deepseek-chat", "deepseek-reasoner", "qwen-plus", "qwen-max",
    "qwen-turbo", "qwen-long", "qwen-flash", "qwen3-max",
}


def amount(value):
    if isinstance(value, bool):
        raise ValueError("Money must be a finite nonnegative decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Invalid RMB amount") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Money must be a finite nonnegative decimal")
    return result


def tokens(value):
    if type(value) is not int or value < 0:
        raise ValueError("Token usage must contain actual nonnegative integers")
    return value


def pricing_rows(model):
    return model.get("pricing_tiers") or [{
        "input_tokens_up_to": None,
        "input_rmb_per_million": model["input_rmb_per_million"],
        "output_rmb_per_million": model["output_rmb_per_million"],
    }]


def official_host(family, parsed):
    host = (parsed.hostname or "").lower()
    organization = parsed.path.strip("/").split("/", 1)[0].lower()
    if family == "Qwen":
        return (host in {"qwen.ai", "qwenlm.github.io"} or host.endswith(".qwen.ai")
                or host == "aliyun.com" or host.endswith(".aliyun.com")
                or host == "alibabacloud.com" or host.endswith(".alibabacloud.com")
                or host in {"github.com", "huggingface.co"} and organization in {"qwen", "qwenlm"})
    return (host == "deepseek.com" or host.endswith(".deepseek.com")
            or host in {"github.com", "huggingface.co"} and organization == "deepseek-ai")


def applicable_rates(model, input_tokens):
    for row in pricing_rows(model):
        if row["input_tokens_up_to"] is None or input_tokens <= row["input_tokens_up_to"]:
            return row
    raise ValueError("Official pricing table does not cover the reported input length")


class ModelCatalog:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if self.raw.get("schema_version") != 1 or self.raw.get("pricing_as_of") != MODEL_CUTOFF or self.raw.get("currency") != "CNY":
            raise ValueError("Registry must use official CNY standard prices as of 2026-10-21")
        if not isinstance(self.raw.get("models"), list):
            raise ValueError("Model registry needs a models list")
        self.models = {}
        for model in self.raw["models"]:
            self._validate(model)
            key = model["model_id"]
            if key in self.models:
                raise ValueError("Duplicate model_id in registry")
            self.models[key] = model

    def _validate(self, model):
        family = model.get("family")
        key = model.get("model_id", "")
        if family not in {"Qwen", "DeepSeek"} or not isinstance(key, str) or not key.lower().startswith(family.lower()):
            raise ValueError("Only explicitly identified Qwen or DeepSeek models are allowed")
        if re.search(r"(^|[-_/])(latest|auto)([-_/]|$)", key.lower()) or key.lower() in ROLLING_ALIASES:
            raise ValueError("Rolling model aliases cannot prove release-cutoff eligibility")
        if date.fromisoformat(model["release_date"]) > date.fromisoformat(MODEL_CUTOFF):
            raise ValueError("Model was released after the course cutoff")
        if model.get("standard_pricing") is not True:
            raise ValueError("Registry entry requires official standard pricing")
        if tokens(model["context_window_tokens"]) == 0:
            raise ValueError("Official context window is required for conservative reservation")
        if model.get("pricing_basis", "input_tokens") != "input_tokens":
            raise ValueError("Supported price tiers must depend on total input tokens")
        rows = pricing_rows(model)
        previous = 0
        for index, row in enumerate(rows):
            for name in ("input_rmb_per_million", "output_rmb_per_million"):
                if amount(row[name]) <= 0:
                    raise ValueError("Use official positive standard prices, not free-credit rates")
            upper = row["input_tokens_up_to"]
            if upper is None:
                if index != len(rows) - 1:
                    raise ValueError("Only the final price tier can be unbounded")
            else:
                if tokens(upper) <= previous:
                    raise ValueError("Input-token price tiers must be strictly increasing")
                previous = upper
        applicable_rates(model, model["context_window_tokens"])
        model.setdefault("response_model_ids", [key])
        if not isinstance(model.get("response_model_ids"), list) or not model["response_model_ids"]:
            raise ValueError("Declare pinned response model IDs returned by the deployment")
        for value in model["response_model_ids"]:
            if not isinstance(value, str) or not value or "latest" in value.lower() or value.lower() in ROLLING_ALIASES:
                raise ValueError("Response model IDs must be pinned")
        evidence = model.get("evidence", [])
        for item in evidence:
            if item["snapshot"] == "registry.json":
                raise ValueError("Evidence cannot overwrite the registry file")
            parsed = urlparse(item["official_url"])
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Evidence needs an HTTPS official-source URL without credentials")
            if not official_host(family, parsed):
                raise ValueError("Evidence URL is not on a recognized official model-publisher domain")
            expected = item.get("sha256", "")
            if not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError("Evidence snapshot needs its SHA-256")
            if digest_file(safe_member(self.path.parent, item["snapshot"])) != expected:
                raise ValueError("Model/pricing evidence snapshot changed")

    def model(self, key):
        if key not in self.models:
            raise ValueError("Requested model is not in the reviewed eligible-model registry")
        return self.models[key]

    def snapshot(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "registry.json").write_bytes(self.path.read_bytes())
        for model in self.models.values():
            for item in model.get("evidence", []):
                target = safe_member(destination, item["snapshot"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(safe_member(self.path.parent, item["snapshot"]).read_bytes())


def official_cost(model, input_tokens, output_tokens):
    input_tokens, output_tokens = tokens(input_tokens), tokens(output_tokens)
    rates = applicable_rates(model, input_tokens)
    return (amount(rates["input_rmb_per_million"]) * input_tokens
            + amount(rates["output_rmb_per_million"]) * output_tokens) / Decimal(1_000_000)


class UsageLedger:
    def __init__(self, directory, evaluation_id, budget="20"):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "ledger.json"
        self.evaluation_id = evaluation_id
        self.budget = min(amount(budget), Decimal("20"))

    @contextmanager
    def locked(self):
        lock = self.directory / "ledger.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise UsageUncertain("Task usage ledger is locked; do not start an independent budget") from exc
        try:
            os.close(descriptor)
            yield
        finally:
            lock.unlink()

    def _read(self):
        if not self.path.exists():
            return {"schema_version": 1, "evaluation_id": self.evaluation_id, "calls": []}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("evaluation_id") != self.evaluation_id:
            raise ValueError("Usage ledger belongs to another evaluation task")
        return state

    def reserve(self, model, max_output_tokens, agent_id, run_id, request_path):
        if tokens(max_output_tokens) == 0:
            raise ValueError("Output-token limit must be positive")
        with self.locked():
            state = self._read()
            if any(call["status"] != "recorded" for call in state["calls"]):
                raise UsageUncertain("Prior request has unreconciled usage; no further model requests allowed")
            spent = sum((amount(call.get("cost_rmb", "0")) for call in state["calls"]), Decimal(0))
            rates = pricing_rows(model)
            # Cover reasoning output too, even on a provider whose max_tokens
            # interpretation is narrower than total generated output tokens.
            reserve = (max(amount(row["input_rmb_per_million"]) for row in rates)
                       + max(amount(row["output_rmb_per_million"]) for row in rates)) * model["context_window_tokens"] / Decimal(1_000_000)
            if spent + reserve > self.budget:
                raise BudgetExhausted("Next worst-case model request would exceed this task's RMB budget")
            call = {"call_id": uuid.uuid4().hex, "agent_id": agent_id, "run_id": run_id,
                    "evaluation_id": self.evaluation_id, "status": "pending",
                    "model_id": model["model_id"], "family": model["family"],
                    "pricing_as_of": MODEL_CUTOFF,
                    "official_standard_pricing": rates,
                    "reserved_rmb": str(reserve), "request_path": str(request_path),
                    "max_output_tokens": max_output_tokens,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            state["calls"].append(call)
            write_json(self.path, state)
            return call

    def finish(self, call_id, model, usage=None, error=None):
        with self.locked():
            state = self._read()
            call = next(item for item in state["calls"] if item["call_id"] == call_id)
            if call["status"] != "pending":
                raise UsageUncertain("Model request was already finalized")
            if usage is None:
                call.update(status="usage_unknown", error=error,
                            cost_rmb=None, accounting_complete=False)
            else:
                prompt = tokens(usage["prompt_tokens"])
                completion = tokens(usage["completion_tokens"])
                if "total_tokens" in usage and tokens(usage["total_tokens"]) != prompt + completion:
                    raise UsageUncertain("Provider token totals do not reconcile; reasoning/output usage may be missing")
                details = usage.get("completion_tokens_details") or {}
                if "reasoning_tokens" in details and tokens(details["reasoning_tokens"]) > completion:
                    raise UsageUncertain("Completion tokens do not include all reasoning output")
                call.update(status="recorded", input_tokens=prompt, output_tokens=completion,
                            raw_usage=usage, cost_rmb=str(official_cost(model, prompt, completion)),
                            accounting_complete=True)
                call["applied_rates"] = applicable_rates(model, prompt)
                if prompt == 0 or prompt > model["context_window_tokens"] or completion > model["context_window_tokens"]:
                    call.update(status="reported_usage_exceeds_reservation", accounting_complete=False)
            write_json(self.path, state)
            append_json(self.directory / "calls.jsonl", call)
            return call

    def summary(self):
        with self.locked():
            state = self._read()
        known = sum((amount(call["cost_rmb"]) for call in state["calls"] if call.get("cost_rmb") is not None), Decimal(0))
        held = sum((amount(call["reserved_rmb"]) for call in state["calls"]
                    if call["status"] in {"pending", "usage_unknown"}), Decimal(0))
        complete = all(call.get("accounting_complete", False) for call in state["calls"])
        inputs = sum(call.get("input_tokens", 0) for call in state["calls"])
        outputs = sum(call.get("output_tokens", 0) for call in state["calls"])
        return {"evaluation_id": self.evaluation_id, "budget_rmb": str(self.budget),
                "actual_recorded_cost_rmb": str(known), "unreconciled_reserved_rmb": str(held),
                "actual_total_cost_rmb": str(known) if complete else None,
                "call_count": len(state["calls"]),
                "recorded_input_tokens": inputs, "recorded_output_tokens": outputs,
                "input_tokens": inputs if complete else None, "output_tokens": outputs if complete else None,
                "accounting_complete": complete,
                "within_budget": known + held <= self.budget,
                "ledger_path": str(self.path), "requires_all_agents_to_share_ledger": True}

    def audit_records(self):
        with self.locked():
            return self._read()
