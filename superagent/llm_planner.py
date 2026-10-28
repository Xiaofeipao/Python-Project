"""Optional Qwen/DeepSeek planning over a constrained, executable candidate menu."""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .io import append_json, event, write_json
from .search import candidate_key, next_candidate
from .submission import ROOT
from .usage import BudgetExhausted, ModelCatalog, UsageLedger, UsageUncertain


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise UsageUncertain("API redirects are not allowed; request usage is uncertain")


def endpoint(base_url):
    parsed = urlparse(base_url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme not in {"https", "http"} or (parsed.scheme == "http" and not local):
        raise ValueError("Use HTTPS for remote APIs, or HTTP only for a local deployment")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API base URL cannot contain credentials, query parameters or fragments")
    return base_url.rstrip("/") + "/chat/completions"


def candidate_menu(profile, history, metric, limit=6):
    planning = list(history)
    menu = []
    for _ in range(limit):
        candidate, reason = next_candidate(profile, planning, metric)
        if candidate is None:
            break
        menu.append({"candidate": candidate, "reason": reason})
        planning.append({"candidate": candidate, "status": "not_executed"})
    return menu


def planning_context(profile, history, metric, menu):
    # No task name, dataset name, file path, feature names, label names or test
    # results are transmitted. Candidate execution remains in the local runner.
    return {
        "data": {"modality": profile["data_type"], "training_rows": profile["rows"],
                 "feature_count": len(profile["features"]),
                 "numeric_feature_count": len(profile["numeric_columns"]),
                 "class_counts": sorted(profile["class_counts"].values()),
                 "missing_fractions": sorted(profile["missing_fractions"].values()),
                 "compute_profile": profile["compute_profile"]},
        "metric": metric,
        "previous_experiments": [{"candidate": record["candidate"], "status": record["status"],
                                  "validation_score": record.get("score")} for record in history],
        "menu": [{"choice": index, **item} for index, item in enumerate(menu)],
    }


class AutonomousPlanner:
    def __init__(self, config, output, evaluation_id):
        self.config = config
        self.output = Path(output)
        self.run_id = uuid.uuid4().hex
        shared_root = os.environ.get("SUPERAGENT_USAGE_ROOT")
        if config.evaluation_stage != "development" and (not shared_root or not Path(shared_root).is_absolute()):
            raise ValueError("Formal runs require one persistent absolute SUPERAGENT_USAGE_ROOT shared across agents and frozen copies")
        root = Path(shared_root or str(ROOT / ".model_usage"))
        key = hashlib.sha256(evaluation_id.encode("utf-8")).hexdigest()
        self.ledger = UsageLedger(root / key, evaluation_id, config.llm_budget_rmb)
        self.catalog = None
        self.model = None
        initial = self.export_usage()
        if not initial["accounting_complete"] or not initial["within_budget"]:
            raise UsageUncertain("Shared task ledger is incomplete or already exceeds its budget")
        if config.llm_mode == "required":
            self.catalog = ModelCatalog(config.llm_registry)
            self.model = self.catalog.model(config.llm_model_id)
            if config.llm_max_output_tokens >= self.model["context_window_tokens"]:
                raise ValueError("Output token limit must be below the pinned model context window")
            endpoint(config.llm_base_url)
            if not os.environ.get(config.llm_api_key_env) and urlparse(config.llm_base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError(f"Set API credentials through environment variable {config.llm_api_key_env}")
            self.catalog.snapshot(self.output / "model_policy")
        self.export_usage()

    def export_usage(self):
        summary = self.ledger.summary()
        summary.update(llm_mode=self.config.llm_mode, run_id=self.run_id,
                       pricing_rule="2026-10-21 official input/output standard CNY prices; no cache discounts or free credits")
        write_json(self.output / "model_usage.json", summary)
        write_json(self.output / "task_model_calls.json", self.ledger.audit_records())
        return summary

    def choose(self, profile, history, metric, deadline):
        fallback, reason = next_candidate(profile, history, metric)
        if self.model is None or not history or fallback is None:
            return fallback, reason
        menu = candidate_menu(profile, history, metric)
        if not menu:
            return fallback, reason
        context = planning_context(profile, history, metric, menu)
        body = {
            "model": self.model["model_id"], "stream": False,
            "max_tokens": self.config.llm_max_output_tokens,
            "messages": [
                {"role": "system", "content": "Plan general classification experiments using only the data statistics and validation feedback. Choose exactly one candidate from the menu. Return JSON only: {\"choice\": integer, \"reason\": string}. Never use task or dataset identity, test data, external tools, or executable code."},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)},
            ],
        }
        serialized = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        # Byte bound is deliberately conservative; also reserve the full
        # official context window rather than guessing a tokenizer's cost.
        if len(serialized) + self.config.llm_max_output_tokens > self.model["context_window_tokens"]:
            event(self.output, "llm_skipped", reason="Conservative context-size bound exceeded")
            return fallback, reason
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return fallback, reason
        headers = {"Content-Type": "application/json"}
        secret = os.environ.get(self.config.llm_api_key_env)
        if secret:
            headers["Authorization"] = "Bearer " + secret
        request = Request(endpoint(self.config.llm_base_url), data=serialized, headers=headers, method="POST")
        folder = self.output / "model_calls"
        folder.mkdir(exist_ok=True)
        request_path = folder / f"{uuid.uuid4().hex}.request.json"
        try:
            call = self.ledger.reserve(self.model, self.config.llm_max_output_tokens,
                                       self.config.llm_agent_id, self.run_id, request_path)
        except BudgetExhausted:
            event(self.output, "llm_skipped", reason="RMB budget cannot cover another worst-case reservation; use deterministic planner")
            self.export_usage()
            return fallback, reason
        recorded = False
        try:
            write_json(request_path, {"call_id": call["call_id"], "body": body})
            with build_opener(NoRedirects()).open(request, timeout=min(self.config.llm_timeout_seconds, remaining)) as response:
                payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise UsageUncertain("Model response exceeds the logging size bound")
            response_path = folder / f"{call['call_id']}.response.json"
            response_path.write_bytes(payload)
            parsed = json.loads(payload)
            if parsed.get("model") not in self.model["response_model_ids"]:
                raise UsageUncertain("Provider returned a model identity outside the pinned registry")
            usage = parsed.get("usage")
            if not isinstance(usage, dict):
                raise UsageUncertain("Provider omitted actual token usage; zero usage cannot be assumed")
            if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                raise UsageUncertain("Provider omitted standard prompt/completion token counts")
            record = self.ledger.finish(call["call_id"], self.model, usage=usage)
            recorded = True
            append_json(self.output / "model_calls.jsonl", {**record, "response_path": str(response_path)})
            if record["status"] != "recorded":
                raise UsageUncertain("Actual token usage exceeded the registered reservation bounds")
            try:
                content = parsed["choices"][0]["message"]["content"]
                selection = json.loads(content)
                choice = selection["choice"]
                if type(choice) is not int or not 0 <= choice < len(menu):
                    raise ValueError("Model choice is not a menu index")
                proposed = menu[choice]["candidate"]
                if candidate_key(proposed) in {candidate_key(item["candidate"]) for item in history}:
                    raise ValueError("Model selected an already executed candidate")
                explanation = str(selection.get("reason", ""))[:2000]
            except (ValueError, TypeError, KeyError, IndexError):
                event(self.output, "llm_invalid_plan", call_id=call["call_id"], fallback=fallback)
                return fallback, reason
            event(self.output, "llm_plan_selected", call_id=call["call_id"], candidate=proposed, explanation=explanation)
            return proposed, f"LLM menu selection ({call['call_id']}): {explanation}"
        except BaseException as exc:
            if not recorded:
                record = self.ledger.finish(call["call_id"], self.model, error=type(exc).__name__)
                append_json(self.output / "model_calls.jsonl", record)
            # Do not retry ambiguous calls or silently continue a reportable run.
            raise
        finally:
            self.export_usage()
