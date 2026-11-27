"""Optional Qwen/DeepSeek planning over a constrained, executable candidate menu."""

import hashlib
import json
import os
import socket
import time
import uuid
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import URLError, HTTPError

from superagent.common.json_io import append_json, event, write_json
from superagent.planning.search import candidate_key, next_candidate
from superagent.planning.proposals import RULES, apply_proposal
from superagent.planning.memory import family_stats, state, trial_key
from superagent.evaluation.metrics import utility
from superagent.governance.frozen_bundle import ROOT
from superagent.governance.usage_ledger import BudgetExhausted, ModelCatalog, UsageLedger, UsageUncertain, amount, pricing_rows, worst_case_cost


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


def parse_plan_response(payload):
    """Parse a complete provider reply without repairing or inventing a plan."""
    choices = payload.get('choices')
    if not isinstance(choices, list) or not choices:
        raise ValueError('Provider returned no choices')
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError('Provider returned an invalid choice')
    if first.get('finish_reason') == 'length':
        raise ValueError('LLM output truncated: finish_reason=length')
    message = first.get('message')
    if not isinstance(message, dict):
        raise ValueError('Provider returned no message object')
    content = message.get('content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('LLM returned empty or non-text content')
    try:
        selection = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f'LLM returned invalid JSON: {exc.msg}') from exc
    if not isinstance(selection, dict):
        raise ValueError('LLM response must be a JSON object')
    return selection


def candidate_menu(profile, history, metric, limit=6):
    planning = list(history)
    menu = []
    stats = family_stats(history, metric)
    abandoned = {f for f, v in stats.items() if v['retired']}
    for family in {h['candidate']['family'] for h in history}:
        recent = [h for h in history if h['candidate']['family'] == family][-2:]
        if len(recent) == 2 and all(h['status'] in {'failed', 'timeout'} for h in recent):
            abandoned.add(family)
    for _ in range(limit * 8):
        candidate, reason = next_candidate(profile, planning, metric)
        if candidate is None:
            break
        planning.append({"candidate": candidate, "status": "not_executed"})
        if candidate['family'] not in abandoned:
            menu.append({"candidate": candidate, "reason": reason})
        if len(menu) >= limit:
            break
    if len(history) >= 6:
        if len(history) % 3:
            menu.sort(key=lambda item: stats.get(item['candidate']['family'], {}).get('best_utility') or -1e9, reverse=True)
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
        'split_risk':profile.get('split_risk'),
        "metric": metric,
        "previous_experiments": [{"candidate": record["candidate"], "status": record["status"],
                                  "elapsed_seconds": record.get('elapsed_seconds'),
                                  "failure_category": ('allocation' if 'out of memory' in record.get('error', '').lower() else record['status']) if record['status'] != 'success' else None,
                                  "validation_score": record.get("score"),
                                  "validation_std": record.get('validation_std'),
                                  'nested_diagnostic':{k:record.get('nested_validation',{}).get(k) for k in ('status','mean','std')}} for record in history[-5:]],
        'family_statistics': family_stats(history, metric),
        'best_previous': max(({'candidate': h['candidate'], 'score': h['score'], 'std': h.get('validation_std')}
                              for h in history if h['status'] == 'success'),
                             key=lambda h: utility(h['score'], metric), default=None),
        "menu": [{"choice": index, **item} for index, item in enumerate(menu)],
    }


class AutonomousPlanner:
    def __init__(self, config, output, evaluation_id, ledger_root=None):
        self.config = config
        self.output = Path(output)
        self.run_id = uuid.uuid4().hex
        shared_root = ledger_root or os.environ.get("SUPERAGENT_USAGE_ROOT")
        if config.evaluation_stage != "development" and (not shared_root or not Path(shared_root).is_absolute()):
            raise ValueError("Formal runs require one persistent absolute SUPERAGENT_USAGE_ROOT shared across agents and frozen copies")
        root = Path(shared_root or str(ROOT / ".model_usage"))
        scope = 'development' if config.evaluation_stage == 'development' else 'final_evaluation'
        key = hashlib.sha256((scope + ':' + evaluation_id).encode("utf-8")).hexdigest()
        self.ledger = UsageLedger(root / key, evaluation_id, config.llm_budget_rmb, config.evaluation_stage)
        self.catalog = None
        self.model = None
        initial = self.export_usage()
        if (not initial["accounting_complete"] and not config.allow_provisional_accounting) or not initial["within_budget"]:
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
                       pricing_rule="2026-10-21 official off-peak input/output CNY prices; all input at cache-miss rate; no cache discounts or free credits")
        write_json(self.output / "model_usage.json", summary)
        write_json(self.output / "task_model_calls.json", self.ledger.audit_records())
        # A task-wide snapshot includes calls from earlier attempts, not only this run.
        records = self.ledger.audit_records()['calls']
        temporary = self.output / 'usage.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        os.replace(temporary, self.output / 'usage.jsonl')
        return summary

    def choose(self, profile, history, metric, deadline):
        if not history:
            return next_candidate(profile, history, metric)
        menu = candidate_menu(profile, history, metric)
        fallback, reason = (menu[0]['candidate'], menu[0]['reason']) if menu else (None, 'Eligible search space exhausted')
        if getattr(self,'_resume_skip_api_once',False):
            self._resume_skip_api_once=False
            event(self.output,'llm_skipped',reason='First restored decision uses local planning; never replay an interrupted API request')
            return fallback,reason
        # Establish the generic anchor before spending API budget on exploration.
        # This is modality/shape-level only and contains no task or dataset identity.
        if len(history) == 1:
            if profile.get('data_type') == 'text' and len(profile.get('text_columns') or []) == 2:
                anchor = {'family': 'pairwise_overlap', 'C': 1., 'fit_intercept': True,
                          'vectorizer_options': {'ngram_range': [3, 6]}}
                event(self.output, 'deterministic_anchor', candidate=anchor,
                      reason='Signed pairwise overlap anchor for two-text-column tasks')
                return anchor, 'Signed pairwise overlap anchor for two-text-column tasks'
            if fallback is not None:
                event(self.output, 'deterministic_anchor', candidate=fallback, reason=reason)
                return fallback, reason
        memory = state(history, metric)
        write_json(Path(self.output) / 'planner_checkpoint.json', {**memory, 'recent_history': history[-5:],
                   'last_rejection': getattr(self, '_last_rejection', None),
                   'restored_reflections':getattr(self,'_restored_reflections',[]),
                   'repair_used': getattr(self, '_repair_used', False),
                   'consecutive_invalid_plans': getattr(self, '_invalid_plans', 0)})
        repair = getattr(self, '_invalid_plans', 0) >= 2
        if repair and getattr(self, '_repair_used', False):
            event(self.output, 'llm_skipped', reason='Controlled repair exhausted; use local planning')
            return fallback, reason
        if self.model is None or not history or fallback is None:
            return fallback, reason
        reserve = worst_case_cost(self.model, self.config.llm_max_output_tokens)
        account = self.ledger.summary()
        if (self.config.evaluation_stage != 'development' and
            amount(account['course_budget_cost_rmb']) + amount(account['unreconciled_reserved_rmb']) + reserve > amount(self.config.max_cost_rmb)):
            event(self.output, 'llm_skipped', reason='Worst-case total with prior unknown reservations exceeds cost ceiling; use local planner')
            return fallback, reason
        menu = candidate_menu(profile, history, metric)
        if not menu:
            return None, 'Remaining candidate families were abandoned after repeated failures'
        context = planning_context(profile, history, metric, menu)
        context['last_rejected_proposal'] = getattr(self, '_last_rejection', None)
        context['restored_reflections'] = getattr(self, '_restored_reflections', [])
        context['experiment_constraint'] = 'Change only features OR hyperparameters of the selected menu seed. Model switches use another seed without overrides. Candidate menu is the complete executable pool for this decision. Compare seed means and standard deviations; seed variation is not split uncertainty.'
        context['format_examples'] = [
            {'choice': 0, 'proposal': {}, 'diagnosis': 'Prior family failed twice', 'next_action': 'Switch to another listed family'},
            {'choice': 0, 'proposal': {'C': .1}, 'hypothesis': 'Stronger regularization reduces variance', 'success_criterion': 'Improve mean validation score without increasing seed standard deviation', 'diagnosis': 'Recent mean scores plateaued', 'next_action': 'Change only C if this seed permits C'}]
        context['parameter_proposal_rules'] = {item['candidate']['family']:RULES[item['candidate']['family']] for item in menu if item['candidate']['family'] in RULES}
        if repair:
            context = {'menu': context['menu'], 'last_rejected_proposal': self._last_rejection,
                       'instruction': 'Choose a listed seed with empty proposal. Return choice, proposal={}, diagnosis and next_action only.'}
        body = {
            "model": self.model["model_id"], "stream": False,
            "max_tokens": self.config.llm_max_output_tokens,
            "messages": [
                {"role": "system", "content": "Return concise JSON only: choice (menu index), proposal (object), diagnosis, next_action, hypothesis, success_criterion. Use validation means/std and rejection feedback. Change ONLY feature settings OR hyperparameters of a menu seed; switching family uses another seed with empty proposal. Stay inside parameter_proposal_rules. No task identity, test data, paths, code, external tools or metric changes. Follow the supplied JSON examples; never invent results."},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)},
            ],
        }
        if self.config.llm_model_id == "DeepSeek-V4-Pro":
            body["reasoning_effort"] = "low"
        serialized = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        while len(serialized) + self.config.llm_max_output_tokens > self.model['context_window_tokens'] and len(menu) > 1:
            menu.pop()
            context['menu'] = [{'choice': i, **item} for i, item in enumerate(menu)]
            context['parameter_proposal_rules'] = {item['candidate']['family']:RULES[item['candidate']['family']] for item in menu if item['candidate']['family'] in RULES}
            body['messages'][1]['content'] = json.dumps(context, ensure_ascii=False, separators=(',', ':'))
            serialized = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
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
                                       self.config.llm_agent_id, self.run_id, request_path,
                                       allow_reserved_unknown=self.config.allow_provisional_accounting)
        except BudgetExhausted:
            event(self.output, "llm_skipped", reason="RMB budget cannot cover another worst-case reservation; use deterministic planner")
            self.export_usage()
            return fallback, reason
        recorded = False
        if repair:
            self._repair_used = True
            event(self.output, 'llm_repair_started', call_id=call['call_id'], reason=self._last_rejection)
        try:
            write_json(request_path, {"call_id": call["call_id"], "body": body})
            with build_opener(NoRedirects()).open(request, timeout=min(self.config.llm_timeout_seconds, remaining)) as response:
                payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise UsageUncertain("Model response exceeds the logging size bound")
            response_path = folder / f"{call['call_id']}.response.json"
            response_path.write_bytes(payload)
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise UsageUncertain('Provider response must be a JSON object')
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
                selection = parse_plan_response(parsed)
                if any(not isinstance(selection.get(k), str) or not selection[k].strip()
                       for k in ('diagnosis', 'next_action')):
                    raise ValueError('Planning requires structured diagnosis and next_action')
                choice = selection["choice"]
                if type(choice) is not int or not 0 <= choice < len(menu):
                    raise ValueError("Model choice is not a menu index")
                proposed = menu[choice]["candidate"]
                proposed = apply_proposal(proposed, selection.get('proposal', {}), profile, memory['trial_memory'])
                if selection.get('proposal') and any(not isinstance(selection.get(k), str) or not selection[k].strip()
                                                     for k in ('hypothesis', 'success_criterion')):
                    raise ValueError('New parameter proposals require a testable hypothesis and success criterion')
                if candidate_key(proposed) in {candidate_key(item["candidate"]) for item in history}:
                    raise ValueError("Model selected an already executed candidate")
                explanation = str(selection.get("reason") or selection.get("diagnosis")
                                  or selection.get("next_action") or "")[:2000]
            except (ValueError, TypeError, KeyError, IndexError) as rejected:
                self._last_rejection = str(rejected)
                self._invalid_plans = getattr(self, '_invalid_plans', 0) + 1
                event(self.output, "llm_invalid_plan", call_id=call["call_id"], fallback=fallback, reason=str(rejected))
                return fallback, reason
            self._invalid_plans = 0
            self._last_rejection = None
            event(self.output, "llm_plan_selected", call_id=call["call_id"], candidate=proposed, explanation=explanation,
                  diagnosis=selection.get('diagnosis', explanation), next_action=selection.get('next_action', explanation),
                  abandoned_strategy=selection.get('abandoned_strategy', ''),
                  proposal=selection.get('proposal', {}), hypothesis=selection.get('hypothesis', ''),
                  success_criterion=selection.get('success_criterion', ''),
                  based_on_trials=len(history), observed_metric=metric)
            return proposed, f"LLM menu selection ({call['call_id']}): {explanation}"
        except BaseException as exc:
            if not recorded:
                dns_not_sent = isinstance(exc, URLError) and isinstance(exc.reason, socket.gaierror)
                error_usage = None
                if isinstance(exc, HTTPError):
                    try:
                        error_payload = exc.read(8 * 1024 * 1024 + 1)
                        (folder / f"{call['call_id']}.response.json").write_bytes(error_payload)
                        error_body = json.loads(error_payload)
                        candidate_usage = error_body.get('usage') if isinstance(error_body, dict) else None
                        if isinstance(candidate_usage, dict) and all(k in candidate_usage for k in ('prompt_tokens', 'completion_tokens')):
                            error_usage = candidate_usage
                    except (ValueError, OSError, HTTPException):
                        pass
                record = self.ledger.finish(call["call_id"], self.model,
                                            usage=error_usage,
                                            error='DNSResolutionError' if dns_not_sent else type(exc).__name__,
                                            dns_not_sent=dns_not_sent,
                                            failed_without_usage=isinstance(exc, (TimeoutError, URLError, ConnectionError)),
                                            http_status=getattr(exc, 'code', None),
                                            provider_request_id=getattr(exc, 'headers', {}).get('x-request-id') if getattr(exc, 'headers', None) else None)
                append_json(self.output / "model_calls.jsonl", record)
                if (record.get('budget_excluded_reason') == 'failed_request_without_usage'
                        or isinstance(exc, (TimeoutError, URLError, ConnectionError)) and record['status'] == 'recorded'):
                    event(self.output, 'llm_unavailable', call_id=call['call_id'], error=type(exc).__name__,
                          http_status=getattr(exc, 'code', None), action='Use local fallback; known usage charged, qualifying unknown usage excluded')
                    return fallback, reason
            if self.config.allow_provisional_accounting and isinstance(exc, (TimeoutError, URLError)):
                event(self.output, 'llm_unavailable', error=type(exc).__name__, action='No retry; use deterministic fallback; retain reservation')
                return fallback, reason
            # Do not retry ambiguous calls or silently continue a reportable run.
            raise
        finally:
            self.export_usage()
