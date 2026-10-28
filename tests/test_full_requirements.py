"""Formal-requirements regressions, prepared on 2026-10-23 and NOT run yet.

All prices/model names below are synthetic unit-test fixtures, never eligible
production-model entries. HTTP is mocked; these tests must not use a real API.
"""

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from superagent.config import TaskConfig
from superagent.llm_planner import AutonomousPlanner, planning_context
from superagent.submission import freeze_submission, unseen_deadline, verify_bundle
from superagent.usage import BudgetExhausted, ModelCatalog, UsageLedger, UsageUncertain, official_cost


class FormalRequirements(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        evidence = self.root / "evidence" / "fixture.txt"
        evidence.parent.mkdir()
        evidence.write_text("SYNTHETIC TEST FIXTURE: NOT A REAL MODEL OR OFFICIAL PRICE", encoding="utf-8")
        checksum = hashlib.sha256(evidence.read_bytes()).hexdigest()
        self.model = {
            "model_id": "qwen-unit-test-20261020", "family": "Qwen", "release_date": "2026-10-20",
            "reviewed": True, "pinned": True, "standard_pricing": True, "context_window_tokens": 4096,
            "input_rmb_per_million": "1", "output_rmb_per_million": "2",
            "response_model_ids": ["qwen-unit-test-20261020"],
            "evidence": [{"kind": kind, "official_url": "https://help.aliyun.com/test-fixture-not-real",
                          "snapshot": "evidence/fixture.txt", "sha256": checksum}
                         for kind in ("release", "pricing", "context")],
        }
        self.registry = self.root / "registry.json"
        self.write_registry()

    def write_registry(self, **changes):
        model = {**self.model, **changes}
        self.registry.write_text(json.dumps({"schema_version": 1, "currency": "CNY",
                                            "pricing_as_of": "2026-10-21", "models": [model]}), encoding="utf-8")

    def test_model_family_release_cutoff_and_evidence(self):
        ModelCatalog(self.registry)
        for changes in ({"family": "Other"}, {"release_date": "2026-10-22"},
                        {"model_id": "qwen-latest"}, {"reviewed": False}):
            self.write_registry(**changes)
            with self.assertRaises(ValueError):
                ModelCatalog(self.registry)
        self.write_registry()
        (self.root / "evidence" / "fixture.txt").write_text("CHANGED", encoding="utf-8")
        with self.assertRaises(ValueError):
            ModelCatalog(self.registry)

    def test_unofficial_evidence_domain_is_rejected(self):
        evidence = [{**item, "official_url": "https://untrusted.example/pricing"} for item in self.model["evidence"]]
        self.write_registry(evidence=evidence)
        with self.assertRaises(ValueError):
            ModelCatalog(self.registry)

    def test_decimal_costs_ignore_cache_discounts_and_share_agents(self):
        ledger = UsageLedger(self.root / "usage", "same-course-task", "20")
        for agent in ("agent-A", "agent-B"):
            call = ledger.reserve(self.model, 128, agent, "run", "request.json")
            record = ledger.finish(call["call_id"], self.model, usage={
                "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 100},
            })
            self.assertEqual(record["cost_rmb"], "0.0002")
        summary = UsageLedger(self.root / "usage", "same-course-task", "20").summary()
        self.assertEqual(summary["actual_recorded_cost_rmb"], "0.0004")
        self.assertEqual(summary["call_count"], 2)
        self.assertEqual(summary["input_tokens"], 200)

    def test_context_price_tiers(self):
        model = {**self.model, "pricing_tiers": [
            {"input_tokens_up_to": 64, "input_rmb_per_million": "1", "output_rmb_per_million": "2"},
            {"input_tokens_up_to": None, "input_rmb_per_million": "3", "output_rmb_per_million": "4"},
        ]}
        self.assertEqual(official_cost(model, 100, 50), Decimal("0.0005"))

    def test_budget_reserved_before_request_and_unknown_usage_blocks_retries(self):
        small = UsageLedger(self.root / "small", "task", "0.001")
        with self.assertRaises(BudgetExhausted):
            small.reserve(self.model, 128, "agent", "run", "request.json")
        ledger = UsageLedger(self.root / "unknown", "task", "20")
        call = ledger.reserve(self.model, 128, "agent", "run", "request.json")
        ledger.finish(call["call_id"], self.model, error="TimeoutError")
        self.assertFalse(ledger.summary()["accounting_complete"])
        self.assertGreater(Decimal(ledger.summary()["unreconciled_reserved_rmb"]), 0)
        with self.assertRaises(UsageUncertain):
            ledger.reserve(self.model, 128, "another-agent", "another-run", "request.json")

    def test_token_totals_must_include_reasoning(self):
        ledger = UsageLedger(self.root / "totals", "task", "20")
        call = ledger.reserve(self.model, 128, "agent", "run", "request.json")
        with self.assertRaises(UsageUncertain):
            ledger.finish(call["call_id"], self.model,
                          usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 250})

    def test_course_budget_cannot_be_raised(self):
        for budget in ("20.01", "0", "NaN"):
            with self.assertRaises(ValueError):
                TaskConfig(train_csv="unused", target_column="label", llm_budget_rmb=budget).validate()

    def test_frozen_policy_rejects_task_specific_search_overrides(self):
        policy = self.root / "policy.json"
        policy.write_text(json.dumps({"max_trials": 2}), encoding="utf-8")
        result = freeze_submission(policy, self.root / "bundle")
        verify_bundle(result["manifest"])
        task = self.root / "task.json"
        raw = {"evaluation_stage": "released", "evaluation_id": "course-task",
               "frozen_manifest": result["manifest"], "train_csv": "unused.csv", "target_column": "label"}
        task.write_text(json.dumps(raw), encoding="utf-8")
        config = TaskConfig.load(task)
        self.assertEqual(config.max_trials, 2)
        task.write_text(json.dumps({**raw, "max_trials": 3}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overrides frozen"):
            TaskConfig.load(task)
        source = self.root / "bundle" / "superagent" / "search.py"
        source.write_bytes(source.read_bytes() + b"\n# changed after submission\n")
        with self.assertRaises(ValueError):
            verify_bundle(result["manifest"])

    def test_unseen_window_uses_release_time_not_process_start(self):
        now = datetime.now(timezone.utc)
        deadline = unseen_deadline((now - timedelta(hours=1)).isoformat())
        self.assertLess(abs((deadline - now).total_seconds() - 23 * 3600), 5)
        for timestamp in ((now - timedelta(hours=25)).isoformat(),
                          (now + timedelta(hours=1)).isoformat(), "2026-10-23T12:00:00"):
            with self.assertRaises(ValueError):
                unseen_deadline(timestamp)

    def profile(self):
        return {"data_type": "tabular", "rows": 20, "features": ["PRIVATE_DATASET_NAME"],
                "numeric_columns": ["PRIVATE_DATASET_NAME"], "class_counts": {"PRIVATE_LABEL": 10, "other": 10},
                "missing_fractions": {"PRIVATE_DATASET_NAME": 0.0}, "compute_profile": "standard"}

    def test_planning_context_excludes_identity_and_test_information(self):
        context = planning_context(self.profile(), [], "accuracy", [])
        serialized = json.dumps(context)
        self.assertNotIn("PRIVATE_DATASET_NAME", serialized)
        self.assertNotIn("PRIVATE_LABEL", serialized)
        self.assertNotIn("test_csv", serialized)

    def test_mocked_model_request_records_actual_tokens(self):
        output = self.root / "run"
        output.mkdir()
        config = TaskConfig(train_csv="unused", target_column="label", llm_mode="required",
                            llm_registry=str(self.registry), llm_model_id=self.model["model_id"],
                            llm_base_url="http://127.0.0.1:1/v1")
        history = [{"candidate": {"family": "dummy"}, "status": "success", "score": 0.5}]
        payload = {"model": self.model["model_id"],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                   "choices": [{"message": {"content": '{"choice": 0, "reason": "validation-driven experiment"}'}}]}
        with patch.dict(os.environ, {"SUPERAGENT_USAGE_ROOT": str(self.root / "shared")}):
            planner = AutonomousPlanner(config, output, "unit-fixture-task")
            with patch("superagent.llm_planner.build_opener") as mocked:
                mocked.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
                import time

                candidate, _ = planner.choose(self.profile(), history, "accuracy", time.monotonic() + 30)
            self.assertEqual(candidate["family"], "logistic")
            self.assertEqual(planner.export_usage()["actual_recorded_cost_rmb"], "0.00012")


if __name__ == "__main__":
    unittest.main()
