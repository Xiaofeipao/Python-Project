"""Reconciliation must price unknown usage at the reservation bound, never at zero."""

import tempfile
import unittest
from pathlib import Path
from decimal import Decimal

from superagent.governance.usage_ledger import UsageLedger, UsageUncertain


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = {"model_id": "qwen-unit-test-20261020", "family": "Qwen",
                      "release_date": "2026-10-20", "standard_pricing": True,
                      "context_window_tokens": 4096, "input_rmb_per_million": "1",
                      "output_rmb_per_million": "2", "response_model_ids": ["qwen-unit-test-20261020"]}
        self.evidence = self.root / "request.json"
        self.evidence.write_text('{"fixture": true}', encoding="utf-8")

    def _timed_out_call(self, ledger):
        call = ledger.reserve(self.model, 128, "agent", "run", str(self.evidence))
        ledger.finish(call["call_id"], self.model, error="TimeoutError")
        return call

    def test_timeout_is_charged_at_the_reservation_bound(self):
        ledger = UsageLedger(self.root / "ledger", "task", "20")
        call = self._timed_out_call(ledger)
        before = ledger.summary()
        self.assertFalse(before["accounting_complete"])
        self.assertGreater(Decimal(before["unreconciled_reserved_rmb"]), 0)

        recorded = ledger.reconcile_unknown(call["call_id"], "Provider never returned usage after a read timeout",
                                            str(self.evidence))
        after = ledger.summary()
        self.assertEqual(recorded["status"], "reconciled_conservative")
        self.assertEqual(Decimal(after["unreconciled_reserved_rmb"]), Decimal("0"))
        self.assertEqual(after["conservative_reconciled_cost_rmb"], recorded["reserved_rmb"])
        self.assertEqual(after["actual_recorded_cost_rmb"], "0")
        self.assertEqual(after["reconciled_call_count"], 1)
        self.assertFalse(after["accounting_complete"])
        self.assertIsNone(after["actual_total_cost_rmb"])
        self.assertNotIn("input_tokens", recorded)
        self.assertEqual(recorded["reconciliation"]["actual_usage"], "unrecoverable")

    def test_reconciliation_requires_evidence_and_is_single_use(self):
        ledger = UsageLedger(self.root / "guarded", "task", "20")
        call = self._timed_out_call(ledger)
        with self.assertRaises(ValueError):
            ledger.reconcile_unknown(call["call_id"], "  ", str(self.evidence))
        with self.assertRaises(ValueError):
            ledger.reconcile_unknown(call["call_id"], "timeout", " ")
        with self.assertRaises(ValueError):
            ledger.reconcile_unknown("not-a-call", "timeout", str(self.evidence))
        ledger.reconcile_unknown(call["call_id"], "timeout", str(self.evidence))
        with self.assertRaises(ValueError):
            ledger.reconcile_unknown(call["call_id"], "timeout again", str(self.evidence))

    def test_reconciled_task_still_refuses_new_requests(self):
        ledger = UsageLedger(self.root / "blocked", "task", "20")
        call = self._timed_out_call(ledger)
        ledger.reconcile_unknown(call["call_id"], "timeout", str(self.evidence))
        with self.assertRaises(UsageUncertain):
            ledger.reserve(self.model, 128, "agent", "run2", str(self.evidence))

    def test_never_finalized_call_can_be_reconciled(self):
        ledger = UsageLedger(self.root / "pending", "task", "20")
        call = ledger.reserve(self.model, 128, "agent", "run", str(self.evidence))
        recorded = ledger.reconcile_unknown(call["call_id"], "Process died before finalization",
                                            str(self.evidence))
        self.assertEqual(recorded["status"], "reconciled_conservative")


if __name__ == "__main__":
    unittest.main()
