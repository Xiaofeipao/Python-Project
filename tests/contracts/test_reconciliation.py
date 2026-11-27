"""Unknown usage stays unknown; historical evidence is never rewritten."""
import json
import tempfile
import unittest
from pathlib import Path
from superagent.governance.usage_ledger import UsageLedger, UsageUncertain

MODEL = {'model_id':'DeepSeek-Fixture', 'family':'DeepSeek', 'context_window_tokens':4096,
         'input_rmb_per_million':'1', 'output_rmb_per_million':'2'}


class ReconciliationTests(unittest.TestCase):
    def test_failed_unknown_is_excluded_without_fabricating_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=UsageLedger(folder,'fixture')
            call=ledger.reserve(MODEL,128,'planner','run','request')
            record=ledger.finish(call['call_id'],MODEL,error='HTTPError',http_status=500)
            self.assertIsNone(record['input_tokens'])
            self.assertIsNone(record['output_tokens'])
            self.assertIsNone(record['cost_rmb'])
            summary=ledger.summary()
            self.assertTrue(summary['accounting_complete'])
            self.assertFalse(summary['token_usage_complete'])
            self.assertEqual(summary['course_budget_cost_rmb'],'0')
            self.assertEqual(summary['unreconciled_reserved_rmb'],'0')
            self.assertEqual(record['http_status'],500)
            ledger.reserve(MODEL,128,'planner','next','request2')

    def test_success_missing_usage_and_pending_requests_are_not_exempt(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=UsageLedger(folder,'fixture')
            call=ledger.reserve(MODEL,128,'planner','run','request')
            ledger.finish(call['call_id'],MODEL,error='UsageUncertain')
            self.assertFalse(ledger.summary()['accounting_complete'])
            with self.assertRaises(UsageUncertain):
                ledger.reserve(MODEL,128,'planner','next','request2',allow_reserved_unknown=True)

    def test_historical_reconciliation_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=UsageLedger(folder,'fixture')
            call=ledger.reserve(MODEL,128,'planner','run','request')
            ledger.finish(call['call_id'],MODEL,error='TimeoutError')
            state=json.loads(ledger.path.read_text())
            state['calls'][0].update(status='reconciled_conservative', cost_rmb='0.1',accounting_complete=False)
            ledger.path.write_text(json.dumps(state))
            before=ledger.path.read_bytes()
            self.assertEqual(ledger.summary()['course_budget_cost_rmb'],'0')
            with self.assertRaises(ValueError):
                ledger.reconcile_unknown(call['call_id'],'timeout','evidence')
            self.assertEqual(ledger.path.read_bytes(),before)

    def test_development_costs_are_recorded_but_not_capped_or_charged_to_course(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=UsageLedger(folder,'fixture','0.000001',evaluation_stage='development')
            call=ledger.reserve(MODEL,128,'planner','run','request')
            ledger.finish(call['call_id'],MODEL,usage={'prompt_tokens':100,'completion_tokens':50})
            summary=ledger.summary()
            self.assertEqual(summary['actual_recorded_cost_rmb'],'0.0002')
            self.assertEqual(summary['course_budget_cost_rmb'],'0')
            self.assertTrue(summary['within_budget'])
            with self.assertRaises(ValueError):
                UsageLedger(folder,'fixture',evaluation_stage='released').summary()
