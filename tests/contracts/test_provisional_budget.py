import tempfile
import unittest
from superagent.governance.usage_ledger import UsageLedger, BudgetExhausted
from superagent.config import TaskConfig


class ProvisionalBudgetTests(unittest.TestCase):
    def test_failed_unknown_never_accumulates_course_reservations(self):
        model = {'model_id':'DeepSeek-Test','family':'DeepSeek','context_window_tokens':1000000,
                 'input_rmb_per_million':'1','output_rmb_per_million':'4'}
        with tempfile.TemporaryDirectory() as folder:
            ledger=UsageLedger(folder,'test','20')
            for _ in range(21):
                c=ledger.reserve(model,64,'planner','run','request',allow_reserved_unknown=True)
                ledger.finish(c['call_id'],model,error='TimeoutError')
            self.assertEqual(ledger.summary()['unreconciled_reserved_rmb'],'0')
            self.assertIsNone(ledger.summary()['actual_total_cost_rmb'])
            self.assertTrue(ledger.summary()['accounting_complete'])
            ledger.reserve(model,64,'planner','run','request',allow_reserved_unknown=True)
        with self.assertRaises(ValueError):
            TaskConfig(train_csv='unused',target_column='y',allow_provisional_accounting=True,
                       evaluation_stage='released').validate()
