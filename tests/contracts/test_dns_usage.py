import tempfile
import unittest
from superagent.governance.usage_ledger import UsageLedger


class DNSUsageTests(unittest.TestCase):
    def test_usage_jsonl_matches_task_summary(self):
        import json, os
        from pathlib import Path
        from unittest.mock import patch
        from superagent.config import TaskConfig
        from superagent.planning.llm import AutonomousPlanner
        from superagent.governance.usage_ledger import ModelCatalog
        with self.assertRaises(ValueError):
            ModelCatalog.__new__(ModelCatalog)._validate({'model_id': 'DeepSeek-V4.1-Flash',
                'input_rmb_per_million': '0.02', 'output_rmb_per_million': '4'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {'SUPERAGENT_USAGE_ROOT': str(root/'ledger')}):
                planner = AutonomousPlanner(TaskConfig(train_csv='unused', target_column='label'), root, 'usage-test')
                model = {'model_id': 'DeepSeek-Test', 'family': 'DeepSeek', 'context_window_tokens': 1024,
                         'input_rmb_per_million': '1', 'output_rmb_per_million': '4'}
                call = planner.ledger.reserve(model, 64, 'planner', 'test', 'request.json')
                planner.ledger.finish(call['call_id'], model, usage={'prompt_tokens':10, 'completion_tokens':2})
                summary = planner.export_usage()
                record = json.loads((root/'usage.jsonl').read_text())
                self.assertEqual(record['cost_rmb'], summary['actual_total_cost_rmb'])
                self.assertEqual(summary['input_tokens'], 10)

    def test_preconnection_failure_is_zero_but_ambiguous_failure_is_not(self):
        model = {'model_id': 'DeepSeek-Test', 'family': 'DeepSeek', 'context_window_tokens': 1024,
                 'input_rmb_per_million': '1', 'output_rmb_per_million': '4'}
        with tempfile.TemporaryDirectory() as directory:
            ledger = UsageLedger(directory, 'dns-check', '20')
            call = ledger.reserve(model, 64, 'planner', 'run', 'request.json')
            ledger.finish(call['call_id'], model, error='DNSResolutionError', dns_not_sent=True)
            self.assertEqual(ledger.summary()['actual_total_cost_rmb'], '0')
            call = ledger.reserve(model, 64, 'planner', 'run', 'request2.json')
            ledger.finish(call['call_id'], model, error='TimeoutError')
            self.assertTrue(ledger.summary()['accounting_complete'])
            self.assertFalse(ledger.summary()['token_usage_complete'])
            self.assertIsNone(ledger.summary()['actual_total_cost_rmb'])
            self.assertEqual(ledger.summary()['course_budget_cost_rmb'], '0')
