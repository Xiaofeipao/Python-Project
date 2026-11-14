import tempfile
import unittest
from superagent.usage import UsageLedger


class DNSUsageTests(unittest.TestCase):
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
            self.assertFalse(ledger.summary()['accounting_complete'])
