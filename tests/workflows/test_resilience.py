import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator
from superagent.runtime.orchestrator import train_agent
from superagent.config import TaskConfig
from superagent.runtime.worker import fit_with_fallback
from superagent.planning.llm import candidate_menu


class AllocationFailure(BaseEstimator):
    def __init__(self, batch_size=4, device='cuda'):
        self.batch_size, self.device = batch_size, device

    def fit(self, X, y):
        if self.device != 'cpu':
            raise RuntimeError('CUDA out of memory')
        return self


class ResilienceTests(unittest.TestCase):
    def test_strict_default_never_predicts_before_final_selection(self):
        from unittest.mock import patch
        import superagent.evaluation.inference as prediction
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, offset in [('train', 0), ('val', 20), ('test', 40)]:
                pd.DataFrame({'id': range(offset, offset+10), 'x': range(10),
                              'label': ['a', 'b']*5}).to_csv(root/f'{name}.csv', index=False)
            original = prediction.checked_input
            observed = []
            def guarded(path, metadata):
                events = [json.loads(s) for s in (root/'run/events.jsonl').read_text().splitlines()]
                self.assertIn('selection_completed', [e['event'] for e in events])
                self.assertNotIn('partial_test_read_started', [e['event'] for e in events])
                observed.append(str(path))
                return original(path, metadata)
            with patch.object(prediction, 'checked_input', side_effect=guarded):
                train_agent(TaskConfig(train_csv=str(root/'train.csv'), validation_csv=str(root/'val.csv'),
                    test_csv=str(root/'test.csv'), target_column='label', id_column='id',
                    enable_torch=False, device='cpu', max_trials=1, time_budget_seconds=300), root/'run')
            self.assertEqual(len(observed), 1)
            self.assertFalse((root/'run/partial').exists())

    def test_oom_reduces_batch_then_uses_cpu(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch is not installed (requirements-gpu.txt)')
        model, steps = fit_with_fallback(AllocationFailure(), [[1]], [0])
        self.assertEqual(model.device, 'cpu')
        self.assertEqual(len(steps), 2)

    def test_cpu_partial_output_and_final_test_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, offset in [('train', 0), ('val', 20), ('test', 40)]:
                pd.DataFrame({'id': range(offset, offset+10), 'x': range(10),
                              'label': ['a', 'b']*5}).to_csv(root/f'{name}.csv', index=False)
            result = train_agent(TaskConfig(train_csv=str(root/'train.csv'), validation_csv=str(root/'val.csv'),
                test_csv=str(root/'test.csv'), target_column='label', id_column='id', device='cpu',
                enable_torch=False, progressive_output=True, max_trials=1, time_budget_seconds=300), root/'run')
            self.assertFalse(result['test_set_used_for_selection'])
            self.assertTrue((root/'run/partial/v001/predictions.csv').exists())
            events = [json.loads(s) for s in (root/'run/events.jsonl').read_text().splitlines()]
            selected = next(e for e in events if e['event']=='selection_completed')
            read = next(e for e in events if e['event']=='test_read_started')
            self.assertLessEqual(selected['timestamp_utc'], read['timestamp_utc'])
            self.assertTrue((root/'run/predictions.csv').exists())
