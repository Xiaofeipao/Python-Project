"""Network-free check of ordered pair features and generic candidate discovery."""
import unittest
import numpy as np
from superagent.models.text_encoder import MiniLMFeatures
from superagent.planning.search_space import next_candidate


class EmbeddingTest(unittest.TestCase):
    def test_invalid_plan_cutoff_uses_local_fallback(self):
        from unittest.mock import patch
        from superagent.planning.llm import AutonomousPlanner
        planner = object.__new__(AutonomousPlanner)
        planner._invalid_plans = 2
        planner.output = 'unused'
        with patch('superagent.planning.llm.event') as logged, patch('superagent.planning.llm.write_json'):
            candidate, _ = planner.choose({'data_type':'tabular'}, [{'candidate':{'family':'dummy'},'status':'success','score':.5}], 'accuracy', 0)
        self.assertIsNotNone(candidate)
        logged.assert_called_once()

    def test_ordered_pair_and_discovery(self):
        a, b = np.array([[1., 2.]]), np.array([[3., 4.]])
        np.testing.assert_array_equal(MiniLMFeatures.combine([a, b]), [[1, 2, 3, 4, -2, -2, 3, 8]])
        import importlib.util
        if importlib.util.find_spec('sentence_transformers'):
            for modality in ('text', 'tabular'):
                candidate, _ = next_candidate({'data_type': modality, 'text_columns': ['x'], 'minilm_available': True},
                    [{'candidate': {'family': 'dummy'}, 'status': 'success', 'score': .5}], 'accuracy')
                self.assertEqual(candidate['family'], 'minilm_logistic')
