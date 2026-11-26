"""Hidden-task boundaries and final-only test access, without external services."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from superagent.config import TaskConfig
from superagent.preprocessing.datasets import inspect_data
from superagent.preprocessing.transformers import ImageFeatures, TfidfVectorizer
from superagent.planning.search_space import allocate_candidates, build_model
from superagent.runtime.orchestrator import train_agent, run_worker


class HiddenSafety(unittest.TestCase):
    def test_tiny_and_singleton_classes_keep_every_training_class(self):
        with tempfile.TemporaryDirectory() as folder:
            for labels in (['a', 'a', 'b', 'b'], ['a', 'a', 'b', 'c', 'c']):
                path = Path(folder) / 'train.csv'
                pd.DataFrame({'x': range(len(labels)), 'y': labels}).to_csv(path, index=False)
                train, validation, _, _, _ = inspect_data(TaskConfig(train_csv=str(path), target_column='y'))
                self.assertEqual(set(train.y), set(labels))
                self.assertGreater(len(validation), 0)

    def test_empty_text_vocabulary_fits_training_only(self):
        vectorizer = TfidfVectorizer(min_df=2).fit(['', '', ''])
        original = vectorizer.vocabulary_.copy()
        matrix = vectorizer.transform(['unseen test words'])
        self.assertEqual(matrix.shape, (1, 1))
        self.assertEqual(matrix.nnz, 0)
        self.assertEqual(original, vectorizer.vocabulary_)

    def test_nonempty_tfidf_matches_existing_sklearn_behavior(self):
        from sklearn.feature_extraction.text import TfidfVectorizer as Original
        documents = ['alpha beta', 'beta gamma', 'alpha gamma']
        for options in ({}, {'analyzer': 'char_wb', 'ngram_range': (3, 6)}):
            old, new = Original(**options), TfidfVectorizer(**options)
            np.testing.assert_array_equal(old.fit_transform(documents).toarray(),
                                          new.fit_transform(documents).toarray())
            self.assertEqual(old.vocabulary_, new.vocabulary_)

    def test_missing_features_multiclass_and_unknown_categories(self):
        frame = pd.DataFrame({'value': [np.nan]*12, 'city': ['a', 'b']*6})
        profile = {'data_type': 'tabular', 'features': ['value', 'city'],
                   'numeric_columns': ['value'], 'categorical_columns': ['city']}
        model = build_model(profile, {'family': 'logistic', 'C': 1}, TaskConfig(train_csv='unused', target_column='y'))
        model.fit(frame, ['one', 'two', 'three']*4)
        probability = model.predict_proba(pd.DataFrame({'value': [np.nan], 'city': ['unknown']}))
        self.assertEqual(probability.shape, (1, 3))
        np.testing.assert_allclose(probability.sum(axis=1), 1)

    def test_corrupt_image_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'bad.png').write_bytes(b'not an image')
            with self.assertRaises(OSError):
                ImageFeatures('path', folder).fit_transform(pd.DataFrame({'path': ['bad.png']}))

    def test_allocation_uses_resources_without_task_identity(self):
        candidates = [{'family': 'minilm_logistic', 'C': 1}, {'family': 'logistic', 'C': 1},
                      {'family': 'rbf_svm', 'C': 1}]
        profile = {'data_type': 'text', 'rows': 30000, 'remaining_search_seconds': 20}
        expected = [{'family': 'logistic', 'C': 1}]
        self.assertEqual(allocate_candidates(candidates, dict(profile, task_name='released'), []), expected)
        self.assertEqual(allocate_candidates(candidates, dict(profile, task_name='hidden'), []), expected)

    def test_failed_refits_use_checkpoint_after_search_before_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, shift in [('train', 0), ('val', 20), ('test', 40)]:
                pd.DataFrame({'id': np.arange(12)+shift, 'x': np.arange(12),
                              'label': ['a', 'b']*6}).to_csv(root/f'{name}.csv', index=False)
            allowances = []
            def worker(request, directory, timeout, threads):
                if request['mode'] == 'final':
                    allowances.append(timeout)
                    return {'status': 'timeout', 'error': 'Injected refit timeout'}
                return run_worker(request, directory, timeout, threads)
            def choose(profile, history, *args):
                return ({'family': 'dummy'} if not history else {'family': 'logistic', 'C': 1}), 'fixture'
            with patch('superagent.runtime.orchestrator.run_worker', side_effect=worker), patch(
                    'superagent.planning.llm.AutonomousPlanner.choose', side_effect=choose), patch(
                    'superagent.runtime.orchestrator.time.monotonic', return_value=1000):
                result = train_agent(TaskConfig(train_csv=str(root/'train.csv'), validation_csv=str(root/'val.csv'),
                    test_csv=str(root/'test.csv'), target_column='label', id_column='id', max_trials=2,
                    time_budget_seconds=300, trial_timeout_seconds=60, enable_torch=False,
                    progressive_output=True), root/'run')
            self.assertTrue(result['partial'])
            self.assertFalse(result['final_refit_succeeded'])
            self.assertLessEqual(allowances[0], 60)
            events = [json.loads(line) for line in (root/'run/events.jsonl').read_text(encoding='utf-8').splitlines()]
            names = [e['event'] for e in events]
            self.assertLess(names.index('validated_checkpoint_selected'), names.index('selection_completed'))
            self.assertLess(names.index('selection_completed'), names.index('test_read_started'))
            self.assertEqual(names.count('test_read_started'), 1)
            self.assertNotIn('partial_test_read_started', names)
            self.assertTrue((root/'run/predictions.csv').exists())


if __name__ == '__main__':
    unittest.main()
