import importlib.util
import unittest
import numpy as np
import pandas as pd
from superagent.preprocessing.transformers import MultiTextFeatures
from superagent.models.tabular import EncodedXGBoost


class MultiTextTests(unittest.TestCase):
    def test_calibrated_and_projected_text_models(self):
        import tempfile, joblib
        from pathlib import Path
        from superagent.preprocessing.transformers import BoundedSVD
        from superagent.planning.search_space import build_model
        from superagent.config import TaskConfig
        X = pd.DataFrame({'a': ['good title', 'bad title'] * 12,
                          'b': ['bad title', 'good title'] * 12})
        profile = {'data_type': 'text', 'text_columns': ['a', 'b']}
        for family in ('linear_svm', 'rbf_svm'):
            candidate = {'family': family, 'C': 1., 'analyzer': 'combined', 'representation': 'separate'}
            if family == 'rbf_svm':
                candidate['svd_components'] = 128
            model = build_model(profile, candidate, TaskConfig(train_csv='unused', target_column='label'))
            model.fit(X, ['yes', 'no'] * 12)
            query = pd.DataFrame({'a': ['unseen headline'], 'b': ['good title']})
            expected = model.predict_proba(query)
            np.testing.assert_allclose(expected.sum(axis=1), 1)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'text.joblib'
                joblib.dump(model, path)
                np.testing.assert_allclose(joblib.load(path).predict_proba(query), expected)
        from scipy import sparse
        tiny = sparse.csr_matrix([[1.], [2.]])
        np.testing.assert_equal(BoundedSVD().fit_transform(tiny), tiny.toarray())

    def test_combined_features_and_weighted_member_cache(self):
        import tempfile
        from joblib import Memory
        from superagent.models.tabular import NativeBoostClassifier, ProbabilityBlend, fit_member
        texts = pd.DataFrame({'a': ['good title', 'bad title'] * 10,
                              'b': ['bad title', 'good title'] * 10})
        features = MultiTextFeatures(['a', 'b'], analyzer='combined').fit(texts)
        swapped = texts.rename(columns={'a': 'b', 'b': 'a'})
        np.testing.assert_allclose(features.transform(texts).toarray(), -features.transform(swapped).toarray())
        X = pd.DataFrame({'number': np.arange(20, dtype=float)})
        y = np.array(['a', 'b'] * 10)
        members = [NativeBoostClassifier(iterations=5, depth=3, random_state=seed) for seed in (1, 2)]
        with tempfile.TemporaryDirectory() as folder:
            model = ProbabilityBlend(members, weights=[.8, .2], cache_dir=folder).fit(X, y)
            cached = Memory(folder, verbose=0).cache(fit_member)
            self.assertTrue(cached.check_call_in_cache(members[0], X, y))
            self.assertFalse(cached.check_call_in_cache(members[0], X, y[::-1]))
            again = ProbabilityBlend(members, weights=[.3, .7], cache_dir=folder).fit(X, y)
            np.testing.assert_allclose(again.predict_proba(X),
                .3 * model.members_[0].predict_proba(X) + .7 * model.members_[1].predict_proba(X))

    def test_shared_vocabulary_and_column_order(self):
        X = pd.DataFrame({'left': ['good title!', 'bad title?'] * 5,
                          'right': ['bad title?', 'good title!'] * 5})
        model = MultiTextFeatures(['left', 'right']).fit(X)
        swapped = X.rename(columns={'left': 'right', 'right': 'left'})
        np.testing.assert_allclose(model.transform(X).toarray(), -model.transform(swapped).toarray())
        fixed = model.vectorizer_.vocabulary_.copy()
        model.transform(pd.DataFrame({'left': ['unseen words'], 'right': ['also unseen']}))
        self.assertEqual(fixed, model.vectorizer_.vocabulary_)

    def test_xgboost_auto_without_cuda_or_torch(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        X = np.arange(80).reshape(40, 2)
        for torch_module in (None, SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))):
            with self.subTest(torch_installed=torch_module is not None):
                with patch.dict('sys.modules', {'torch': torch_module}):
                    model = EncodedXGBoost(iterations=5, depth=3).fit(X, ['accept', 'decline'] * 20)
                self.assertEqual(model.device, 'auto')
                self.assertEqual(model.resource_report_['device'], 'cpu')
                self.assertEqual(model.model_.get_params()['device'], 'cpu')
                np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)

    def test_xgboost_gpu_with_string_classes_and_cpu_inference(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch is not installed (requirements-gpu.txt)')
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        X = np.arange(80).reshape(40, 2)
        model = EncodedXGBoost(iterations=5, depth=3).fit(X, ['accept', 'decline'] * 20)
        self.assertEqual(model.resource_report_['device'], 'cuda')
        self.assertEqual(set(model.classes_), {'accept', 'decline'})
        self.assertEqual(model.model_.get_params()['device'], 'cpu')
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)

    def test_cross_boost_pipeline_roundtrip(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('torch is not installed (requirements-gpu.txt)')
        import torch, tempfile, joblib
        from pathlib import Path
        from superagent.planning.search_space import build_model
        from superagent.config import TaskConfig
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        X = pd.DataFrame({'kind': ['a', 'b', None] * 20})
        profile = {'data_type': 'tabular', 'features': ['kind'], 'numeric_columns': [],
                   'categorical_columns': ['kind']}
        model = build_model(profile, {'family': 'cross_boost_blend', 'iterations': 5,
                                     'gpu_seeds': 3, 'gpu_weight': .25, 'leaf_weight': .15},
                            TaskConfig(train_csv='unused', target_column='label'))
        model.fit(X, ['yes', 'no', 'yes'] * 20)
        query = pd.DataFrame({'kind': ['unknown', None]})
        expected = model.predict_proba(query)
        np.testing.assert_allclose(expected.sum(axis=1), 1)
        self.assertEqual(model.named_steps['classifier'].resource_report_['device'], 'cuda')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cross.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(query), expected)
