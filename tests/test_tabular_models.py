import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier

from superagent.config import TaskConfig
from superagent.search import build_model, next_candidate
from superagent.tabular_models import BalancedExtraTrees, BalancedRandomForest


class TabularModelTests(unittest.TestCase):
    def test_leaf_boost_and_ensembles_handle_missing_and_unseen_categories(self):
        frame = pd.DataFrame({'number': np.arange(60, dtype=float), 'category': ['a', 'b', None] * 20})
        y = np.array(['decline', 'accept', 'other'] * 20)
        profile = {'data_type': 'tabular', 'features': list(frame), 'numeric_columns': ['number'],
                   'categorical_columns': ['category']}
        for family in ('lightgbm', 'catboost_ensemble', 'tabular_blend'):
            candidate = {'family': family, 'iterations': 10, 'depth': 3, 'learning_rate': .1,
                         'l2_leaf_reg': 3, 'num_leaves': 7, 'min_child_samples': 2, 'reg_lambda': 1}
            model = build_model(profile, candidate, TaskConfig(train_csv='unused', target_column='label'))
            model.fit(frame, y)
            query = pd.DataFrame({'number': [np.nan, 1.], 'category': ['unseen', None]})
            expected = model.predict_proba(query)
            np.testing.assert_allclose(expected.sum(axis=1), 1)
            self.assertEqual(set(model.classes_), set(y))
            self.assertEqual(model.predict(query).shape, (2,))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'advanced.joblib'
                joblib.dump(model, path)
                np.testing.assert_allclose(joblib.load(path).predict_proba(query), expected)

    def test_balanced_string_labels_match_explicit_sample_weights(self):
        X = np.arange(60).reshape(30, 2)
        for labels in (["0"] * 20 + ["1"] * 10, ["a"] * 20 + ["b"] * 10):
            y = np.asarray(labels, dtype=object)
            weights = np.array([0.75] * 20 + [1.5] * 10)
            for cls in (BalancedExtraTrees, BalancedRandomForest):
                model = cls(n_estimators=5, class_weight="balanced", random_state=3)
                model.fit(X, y)
                reference = clone(model).set_params(class_weight=None).fit(X, y, sample_weight=weights)
                np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X))
                self.assertEqual(model.class_weight, "balanced")

    def test_native_category_pipeline_roundtrip_and_unknown_categories(self):
        frame = pd.DataFrame({"number": np.arange(30, dtype=float), "category": ["x", "y", None] * 10})
        y = np.array(["decline", "accept", "decline"] * 10)
        profile = {"data_type": "tabular", "features": list(frame), "numeric_columns": ["number"],
                   "categorical_columns": ["category"]}
        candidate = {"family": "catboost", "iterations": 15, "depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 3}
        model = build_model(profile, candidate, TaskConfig(train_csv="unused", target_column="label"))
        model.fit(frame, y)
        query = pd.DataFrame({"number": [2., np.nan], "category": ["unseen", None]})
        probability = model.predict_proba(query)
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(model.predict(query).shape, (2,))
        self.assertEqual(set(model.classes_), set(y))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(query), probability)

    def test_tabular_search_includes_boosting_without_task_identity(self):
        profile = {"data_type": "tabular", "torch_available": False, "class_counts": {"a": 20, "b": 10}}
        first = {"candidate": {"family": "dummy"}, "status": "success", "score": .5}
        candidate, _ = next_candidate(profile, [first], "accuracy")
        self.assertEqual(candidate["family"], "catboost")
