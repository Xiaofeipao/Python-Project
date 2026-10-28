"""Prepared contract checks; not executed during initial code creation."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from superagent.config import TaskConfig
from superagent.data import inspect_data
from superagent.features import ImageFeatures
from superagent.metrics import score_model, utility
from superagent.prediction import protect_outputs
from superagent.search import build_model, next_candidate


class SuperAgentContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame = pd.DataFrame({
            "id": [f"{i:03d}" for i in range(40)],
            "value": list(range(40)), "category": ["a", "b"] * 20,
            "label": ["first", "second"] * 20,
        })
        self.train_path = self.root / "train.csv"
        self.frame.to_csv(self.train_path, index=False)

    def config(self, **changes):
        values = {"train_csv": str(self.train_path), "target_column": "label", "id_column": "id"}
        values.update(changes)
        return TaskConfig(**values)

    def test_test_file_is_not_opened_and_holdout_is_disjoint(self):
        config = self.config(test_csv=str(self.root / "does_not_exist.csv"))
        train, validation, profile, split, hashes = inspect_data(config)
        self.assertFalse(set(train.id) & set(validation.id))
        self.assertNotIn("id", profile["features"])
        self.assertNotIn("label", profile["features"])
        self.assertNotIn("test_csv", hashes)
        self.assertEqual(set(split["train_indices"]) | set(split["validation_indices"]), set(range(40)))
        self.assertEqual(len(train) + len(validation), 40)

    def test_reject_target_as_feature_and_overlapping_validation_ids(self):
        with self.assertRaises(ValueError):
            inspect_data(self.config(feature_columns=["label"]))
        validation_path = self.root / "validation.csv"
        self.frame.iloc[:8].to_csv(validation_path, index=False)
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            inspect_data(self.config(validation_csv=str(validation_path)))

    def test_log_loss_minimization_drives_feedback(self):
        history = [
            {"candidate": {"family": "dummy"}, "status": "success", "score": 0.7},
            {"candidate": {"family": "logistic", "C": 1.0, "analyzer": "word"}, "status": "success", "score": 0.2},
            {"candidate": {"family": "logistic", "C": 1.0, "analyzer": "char"}, "status": "success", "score": 0.4},
            {"candidate": {"family": "nb", "alpha": 1.0, "analyzer": "word"}, "status": "failed"},
        ]
        candidate, reason = next_candidate({"data_type": "text"}, history, "log_loss")
        self.assertEqual(candidate["analyzer"], "word")
        self.assertIn("0.2", reason)
        self.assertGreater(utility(0.2, "log_loss"), utility(0.4, "log_loss"))

    def test_tabular_pipeline_roundtrip_and_unknown_categories(self):
        config = self.config()
        train, validation, profile, _, _ = inspect_data(config)
        model = build_model(profile, {"family": "logistic", "C": 1.0}, config)
        model.fit(train, train.label)
        future = validation.drop(columns="label").copy()
        future["category"] = "previously unseen category"
        prediction = model.predict(future)
        destination = self.root / "model.joblib"
        joblib.dump(model, destination)
        np.testing.assert_array_equal(prediction, joblib.load(destination).predict(future))
        self.assertTrue(np.isfinite(score_model(model, validation, validation.label, "log_loss")))

    def test_text_roundtrip_and_image_features(self):
        config = self.config(data_type="text", text_columns=["category"])
        train, validation, profile, _, _ = inspect_data(config)
        model = build_model(profile, {"family": "logistic", "C": 1.0, "analyzer": "word"}, config)
        model.fit(train, train.label)
        self.assertEqual(len(model.predict(validation.drop(columns="label"))), len(validation))
        Image.new("RGB", (8, 8), color="red").save(self.root / "red.png")
        features = ImageFeatures("path", str(self.root), size=4).transform(pd.DataFrame({"path": ["red.png"]}))
        self.assertEqual(features.shape, (1, 4 * 4 * 3 + 3 * 16))
        self.assertTrue(np.isfinite(features).all())

    def test_reserved_output_cannot_destroy_model(self):
        with self.assertRaises(ValueError):
            protect_outputs(self.train_path, self.root / "model.joblib", self.train_path)

    def test_invalid_budgets_are_rejected(self):
        for changes in ({"time_budget_seconds": float("nan")}, {"threads": 0},
                        {"max_trials": True}, {"final_fit_reserve_fraction": 1}):
            with self.assertRaises(ValueError):
                self.config(**changes).validate()


if __name__ == "__main__":
    unittest.main()
