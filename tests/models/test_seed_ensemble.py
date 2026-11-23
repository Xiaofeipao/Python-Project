import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from superagent.config import TaskConfig
from superagent.models.ensemble import SeedAveragedModel


def fitted(seed):
    X = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.tile([0.0, 1.0], 10)})
    y = np.array([0, 1] * 10)
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(random_state=seed))])
    model.fit(X, y)
    return model, X, y


class SeedAveragedModelTests(unittest.TestCase):
    def test_probabilities_are_the_member_mean_and_survive_joblib(self):
        first, X, y = fitted(1)
        second, _, _ = fitted(2)
        ensemble = SeedAveragedModel([first, second], [1, 2])
        query = pd.DataFrame({"a": [-1.0, 5.0, 17.0], "b": [0.0, 1.0, 0.0]})
        expected = (first.predict_proba(query) + second.predict_proba(query)) / 2
        np.testing.assert_allclose(ensemble.predict_proba(query), expected)
        np.testing.assert_allclose(ensemble.predict_proba(query).sum(axis=1), 1)
        np.testing.assert_array_equal(ensemble.predict(query),
                                      np.asarray(ensemble.classes_)[np.argmax(expected, axis=1)])
        self.assertEqual([str(c) for c in ensemble.classes_], [str(c) for c in first.classes_])
        self.assertIn("classifier", ensemble.named_steps)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ensemble.joblib"
            joblib.dump({"model": ensemble, "metadata": {"schema_version": 1}}, path)
            loaded = joblib.load(path)["model"]
            np.testing.assert_allclose(loaded.predict_proba(query), expected)
            self.assertEqual(loaded.seeds, [1, 2])

    def test_mismatched_class_order_is_rejected(self):
        first, _, _ = fitted(1)

        class Stub:
            def __init__(self, classes):
                self.classes_ = np.asarray(classes)

            def predict_proba(self, frame):
                return np.tile([0.5, 0.5], (len(frame), 1))

        with self.assertRaises(ValueError):
            SeedAveragedModel([first, Stub(["1", "0"])], [1, 2])

    def test_config_bounds(self):
        config = TaskConfig(train_csv="unused", target_column="label", final_fit_seeds=5)
        config.validate()
        for invalid in (0, 6, True):
            with self.assertRaises(ValueError):
                TaskConfig(train_csv="unused", target_column="label", final_fit_seeds=invalid).validate()


if __name__ == "__main__":
    unittest.main()
