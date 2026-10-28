"""Neural checkpoint portability and CUDA execution contracts."""

import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from superagent.torch_models import TorchClassifier, TorchEnsemble
from superagent.config import TaskConfig
from superagent.search import build_model


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch package is absent")
class TorchContracts(unittest.TestCase):
    def test_train_only_stopping_and_full_data_refit(self):
        rng = np.random.default_rng(52)
        X = rng.normal(size=(80, 8)).astype(np.float32)
        y = np.where(X[:, 0] > 0, "a", "b")
        model = TorchClassifier(architecture="residual_mlp", hidden=32, depth=2,
                                early_stopping=True, epochs=8, patience=2, device="cpu")
        model.fit(X, y)
        report = model.training_report_
        self.assertGreater(report["internal_monitor_rows"], 0)
        self.assertEqual(report["training_rows"] + report["internal_monitor_rows"], len(y))
        model.set_params(refit_epochs=report["best_epoch"])
        model.fit(X, y)
        self.assertEqual(model.training_report_["internal_monitor_rows"], 0)
        self.assertEqual(model.training_report_["training_rows"], len(y))
        self.assertEqual(model.training_report_["epochs"], report["best_epoch"])

    def test_seed_ensemble_probability_and_roundtrip(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(60, 6)).astype(np.float32)
        y = np.where(X[:, 0] > 0, "a", "b")
        model = TorchEnsemble(member_params={"hidden": 32, "epochs": 3}, size=2, device="cpu")
        model.fit(X, y)
        expected = np.mean([member.predict_proba(X) for member in model.estimators_], axis=0)
        np.testing.assert_allclose(model.predict_proba(X), expected)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_allclose(restored.predict_proba(X), expected)
        restored.set_params(refit_epochs=[1, 2])
        restored.fit(X, y)
        self.assertEqual(restored.training_report_["best_epoch"], [1, 2])

    def test_amp_gpu_cache_and_residual_image_network(self):
        import torch

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        rng = np.random.default_rng(19)
        X = rng.random((24, 8 * 8 * 3 + 48), dtype=np.float32)
        model = TorchClassifier(architecture="resnet", image_size=8, hidden=32, width=8,
                                epochs=2, mixed_precision=True, augment=True, normalize_images=True,
                                cosine_schedule=True, batch_size=12)
        model.fit(X, np.array(["a", "b"] * 12))
        probability = model.predict_proba(X)
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(probability).all())
        if torch.cuda.is_available():
            self.assertEqual(model.training_report_["data_placement"], "gpu_cached")
            self.assertNotEqual(model.training_report_["precision"], "float32")
        model.set_params(device="cpu")
        np.testing.assert_allclose(model.predict_proba(X), probability, atol=1e-4)

    def test_complete_image_pipeline_is_fitted_and_predicts(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, color in enumerate(("red", "blue", "red", "blue")):
                Image.new("RGB", (8, 8), color=color).save(Path(folder) / f"{index}.png")
            frame = pd.DataFrame({"path": [f"{i}.png" for i in range(4)]})
            config = TaskConfig(train_csv="unused.csv", target_column="label", image_root=folder, device="cpu")
            profile = {"data_type": "image", "image_column": "path"}
            candidate = {"family": "torch_cnn", "size": 8, "hidden": 32, "epochs": 2,
                         "learning_rate": 0.001, "weight_decay": 0.001}
            pipeline = build_model(profile, candidate, config)
            pipeline.fit(frame, ["red", "blue", "red", "blue"])
            self.assertEqual(pipeline.predict_proba(frame).shape, (4, 2))
            self.assertEqual(len(pipeline.predict(frame)), 4)

    def test_sparse_training_and_device_independent_checkpoint(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(64, 6)).astype(np.float32)
        y = np.where(X[:, 0] > 0, "yes", "no")
        classifier = TorchClassifier(epochs=8, hidden=32, batch_size=16, device="cpu")
        classifier.fit(sparse.csr_matrix(X), y)
        before = classifier.predict_proba(X)
        buffer = io.BytesIO()
        joblib.dump(classifier, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_allclose(before, restored.predict_proba(X), atol=1e-6)
        self.assertTrue(all(isinstance(value, np.ndarray) for value in restored.weights_.values()))
        np.testing.assert_allclose(before.sum(axis=1), 1, atol=1e-6)
        self.assertLess(classifier.training_report_["training_loss"][-1], classifier.training_report_["training_loss"][0])

    def test_cnn_on_available_device_and_cpu_reload(self):
        import torch

        rng = np.random.default_rng(5)
        X = rng.random((20, 8 * 8 * 3 + 48), dtype=np.float32)
        y = np.array(["a", "b"] * 10)
        model = TorchClassifier(architecture="cnn", image_size=8, epochs=2, hidden=32, batch_size=10)
        model.fit(X, y)
        gpu_predictions = model.predict_proba(X)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(model.training_report_["device"], expected)
        model.set_params(device="cpu")
        np.testing.assert_allclose(model.predict_proba(X), gpu_predictions, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
