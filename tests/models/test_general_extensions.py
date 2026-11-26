"""Generalisation regressions: image transfer, hybrid inputs and identity blindness."""

import importlib.util
import io
import re
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from superagent.config import TaskConfig
from superagent.planning.search_space import build_model, candidate_key, next_candidate


def write_images(folder, count=12):
    for index in range(count):
        color = "red" if index % 2 == 0 else "blue"
        image = Image.new("RGB", (16, 16), color=color)
        image.save(Path(folder) / f"{index}.png")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "Optional torch/torchvision packages are absent")
class PretrainedImageContracts(unittest.TestCase):
    def test_frozen_image_features_cpu_roundtrip_without_prediction_download(self):
        from unittest.mock import patch
        from superagent.models.neural import TorchClassifier
        from superagent.evaluation.inference import _replace_image_roots
        with tempfile.TemporaryDirectory() as folder:
            write_images(folder)
            frame = pd.DataFrame({'path': [f'{i}.png' for i in range(12)]})
            config = TaskConfig(train_csv='unused', target_column='label', image_root=folder, device='cpu')
            profile = {'data_type': 'image', 'image_column': 'path'}
            candidate = {'family': 'frozen_image_logistic', 'C': 1., 'size': 32,
                         'pretrained': False, 'batch_size': 4}
            model = build_model(profile, candidate, config).fit(frame, ['a', 'b']*6)
            expected = model.predict_proba(frame)
            self.assertEqual(expected.shape, (12, 2))
            self.assertTrue(model.named_steps['preprocess'].resource_report_['frozen'])
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            restored = joblib.load(buffer)
            original = TorchClassifier._network
            def no_download(probe, load_weights=False):
                self.assertFalse(load_weights)
                return original(probe, load_weights=False)
            with patch.object(TorchClassifier, '_network', no_download):
                np.testing.assert_allclose(restored.predict_proba(frame), expected)
            self.assertEqual(_replace_image_roots(restored, folder), 1)

    def test_offline_initial_download_falls_back_without_claiming_weights(self):
        from unittest.mock import patch
        from superagent.models.neural import TorchClassifier
        original = TorchClassifier._network
        def offline(model, load_weights=False):
            if load_weights:
                raise OSError('Injected initial download failure')
            return original(model, load_weights=False)
        with tempfile.TemporaryDirectory() as folder:
            write_images(folder)
            frame = pd.DataFrame({'path': [f'{i}.png' for i in range(12)]})
            config = TaskConfig(train_csv='unused', target_column='label', image_root=folder, device='cpu')
            model = build_model({'data_type': 'image', 'image_column': 'path'},
                {'family': 'torch_pretrained', 'architecture': 'pretrained_resnet18', 'size': 16,
                 'epochs': 1, 'learning_rate': .001, 'weight_decay': .001, 'batch_size': 6}, config)
            with patch.object(TorchClassifier, '_network', offline):
                model.fit(frame, ['red', 'blue']*6)
                self.assertEqual(model.predict_proba(frame).shape, (12, 2))
            report = model.named_steps['classifier'].training_report_
            self.assertIn('download failure', report['pretrained_weight_fallback'])
            self.assertIsNone(report['pretrained_weights'])

    def test_explicit_weight_source_and_full_hash_are_recorded(self):
        import hashlib
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from superagent.models.neural import TorchClassifier
        from torchvision import models
        with tempfile.TemporaryDirectory() as folder:
            payload = b'controlled weight fixture'
            digest = hashlib.sha256(payload).hexdigest()
            name = f'resnet18-{digest[:8]}.pth'
            directory = Path(folder)/'checkpoints'
            directory.mkdir()
            (directory/name).write_bytes(payload)
            weights = MagicMock(url='https://download.pytorch.org/models/'+name)
            weights.get_state_dict.return_value = {}
            network = MagicMock()
            network.fc.in_features = 4
            classifier = TorchClassifier(architecture='pretrained_resnet18')
            classifier.classes_ = np.array(['a', 'b'])
            with patch.object(models, 'ResNet18_Weights', SimpleNamespace(IMAGENET1K_V1=weights)), patch.object(
                    models, 'resnet18', return_value=network), patch('torch.hub.get_dir', return_value=folder):
                classifier._network(load_weights=True)
                self.assertEqual(classifier.pretrained_weights_['sha256'], digest)
                weights.get_state_dict.assert_called_once_with(progress=True, check_hash=True)
                (directory/name).write_bytes(b'corrupted cache')
                with self.assertRaisesRegex(ValueError, 'hash differs'):
                    classifier._network(load_weights=True)

    def test_pretrained_backbone_roundtrip_without_download_at_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            write_images(folder)
            frame = pd.DataFrame({"path": [f"{index}.png" for index in range(12)]})
            labels = ["red" if index % 2 == 0 else "blue" for index in range(12)]
            config = TaskConfig(train_csv="unused.csv", target_column="label",
                                image_root=folder, device="cpu")
            profile = {"data_type": "image", "image_column": "path"}
            candidate = {"family": "torch_pretrained", "architecture": "pretrained_resnet18",
                         "size": 16, "epochs": 1, "learning_rate": .001, "weight_decay": .001,
                         "pretrained": False, "imagenet_normalize": True, "tta": True,
                         "batch_size": 6}
            model = build_model(profile, candidate, config)
            model.fit(frame, labels)
            probability = model.predict_proba(frame)
            self.assertEqual(probability.shape, (12, 2))
            self.assertTrue(np.isfinite(probability).all())
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            restored = joblib.load(buffer)
            np.testing.assert_allclose(restored.predict_proba(frame), probability, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("catboost"), "Optional catboost package is absent")
    def test_image_metadata_blend_keeps_tabular_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            write_images(folder, 24)
            frame = pd.DataFrame({
                "path": [f"{index}.png" for index in range(24)],
                "price": np.linspace(1, 24, 24),
                "city": ["A", "B", "C"] * 8,
            })
            labels = ["red" if index % 2 == 0 else "blue" for index in range(24)]
            config = TaskConfig(train_csv="unused.csv", target_column="label",
                                image_root=folder, device="cpu")
            profile = {"data_type": "image", "image_column": "path",
                       "features": ["path", "price", "city"],
                       "tabular_columns": ["price", "city"],
                       "numeric_columns": ["price"], "categorical_columns": ["city"],
                       "datetime_columns": [], "categorical_cardinality": {"city": 3}}
            candidate = {"family": "image_meta_blend", "image_weight": .7, "size": 16,
                         "epochs": 1, "learning_rate": .001, "pretrained": False,
                         "meta_iterations": 40}
            model = build_model(profile, candidate, config)
            model.fit(frame, labels)
            self.assertEqual(model.predict_proba(frame).shape, (24, 2))
            self.assertEqual(len(model.predict(frame)), 24)


class HybridContracts(unittest.TestCase):
    def test_text_blend_diff_fuses_signed_pair_features(self):
        frame = pd.DataFrame({
            "left": ["great battery life", "poor screen", "fast and light",
                     "heavy and slow", "excellent camera", "bad speaker"] * 2,
            "right": ["poor screen", "great battery life", "heavy and slow",
                      "fast and light", "bad speaker", "excellent camera"] * 2,
        })
        labels = ["same", "different"] * 6
        profile = {"data_type": "text", "text_columns": ["left", "right"],
                   "features": ["left", "right"], "tabular_columns": []}
        config = TaskConfig(train_csv="unused.csv", target_column="label")
        model = build_model(profile, {"family": "text_blend_diff", "linear_weight": .5,
                                      "C": .1}, config)
        model.fit(frame, labels)
        self.assertEqual(model.predict_proba(frame).shape, (12, 2))

    def test_hybrid_text_and_tabular_union(self):
        frame = pd.DataFrame({
            "review": ["great battery life", "poor screen quality", "fast and light",
                       "heavy and slow", "excellent camera", "bad speaker"] * 2,
            "price": [10, 20, 30, 40, 50, 60] * 2,
            "city": ["A", "B", "C"] * 4,
        })
        labels = ["good", "bad"] * 6
        profile = {"data_type": "text", "text_columns": ["review"],
                   "features": ["review", "price", "city"],
                   "tabular_columns": ["price", "city"],
                   "numeric_columns": ["price"], "categorical_columns": ["city"],
                   "datetime_columns": [], "categorical_cardinality": {"city": 3}}
        config = TaskConfig(train_csv="unused.csv", target_column="label")
        model = build_model(profile, {"family": "hybrid_logistic", "C": 1., "analyzer": "word"}, config)
        model.fit(frame, labels)
        probability = model.predict_proba(frame)
        self.assertEqual(probability.shape, (12, 2))
        self.assertEqual(len(model.predict(frame)), 12)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Optional xgboost package is absent")
    def test_hybrid_gbdt_accepts_sparse_text_and_tabular_union(self):
        frame = pd.DataFrame({
            "review": ["great battery life", "poor screen quality", "fast and light",
                       "heavy and slow", "excellent camera", "bad speaker"] * 2,
            "price": [10, 20, 30, 40, 50, 60] * 2,
            "city": ["A", "B", "C"] * 4,
        })
        labels = ["good", "bad"] * 6
        profile = {"data_type": "text", "text_columns": ["review"],
                   "features": ["review", "price", "city"],
                   "tabular_columns": ["price", "city"],
                   "numeric_columns": ["price"], "categorical_columns": ["city"],
                   "datetime_columns": [], "categorical_cardinality": {"city": 3}}
        config = TaskConfig(train_csv="unused.csv", target_column="label")
        model = build_model(profile, {"family": "hybrid_xgboost", "iterations": 50, "depth": 4,
                                      "learning_rate": .05, "reg_lambda": 10,
                                      "analyzer": "combined", "text_max_features": 300,
                                      "device": "cpu"}, config)
        model.fit(frame, labels)
        self.assertEqual(model.predict_proba(frame).shape, (12, 2))
        self.assertEqual(len(model.predict(frame)), 12)

    def test_datetime_expansion_and_frequency_fallback(self):
        dates = pd.date_range("2026-01-01", periods=24, freq="D")
        frame = pd.DataFrame({
            "when": dates.astype(str),
            "city": [f"city-{index % 200}" for index in range(24)],
            "value": np.arange(24, dtype=float),
        })
        labels = ["a", "b"] * 12
        profile = {"data_type": "tabular", "features": ["when", "city", "value"],
                   "numeric_columns": ["value"], "categorical_columns": ["city"],
                   "datetime_columns": ["when"], "categorical_cardinality": {"city": 200}}
        config = TaskConfig(train_csv="unused.csv", target_column="label")
        # Frequency encoding is retained as an explicit fallback; the default
        # remains the verified one-hot representation.
        model = build_model(profile, {"family": "logistic", "C": 1., "encoding": "frequency"}, config)
        model.fit(frame, labels)
        self.assertEqual(model.predict_proba(frame).shape, (24, 2))


class ImagePathContracts(unittest.TestCase):
    def test_direct_config_uses_train_parent_for_implicit_image_root(self):
        config = TaskConfig(train_csv="data/train.csv", target_column="label")
        self.assertEqual(config.image_root, "data")

    def test_image_root_defaults_to_train_csv_parent(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / "data" / "train.csv").write_text("path,label\n", encoding="utf-8")
            (root / "data" / "test.csv").write_text("path,label\n", encoding="utf-8")
            config_path = root / "task.json"
            config_path.write_text(json.dumps({
                "data_type": "image", "train_csv": "data/train.csv",
                "test_csv": "data/test.csv", "target_column": "label",
                "image_column": "path", "eval": "development",
            }).replace('"eval": "development"', '"evaluation_stage": "development"'), encoding="utf-8")
            config = TaskConfig.load(config_path)
            self.assertEqual(config.image_root, str((root / "data").resolve()))

    def test_nested_image_roots_are_replaced(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline
        from superagent.evaluation.inference import _replace_image_roots
        from superagent.models.tabular import ProbabilityBlend
        from superagent.preprocessing.transformers import ImageFeatures
        with tempfile.TemporaryDirectory() as folder:
            feature = ImageFeatures("path", folder, 16)
            member = Pipeline([("preprocess", feature), ("classifier", DummyClassifier())])
            wrapper = Pipeline([
                ("preprocess", "passthrough"),
                ("classifier", ProbabilityBlend([member])),
            ])
            self.assertEqual(_replace_image_roots(wrapper, folder + "-new"), 1)
            self.assertEqual(feature.root, str(Path(folder + "-new").resolve()))


class IdentityContracts(unittest.TestCase):
    def test_task_and_dataset_names_do_not_change_candidates(self):
        base = {"data_type": "tabular", "class_counts": {"a": 30, "b": 10},
                "compute_profile": "gpu_large", "torch_available": True,
                "features": ["x"], "numeric_columns": ["x"], "categorical_columns": []}
        first = [dict(base, task_name="coupon", dataset_name="coupon_release")]
        second = [dict(base, task_name="unseen-alpha", dataset_name="private-holdout")]
        histories = [[], []]
        for index in range(6):
            left, _ = next_candidate(first[0], histories[0], "accuracy")
            right, _ = next_candidate(second[0], histories[1], "accuracy")
            self.assertEqual(candidate_key(left), candidate_key(right))
            histories[0].append({"candidate": left, "status": "success", "score": .5 + index / 100})
            histories[1].append({"candidate": right, "status": "success", "score": .5 + index / 100})

    def test_source_has_no_task_name_dispatch(self):
        root = Path(__file__).resolve().parents[2] / "superagent"
        offenders = [str(path) for path in root.rglob("*.py")
                     if re.search(r"coupon|headline", path.read_text(encoding="utf-8"), re.I)]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
