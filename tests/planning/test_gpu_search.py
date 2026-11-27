"""High-compute exploration must not starve GPU models behind a CPU winner."""

import unittest

from superagent.config import TaskConfig
from superagent.planning.search_space import next_candidate


class GpuPlannerContracts(unittest.TestCase):
    def test_explores_neural_capacity_and_ensembles_even_if_cpu_is_better(self):
        profile = {"data_type": "image", "torch_available": True,
                   "compute_profile": "gpu_large", "class_counts": {"a": 30, "b": 30}}
        history = []
        for _ in range(12):
            candidate, _ = next_candidate(profile, history, "accuracy")
            self.assertIsNotNone(candidate)
            self.assertNotIn(candidate, [item["candidate"] for item in history])
            history.append({"candidate": candidate, "status": "success",
                            "score": 0.8 if candidate["family"].startswith("torch_") else 0.9})
        families = [record["candidate"]["family"] for record in history]
        self.assertIn("torch_ensemble", families)
        self.assertGreaterEqual(families.count("torch_resnet"), 2)
        self.assertGreaterEqual(families.count("torch_pretrained"), 2)
        self.assertEqual(len({str(record["candidate"]) for record in history}), 12)

    def test_pairwise_text_exposes_semantic_mlp_fusion(self):
        profile = {"data_type": "text", "torch_available": True,
                   "text_encoder_available": True, "compute_profile": "gpu_large",
                   "text_columns": ["headline_A", "headline_B"],
                   "neural_batch_size": 64, "class_counts": {"a": 30, "b": 30}}
        history = [{"candidate": {"family": "dummy"}, "status": "success", "score": .5}]
        candidate, _ = next_candidate(profile, history, "accuracy")
        self.assertEqual(candidate["family"], "pairwise_semantic_mlp_blend")
        self.assertAlmostEqual(candidate["semantic_weight"] + candidate["mlp_weight"], .405)

    def test_invalid_gpu_options(self):
        for changes in ({"compute_profile": "wrong"}, {"neural_batch_size": 0},
                        {"mixed_precision": 1}, {"gpu_cache_fraction": 1.0}):
            with self.assertRaises(ValueError):
                TaskConfig(train_csv="unused", target_column="label", **changes).validate()


if __name__ == "__main__":
    unittest.main()
