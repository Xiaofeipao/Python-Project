"""High-compute exploration must not starve GPU models behind a CPU winner."""

import unittest

from superagent.config import TaskConfig
from superagent.search import next_candidate


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
        self.assertGreaterEqual(families.count("torch_resnet"), 3)
        self.assertEqual(len({str(record["candidate"]) for record in history}), 12)

    def test_invalid_gpu_options(self):
        for changes in ({"compute_profile": "wrong"}, {"neural_batch_size": 0},
                        {"mixed_precision": 1}, {"gpu_cache_fraction": 1.0}):
            with self.assertRaises(ValueError):
                TaskConfig(train_csv="unused", target_column="label", **changes).validate()


if __name__ == "__main__":
    unittest.main()
