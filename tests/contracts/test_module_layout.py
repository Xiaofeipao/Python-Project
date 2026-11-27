"""The organized packages must preserve historical imports and pickle globals."""

import importlib
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ModuleLayoutContracts(unittest.TestCase):
    def test_model_cache_paths_are_explicit_and_overridable(self):
        from superagent.models.cache import encoder_cache_path, text_cache_root, torch_cache_root
        from superagent.models.text.encoders import encoder_spec

        with tempfile.TemporaryDirectory() as folder:
            text = Path(folder) / 'text'
            images = Path(folder) / 'torch'
            with patch.dict(os.environ, {'SUPERAGENT_MODEL_CACHE': str(text), 'TORCH_HOME': str(images)}):
                self.assertEqual(text_cache_root(), text.resolve())
                self.assertEqual(torch_cache_root(), images.resolve())
                self.assertEqual(encoder_cache_path(encoder_spec('minilm-l6')),
                    text.resolve() / 'sentence-transformers--all-MiniLM-L6-v2-1110a243fdf4')

    def test_legacy_estimators_resolve_to_canonical_classes(self):
        paths = {
            'superagent.models.text_encoder': ('superagent.models.text.encoders',
                ['MiniLMFeatures', 'TextEncoderFeatures']),
            'superagent.embeddings': ('superagent.models.text.encoders', ['MiniLMFeatures']),
            'superagent.models.text_neural': ('superagent.models.text.networks', ['TextNeuralClassifier']),
            'superagent.models.text_finetune': ('superagent.models.text.finetuning',
                ['TransformerFinetuneClassifier']),
            'superagent.preprocessing.transformers': ('superagent.preprocessing.components.tabular',
                ['FrameSchema', 'DatetimeFeatures', 'FrequencyEncoder']),
            'superagent.features': ('superagent.preprocessing.components.text',
                ['MultiTextFeatures', 'PairwiseOverlapFeatures', 'TfidfVectorizer']),
        }
        for legacy, (canonical, names) in paths.items():
            for name in names:
                with self.subTest(module=legacy, name=name):
                    expected = getattr(importlib.import_module(canonical), name)
                    self.assertIs(getattr(importlib.import_module(legacy), name), expected)
                    # Protocol 0 GLOBAL is the lookup used by older trusted pickles.
                    reference = f'c{legacy}\n{name}\n.'.encode('ascii')
                    self.assertIs(pickle.loads(reference), expected)

    def test_search_and_pipeline_compatibility_exports(self):
        from superagent.models.factory import build_model
        from superagent.planning import search, search_space
        from superagent.preprocessing import pipelines, transformers

        self.assertIs(search_space.build_model, build_model)
        for name in ('candidate_key', 'next_candidate', 'metric_candidates', 'allocate_candidates'):
            self.assertIs(getattr(search_space, name), getattr(search, name))
        for name in ('preprocessor', 'tabular_preprocessor', 'hybrid_preprocessor'):
            self.assertIs(getattr(transformers, name), getattr(pipelines, name))


if __name__ == '__main__':
    unittest.main()
