"""Explicit project-local locations for general pretrained model resources."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def text_cache_root():
    return Path(os.environ.get('SUPERAGENT_MODEL_CACHE',
                               PROJECT_ROOT / '.pretrained_cache' / 'text')).expanduser().resolve()


def torch_cache_root():
    return Path(os.environ.get('TORCH_HOME',
                               PROJECT_ROOT / '.pretrained_cache' / 'torch')).expanduser().resolve()


def encoder_cache_path(spec):
    return text_cache_root() / (spec['model_id'].replace('/', '--') + '-' + spec['revision'][:12])
