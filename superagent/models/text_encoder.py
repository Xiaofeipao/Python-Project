"""Frozen general-purpose encoder; no task-name dispatch or label fitting."""
import os
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


class MiniLMFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, text=False, device='auto', batch_size=128):
        self.columns = columns
        self.text = text
        self.device = device
        self.batch_size = batch_size

    def fit(self, X, y=None):
        from sentence_transformers import SentenceTransformer
        spec = encoder_spec('minilm-l6')
        path = _download_snapshot(spec)
        self.encoder_ = SentenceTransformer(path, device='cpu', trust_remote_code=False,
                                            local_files_only=True)
        self.model_metadata_ = {'model_id': MODEL_ID, 'revision': REVISION,
                                'purpose': 'frozen feature extraction', 'fine_tuned': False}
        self.resource_report_ = {
            'backend': 'sentence_transformers', 'frozen': True, 'model_id': MODEL_ID,
            'revision': REVISION, 'output_features': _embedding_dimension(self.encoder_),
        }
        return self

    def transform(self, X):
        import torch
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.device == 'auto' else self.device
        frame = X[self.columns].fillna('').astype(str)
        if self.text:
            documents = [frame[c].tolist() for c in self.columns]
        else:
            documents = [frame.apply(lambda row: '; '.join(f'{c}: {row[c]}' for c in self.columns), axis=1).tolist()]
        try:
            parts = [self.encoder_.encode(d, batch_size=self.batch_size, device=device,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                     for d in documents]
        finally:
            self.encoder_.to('cpu')
        return self.combine(parts)

    @staticmethod
    def combine(parts):
        # Preserve column order; pair interactions apply to any two text columns.
        if len(parts) == 2:
            a, b = parts
            parts = [a, b, a-b, a*b]
        return np.concatenate(parts, axis=1).astype(np.float32)


# A small, deliberately explicit registry.  Every entry is pinned to a commit;
# a model is never selected by a floating branch name at training time.
ENCODER_REGISTRY = {
    'minilm-l6': {
        'model_id': MODEL_ID, 'revision': REVISION, 'backend': 'sentence_transformers',
        'pooling': 'sentence-transformer', 'prefix': '',
    },
    'minilm-l12': {
        'model_id': 'sentence-transformers/all-MiniLM-L12-v2',
        'revision': 'a50ef00143b4d5391434df20ae11632588ac25be',
        'backend': 'sentence_transformers', 'pooling': 'sentence-transformer', 'prefix': '',
    },
    'mpnet': {
        'model_id': 'sentence-transformers/all-mpnet-base-v2',
        'revision': 'e8c3b32edf5434bc2275fc9bab85f82640a19130',
        'backend': 'sentence_transformers', 'pooling': 'sentence-transformer', 'prefix': '',
    },
    'bge': {
        'model_id': 'BAAI/bge-base-en-v1.5',
        'revision': 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a',
        'backend': 'sentence_transformers', 'pooling': 'sentence-transformer', 'prefix': '',
    },
    'e5': {
        'model_id': 'intfloat/e5-base-v2',
        'revision': 'f52bf8ec8c7124536f0efb74aca902b2995e5bcd',
        'backend': 'sentence_transformers', 'pooling': 'sentence-transformer',
        'prefix': 'query: ',
    },
    'gte': {
        'model_id': 'thenlper/gte-base',
        'revision': 'c078288308d8dee004ab72c6191778064285ec0c',
        'backend': 'sentence_transformers', 'pooling': 'sentence-transformer', 'prefix': '',
    },
    'modernbert': {
        'model_id': 'answerdotai/ModernBERT-base',
        'revision': '8949b909ec900327062f0ebf497f51aef5e6f0c8',
        'backend': 'transformers', 'pooling': 'mean', 'prefix': '',
    },
}


def encoder_spec(name):
    try:
        return dict(ENCODER_REGISTRY[name])
    except KeyError as exc:
        raise ValueError(f"Unknown text encoder: {name}") from exc


def _embedding_dimension(model):
    for name in ('get_embedding_dimension', 'get_sentence_embedding_dimension'):
        method = getattr(model, name, None)
        if callable(method):
            return int(method())
    raise ValueError('Encoder does not expose an embedding dimension')


def _download_snapshot(spec):
    """Resolve a pinned snapshot into a real local directory.

    HuggingFace's shared cache creates symlinks, which fail on a standard
    Windows account without Developer Mode.  ``local_dir`` materializes
    regular files instead, so the same path works for training and for the
    persisted inference pipeline.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    root = Path(os.environ.get('SUPERAGENT_MODEL_CACHE',
                               Path.home() / '.cache' / 'superagent_models'))
    target = root / (spec['model_id'].replace('/', '--') + '-' + spec['revision'][:12])
    target.mkdir(parents=True, exist_ok=True)
    kwargs = dict(revision=spec['revision'], local_dir=str(target),
                  allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors', '*.bin'])
    try:
        snapshot_download(spec['model_id'], local_files_only=True, **kwargs)
    except LocalEntryNotFoundError:
        snapshot_download(spec['model_id'], local_files_only=False, **kwargs)
    return str(target)


class TextEncoderFeatures(BaseEstimator, TransformerMixin):
    """Frozen transformer features with an explicit backend and revision.

    Two text columns are expanded into A, B, A-B and A*B blocks.  The class
    keeps the same preprocessing contract as MiniLMFeatures so it can be
    dropped into the existing linear-probe pipeline without touching labels or
    the test split.  No head is fine-tuned here.
    """

    def __init__(self, columns, text=False, encoder='mpnet', device='auto',
                 batch_size=64, normalize=True, max_length=512):
        self.columns = columns
        self.text = text
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_length = max_length

    def _resolve_device(self):
        import torch
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        return 'cuda' if self.device != 'cpu' and torch.cuda.is_available() else 'cpu'

    def fit(self, X, y=None):
        spec = encoder_spec(self.encoder)
        path = _download_snapshot(spec)

        if spec['backend'] == 'sentence_transformers':
            from sentence_transformers import SentenceTransformer
            self.encoder_ = SentenceTransformer(path, device='cpu', trust_remote_code=False,
                                                local_files_only=True)
            self.tokenizer_ = None
            self.transformer_ = None
        else:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer_ = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                            trust_remote_code=False)
            self.transformer_ = AutoModel.from_pretrained(path, local_files_only=True,
                                                          trust_remote_code=False)
            self.encoder_ = None
        self.model_metadata_ = {
            'encoder': self.encoder,
            'model_id': spec['model_id'],
            'revision': spec['revision'],
            'backend': spec['backend'],
            'pooling': spec['pooling'],
            'purpose': 'frozen feature extraction',
            'fine_tuned': False,
        }
        self.resource_report_ = {
            'backend': spec['backend'], 'frozen': True, 'encoder': self.encoder,
            'model_id': spec['model_id'], 'revision': spec['revision'],
            'output_features': (_embedding_dimension(self.encoder_)
                                if self.encoder_ is not None else self.transformer_.config.hidden_size),
        }
        return self

    def _documents(self, X):
        frame = X[self.columns].fillna('').astype(str)
        if self.text:
            return [frame[c].tolist() for c in self.columns]
        return [frame.apply(lambda row: '; '.join(f'{c}: {row[c]}' for c in self.columns),
                            axis=1).tolist()]

    def _encode_transformer(self, documents, device):
        import torch
        import torch.nn.functional as functional

        model = self.transformer_.to(device).eval()
        tokenizer = self.tokenizer_
        parts = []
        try:
            with torch.inference_mode():
                for start in range(0, len(documents), self.batch_size):
                    batch = documents[start:start + self.batch_size]
                    tokens = tokenizer(batch, padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors='pt')
                    tokens = {key: value.to(device) for key, value in tokens.items()}
                    hidden = model(**tokens).last_hidden_state
                    mask = tokens.get('attention_mask')
                    if mask is None:
                        pooled = hidden.mean(dim=1)
                    else:
                        weights = mask.unsqueeze(-1).to(hidden.dtype)
                        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
                    if self.normalize:
                        pooled = functional.normalize(pooled, p=2, dim=1)
                    parts.append(pooled.float().cpu().numpy())
        finally:
            model.to('cpu')
            if device == 'cuda':
                torch.cuda.empty_cache()
        return np.concatenate(parts, axis=0).astype(np.float32)

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ['model_metadata_'])
        device = self._resolve_device()
        documents = self._documents(X)
        if self.model_metadata_['backend'] == 'transformers':
            parts = [self._encode_transformer(d, device) for d in documents]
        else:
            spec = encoder_spec(self.encoder)
            try:
                parts = [self.encoder_.encode([spec['prefix'] + value for value in d],
                             batch_size=self.batch_size, device=device,
                             normalize_embeddings=self.normalize, convert_to_numpy=True,
                             show_progress_bar=False)
                         for d in documents]
            finally:
                self.encoder_.to('cpu')
                if device == 'cuda':
                    import torch
                    torch.cuda.empty_cache()
        if len(parts) == 2:
            a, b = parts
            parts = [a, b, a - b, a * b]
        return np.concatenate(parts, axis=1).astype(np.float32)
