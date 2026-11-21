"""Frozen general-purpose encoder; no task-name dispatch or label fitting."""
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
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL_ID, revision=REVISION,
                                 allow_patterns=['*.json', '*.txt', 'model.safetensors'])
        self.encoder_ = SentenceTransformer(path, device='cpu', trust_remote_code=False,
                                            local_files_only=True)
        self.model_metadata_ = {'model_id': MODEL_ID, 'revision': REVISION,
                                'purpose': 'frozen feature extraction', 'fine_tuned': False}
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
