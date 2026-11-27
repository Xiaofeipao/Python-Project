"""Deterministic image loading, resized pixels and histograms."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin


class ImageFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, column, root, size=24):
        self.column = column
        self.root = root
        self.size = size

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        vectors = []
        for value in X[self.column]:
            if pd.isna(value) or not str(value).strip():
                raise ValueError("Image path must not be missing")
            path = Path(self.root) / str(value)
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                pixels = np.asarray(rgb.resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            histograms = [np.histogram(pixels[:, :, c], bins=16, range=(0, 1), density=False)[0]
                          / (self.size * self.size) for c in range(3)]
            vectors.append(np.concatenate([pixels.ravel(), *histograms]).astype(np.float32))
        return np.asarray(vectors, dtype=np.float32)
