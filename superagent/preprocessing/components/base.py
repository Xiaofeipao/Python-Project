"""Constant features and bounded dimensionality reduction."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class ConstantFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return np.zeros((len(X), 1), dtype=np.float32)


class BoundedSVD(BaseEstimator, TransformerMixin):
    """Fit a training-only projection, including on small input matrices."""
    def __init__(self, n_components=128):
        self.n_components = n_components

    def fit(self, X, y=None):
        count = min(self.n_components, X.shape[0], X.shape[1])
        self.model_ = TruncatedSVD(n_components=count, random_state=42) if X.shape[1] > 1 else None
        if self.model_ is not None:
            self.model_.fit(X)
        return self

    def transform(self, X):
        return self.model_.transform(X) if self.model_ is not None else X.toarray()
