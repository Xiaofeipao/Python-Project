"""Portable general tabular estimators, independent of dataset identity."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from .features import FrameSchema


class NativeCategories(FrameSchema):
    def transform(self, X):
        frame = super().transform(X)
        for column in self.columns:
            if column not in self.numeric:
                frame[column] = frame[column].fillna("__missing__").astype(str).astype(object)
        return frame


class NativeBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3,
                 random_state=42, threads=1):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.threads = threads

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self.classes_, encoded = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        categorical = [c for c in X.columns if X[c].dtype == object]
        self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth,
                                        learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
                                        random_seed=self.random_state, thread_count=self.threads,
                                        cat_features=categorical, verbose=False, allow_writing_files=False)
        # No external validation/test rows enter target statistics or fitting.
        self.model_.fit(X, encoded)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class BalancedWeights:
    def fit(self, X, y, sample_weight=None):
        if self.class_weight != "balanced":
            return super().fit(X, y, sample_weight=sample_weight)
        # Avoid sklearn's string-label class_weight conversion regression.
        _, inverse, counts = np.unique(np.asarray(y, dtype=str), return_inverse=True, return_counts=True)
        weights = len(y) / (len(counts) * counts[inverse])
        if sample_weight is not None:
            weights *= np.asarray(sample_weight)
        self.class_weight = None
        try:
            return super().fit(X, y, sample_weight=weights)
        finally:
            self.class_weight = "balanced"


class BalancedExtraTrees(BalancedWeights, ExtraTreesClassifier):
    pass


class BalancedRandomForest(BalancedWeights, RandomForestClassifier):
    pass
