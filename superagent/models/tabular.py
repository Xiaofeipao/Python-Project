"""Portable general tabular estimators, independent of dataset identity."""

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from superagent.preprocessing.transformers import FrameSchema


class NativeCategories(FrameSchema):
    def __init__(self, columns, numeric, datetime=()):
        super().__init__(columns, numeric, datetime)

    def transform(self, X):
        frame = super().transform(X)
        for column in self.datetime:
            parsed = frame[column]
            parts = pd.DataFrame({
                f"{column}__year": parsed.dt.year.astype(float),
                f"{column}__month": parsed.dt.month.astype(float),
                f"{column}__day": parsed.dt.day.astype(float),
                f"{column}__weekday": parsed.dt.weekday.astype(float),
                f"{column}__hour": parsed.dt.hour.astype(float),
            }, index=frame.index)
            frame = pd.concat([frame.drop(columns=[column]), parts], axis=1)
        for column in frame.columns:
            if column not in self.numeric and pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = frame[column].astype(float)
            elif column not in self.numeric:
                frame[column] = frame[column].fillna("__missing__").astype(str).astype(object)
        return frame


class NativeBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3,
                 random_state=42, threads=1, balanced=False):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.threads = threads
        self.balanced = balanced

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self.classes_, encoded = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        categorical = [c for c in X.columns if X[c].dtype == object]
        self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth,
                                        learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
                                        random_seed=self.random_state, thread_count=self.threads,
                                        cat_features=categorical, verbose=False, allow_writing_files=False,
                                        auto_class_weights='Balanced' if self.balanced else None)
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


class NativeLeafBoost(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=800, num_leaves=31, learning_rate=0.035,
                 min_child_samples=30, reg_lambda=5, random_state=42, threads=1, balanced=False):
        self.iterations = iterations
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.threads = threads
        self.balanced = balanced

    def _frame(self, X):
        import pandas as pd

        frame = X.copy()
        for column, levels in self.category_levels_.items():
            frame[column] = pd.Categorical(frame[column], categories=levels)
        return frame

    def fit(self, X, y):
        from lightgbm import LGBMClassifier

        self.category_levels_ = {c: list(X[c].unique()) for c in X if X[c].dtype == object}
        self.classes_, encoded = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        self.model_ = LGBMClassifier(n_estimators=self.iterations, num_leaves=self.num_leaves,
                                     learning_rate=self.learning_rate, min_child_samples=self.min_child_samples,
                                     reg_lambda=self.reg_lambda, random_state=self.random_state,
                                     n_jobs=self.threads, verbosity=-1)
        weight = None
        if self.balanced:
            counts = np.bincount(encoded, minlength=len(self.classes_))
            weight = (len(encoded) / (len(counts) * counts))[encoded]
        self.model_.fit(self._frame(X), encoded, sample_weight=weight)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._frame(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class EncodedXGBoost(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=1600, depth=6, learning_rate=.03, reg_lambda=10,
                 random_state=42, threads=1, device='auto', balanced=False):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.threads = threads
        self.device = device
        self.balanced = balanced

    def fit(self, X, y):
        from xgboost import XGBClassifier
        device = self.device
        if device == 'auto':
            try:
                import torch
            except ImportError:
                device = 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes_, encoded = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        self.model_ = XGBClassifier(n_estimators=self.iterations, max_depth=self.depth,
            learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
            subsample=.85, colsample_bytree=.85, min_child_weight=3,
            random_state=self.random_state, n_jobs=self.threads, tree_method='hist', device=device)
        weight = None
        if self.balanced:
            counts = np.bincount(encoded, minlength=len(self.classes_))
            weight = (len(encoded) / (len(counts) * counts))[encoded]
        self.model_.fit(X, encoded, sample_weight=weight)
        self.resource_report_ = {'device': device, 'backend': 'xgboost', 'training_rows': X.shape[0]}
        # Use CPU inference so the saved pipeline does not require CUDA.
        self.model_.set_params(device='cpu')
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]



def fit_member(estimator, X, y):
    return clone(estimator).fit(X, y)


class ProbabilityBlend(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators, weights=None, cache_dir=None):
        self.estimators = estimators
        self.weights = weights
        self.cache_dir = cache_dir

    def fit(self, X, y):
        fit = Memory(self.cache_dir, verbose=0).cache(fit_member) if self.cache_dir else fit_member
        self.members_ = [fit(estimator, X, y) for estimator in self.estimators]
        self.classes_ = self.members_[0].classes_
        for member in self.members_:
            if not np.array_equal(member.classes_, self.classes_):
                raise ValueError("Ensemble members have inconsistent class order")
        self.n_features_in_ = X.shape[1]
        leaves = [m.named_steps['classifier'] if hasattr(m, 'named_steps') else m for m in self.members_]
        resources = [getattr(m, 'resource_report_', {}) for m in leaves]
        self.resource_report_ = {'device': 'cuda' if any(r.get('device') == 'cuda' for r in resources) else 'cpu',
                                 'members': resources}
        return self

    def predict_proba(self, X):
        return np.average([member.predict_proba(X) for member in self.members_], axis=0,
                          weights=getattr(self, 'weights', None))

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('weights', None)
        self.__dict__.setdefault('cache_dir', None)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def advanced_estimator(candidate, config):
    family = candidate['family']
    if family == 'lightgbm':
        return NativeLeafBoost(iterations=candidate['iterations'], num_leaves=candidate['num_leaves'],
                               learning_rate=candidate['learning_rate'],
                               min_child_samples=candidate['min_child_samples'], reg_lambda=candidate['reg_lambda'],
                               random_state=config.seed, threads=config.threads,
                               balanced=candidate.get('balanced', False))
    base = dict(iterations=candidate.get('iterations', 1200), depth=candidate.get('depth', 8),
                learning_rate=candidate.get('learning_rate', .05), l2_leaf_reg=candidate.get('l2_leaf_reg', 5),
                threads=config.threads)
    if family == 'catboost_ensemble':
        return ProbabilityBlend([NativeBoostClassifier(**base, balanced=candidate.get('balanced', False),
                                                       random_state=config.seed + 101 * index)
                                 for index in range(3)])
    return ProbabilityBlend([NativeBoostClassifier(**base, balanced=candidate.get('balanced', False),
                                                   random_state=config.seed),
                             NativeLeafBoost(random_state=config.seed, threads=config.threads,
                                             balanced=candidate.get('balanced', False))])
