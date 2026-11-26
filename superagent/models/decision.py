"""Metric-specific decisions; all calibration folds stay inside fitting data."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted


class DecisionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, threshold=None, calibration=None, positive_label=None):
        self.estimator = estimator
        self.threshold = threshold
        self.calibration = calibration
        self.positive_label = positive_label

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        if self.threshold is not None:
            if len(classes) != 2 or type(self.threshold) not in (int, float) or not .05 <= self.threshold <= .95:
                raise ValueError('Decision thresholds require two classes and a value in [0.05, 0.95]')
            if self.positive_label is not None and self.positive_label not in classes:
                raise ValueError('Positive label is absent from fitting data')
        self.model_ = clone(self.estimator)
        if self.calibration is not None:
            if self.calibration != 'sigmoid' or counts.min() < 2:
                raise ValueError('Sigmoid calibration needs at least two fitting rows in every class')
            # Wrapping the complete pipeline makes preprocessing train-only
            # inside every calibration fold, not just inside the outer trial.
            self.model_ = CalibratedClassifierCV(self.model_, method='sigmoid',
                                               cv=min(3, int(counts.min())), n_jobs=1)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, 'model_')
        if self.threshold is None:
            return self.model_.predict(X)
        positive = self.classes_[-1] if self.positive_label is None else self.positive_label
        index = list(self.classes_).index(positive)
        return np.where(self.predict_proba(X)[:, index] >= self.threshold,
                        self.classes_[index], self.classes_[1-index])
