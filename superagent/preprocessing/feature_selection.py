"""Optional train-fitted feature filtering, including signed sparse features."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, mutual_info_classif
from functools import partial
from scipy import sparse


class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, method='none', percentile=50, seed=42):
        self.method=method
        self.percentile=percentile
        self.seed=seed

    def fit(self, X, y=None):
        if self.method == 'variance':
            self.selector_=VarianceThreshold()
        elif self.method == 'mutual_info':
            self.selector_=SelectPercentile(partial(mutual_info_classif, discrete_features=False,
                                                  random_state=self.seed), percentile=self.percentile)
        elif self.method == 'none':
            self.selector_=None
        else:
            raise ValueError('Unknown feature filter')
        if self.selector_ is not None:
            self.selector_.fit(self._input(X),y)
        return self

    def _input(self, X):
        if self.method == 'mutual_info' and sparse.issparse(X):
            if X.shape[0]*X.shape[1]*8 > 128*1024*1024:
                raise ValueError('Mutual information dense input exceeds 128 MiB bound; use variance filter')
            return X.toarray()
        return X

    def transform(self, X):
        return self.selector_.transform(self._input(X)) if self.selector_ is not None else X
