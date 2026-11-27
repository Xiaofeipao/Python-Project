"""Train-fitted frame schemas, datetime features and category frequencies."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrameSchema(BaseEstimator, TransformerMixin):
    def __init__(self, columns, numeric, datetime=()):
        self.columns = columns
        self.numeric = numeric
        self.datetime = datetime

    def __setstate__(self, state):
        # Models persisted before datetime-aware schemas have no attribute.
        state.setdefault("datetime", ())
        super().__setstate__(state)

    def fit(self, X, y=None):
        self.transform(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing prediction features: {sorted(missing)}")
        result = X[self.columns].copy()
        datetimes = set(self.datetime)
        for column in self.columns:
            if column in datetimes:
                result[column] = pd.to_datetime(result[column], errors="coerce", format="mixed")
            elif column in self.numeric:
                result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            else:
                result[column] = result[column].map(lambda v: np.nan if pd.isna(v) else str(v))
        return result


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Calendar components fitted with a train-only reference timestamp."""

    def __init__(self, columns):
        self.columns = columns

    def _parsed(self, X):
        return {column: pd.to_datetime(X[column], errors="coerce", format="mixed")
                for column in self.columns}

    def fit(self, X, y=None):
        parsed = self._parsed(X)
        finite = [series.dropna() for series in parsed.values() if series.notna().any()]
        self.reference_ = min((series.min() for series in finite), default=pd.Timestamp("1970-01-01"))
        return self

    def transform(self, X):
        parsed = self._parsed(X)
        blocks = []
        for column in self.columns:
            series = parsed[column]
            blocks.append(np.column_stack([
                series.dt.year, series.dt.month, series.dt.day, series.dt.weekday,
                series.dt.hour,
                (series - self.reference_).dt.total_seconds() / 86400.0,
                series.dt.weekday >= 5,
            ]).astype(np.float32))
        return np.hstack(blocks)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """High-cardinality fallback: smoothed frequency, unknown levels map to zero."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.maps_ = {}
        for column in self.columns:
            values = X[column].fillna("__missing__").astype(str)
            self.maps_[column] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        values = []
        for column in self.columns:
            series = X[column].fillna("__missing__").astype(str)
            values.append(series.map(self.maps_[column]).fillna(0.0).to_numpy(dtype=np.float32))
        return np.column_stack(values)
