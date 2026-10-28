"""Serializable train-only preprocessors, shared by training and prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ConstantFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return np.zeros((len(X), 1), dtype=np.float32)


class FrameSchema(BaseEstimator, TransformerMixin):
    def __init__(self, columns, numeric):
        self.columns = columns
        self.numeric = numeric

    def fit(self, X, y=None):
        self.transform(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing prediction features: {sorted(missing)}")
        result = X[self.columns].copy()
        for column in self.columns:
            if column in self.numeric:
                result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            else:
                result[column] = result[column].map(lambda v: np.nan if pd.isna(v) else str(v))
        return result


class JoinText(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return X[self.columns].fillna("").astype(str).agg(" ".join, axis=1).tolist()


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


def preprocessor(profile, candidate, image_root):
    if candidate["family"] == "dummy":
        return ConstantFeatures()
    modality = profile["data_type"]
    if modality == "text":
        character = candidate.get("analyzer", "word") == "char"
        return Pipeline([
            ("join", JoinText(profile["text_columns"])),
            ("tfidf", TfidfVectorizer(
                analyzer="char" if character else "word",
                ngram_range=(2, 4) if character else (1, 2),
                token_pattern=None if character else r"(?u)\b\w+\b",
                max_features=12000 if candidate["family"].startswith("torch_") else 30000,
                sublinear_tf=True, dtype=np.float32,
            )),
        ])
    if modality == "image":
        if candidate["family"] in {"torch_cnn", "torch_resnet", "torch_ensemble"}:
            return Pipeline([("images", ImageFeatures(profile["image_column"], image_root,
                                                       candidate.get("size", 32)))])
        return Pipeline([
            ("images", ImageFeatures(profile["image_column"], image_root, candidate.get("size", 24))),
            ("scale", StandardScaler()),
        ])
    numeric = profile["numeric_columns"]
    categorical = profile["categorical_columns"]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transforms.append(("category", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2, max_categories=128)),
        ]), categorical))
    return Pipeline([
        ("schema", FrameSchema(profile["features"], numeric)),
        ("columns", ColumnTransformer(transforms, sparse_threshold=1.0)),
    ])
