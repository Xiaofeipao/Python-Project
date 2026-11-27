"""Compose feature components into tabular, text, image and mixed pipelines."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from .components.base import BoundedSVD, ConstantFeatures
from .components.tabular import DatetimeFeatures, FrameSchema, FrequencyEncoder
from .components.text import JoinText, MultiTextFeatures, TfidfVectorizer, checked_vectorizer_options, text_vectorizer
from .components.image import ImageFeatures


def preprocessor(profile, candidate, image_root):
    if candidate["family"] == "dummy":
        return ConstantFeatures()
    modality = profile["data_type"]
    if candidate.get('pairwise_word_features') and (modality!='text'
            or len(profile.get('text_columns') or [])!=2 or candidate.get('representation')!='difference'
            or candidate.get('analyzer') not in {'char','char_wb'}):
        raise ValueError('Pairwise word features require two text columns, character features and signed differences')
    if modality == "text":
        if candidate.get('representation'):
            features = MultiTextFeatures(profile['text_columns'], candidate.get('analyzer', 'word'),
                                     candidate['representation'],
                                     10000 if candidate['family'].startswith('torch_') else 30000,
                                     candidate.get('vectorizer_options'),
                                     candidate.get('pairwise_word_features',False))
            if candidate.get('svd_components'):
                return Pipeline([('text', features), ('svd', BoundedSVD(candidate['svd_components'])),
                                 ('scale', StandardScaler())])
            return features
        character = candidate.get("analyzer", "word") in {'char','char_wb'}
        return Pipeline([
            ("join", JoinText(profile["text_columns"])),
            ("tfidf", TfidfVectorizer(
                analyzer=candidate.get('analyzer','word') if character else 'word',
                ngram_range=(2, 4) if character else (1, 2),
                token_pattern=None if character else r"(?u)\b\w+\b",
                max_features=12000 if candidate["family"].startswith("torch_") else 30000,
                sublinear_tf=True, dtype=np.float32,
            ).set_params(**checked_vectorizer_options(candidate.get('vectorizer_options')))),
        ])
    if modality == "image":
        if candidate["family"] in {"torch_cnn", "torch_resnet", "torch_ensemble", "torch_pretrained"}:
            return Pipeline([("images", ImageFeatures(profile["image_column"], image_root,
                                                       candidate.get("size", 32)))])
        return Pipeline([
            ("images", ImageFeatures(profile["image_column"], image_root, candidate.get("size", 24))),
            ("scale", StandardScaler()),
        ])
    return tabular_preprocessor(profile, candidate)


def tabular_preprocessor(profile, candidate, columns=None, numeric=None,
                         categorical=None, datetimes=None):
    columns = list(columns if columns is not None else profile["features"])
    numeric = list(numeric if numeric is not None else profile["numeric_columns"])
    categorical = list(categorical if categorical is not None else profile["categorical_columns"])
    datetimes = list(datetimes if datetimes is not None else profile.get("datetime_columns", []))
    cardinality = profile.get("categorical_cardinality", {})
    # Keep the established one-hot/ordinal path as the default. Frequency
    # encoding remains available as an explicit high-cardinality fallback,
    # but it must not silently replace the representation that produced the
    # verified tabular results.
    high = [c for c in categorical if cardinality.get(c, 0) > 128]
    frequency_mode = candidate.get('encoding') == 'frequency'
    frequency = high if frequency_mode else []
    low = [c for c in categorical if c not in frequency]
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy=candidate.get('imputer', 'median'), keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if datetimes:
        transforms.append(("datetime", Pipeline([
            ("parts", DatetimeFeatures(datetimes)),
            ("impute", SimpleImputer(strategy='median', keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), datetimes))
    if low:
        ordinal = candidate.get('encoding') == 'ordinal'
        transforms.append(("category", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
            ("encode", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) if ordinal
             else OneHotEncoder(handle_unknown="ignore", min_frequency=2, max_categories=128)),
        ]), low))
    if frequency:
        transforms.append(("frequency", Pipeline([
            ("frequency", FrequencyEncoder(frequency)),
            ("impute", SimpleImputer(strategy='median', keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), high))
    return Pipeline([
        ("schema", FrameSchema(columns, numeric, datetimes)),
        ("columns", ColumnTransformer(transforms, sparse_threshold=1.0)),
    ])


def hybrid_preprocessor(profile, candidate):
    """Text TF-IDF and tabular features share one estimator without dropping columns."""
    text_columns = profile.get("text_columns") or []
    tabular_columns = profile.get("tabular_columns") or []
    if not text_columns:
        raise ValueError("Hybrid models require at least one text column")
    text = Pipeline([("join", JoinText(text_columns)), ("tfidf", text_vectorizer(candidate))])
    if not tabular_columns:
        return Pipeline([("text", text)])
    tabular = tabular_preprocessor(dict(profile, features=tabular_columns), candidate)
    return FeatureUnion([("text", text), ("tabular", tabular)])
