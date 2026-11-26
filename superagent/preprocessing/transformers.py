"""Serializable train-only preprocessors, shared by training and prediction."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class TfidfVectorizer(SklearnTfidfVectorizer):
    """Keep a zero-valued training vocabulary when text has no usable tokens."""
    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        documents = list(raw_documents)
        try:
            return super().fit_transform(documents, y)
        except ValueError as exc:
            if not any(reason in str(exc) for reason in ('empty vocabulary', 'After pruning', 'max_df corresponds')):
                raise
            self.vocabulary = {'__empty_training_vocabulary__': 0}
            self.min_df, self.max_df = 1, 1.0
            self.empty_vocabulary_fallback_ = True
            return super().fit_transform(documents, y)


def checked_vectorizer_options(options):
    if options is not None and not isinstance(options,dict):raise ValueError('Vectorizer options must be an object')
    options=dict(options or {})
    for key,value in options.items():
        if key in {'use_idf','binary'}:
            valid=type(value) is bool
        elif key=='min_df':
            valid=type(value) is int and 1<=value<=10
        elif key=='max_features':
            valid=type(value) is int and 1000<=value<=100000
        elif key=='stop_words':
            valid=value in (None,'english')
        elif key=='ngram_range':
            valid=isinstance(value,(list,tuple)) and len(value)==2 and all(type(v) is int for v in value) and 1<=value[0]<=value[1]<=7
        else:
            valid=False
        if not valid:raise ValueError('Unapproved vectorizer option')
    if 'ngram_range' in options:options['ngram_range']=tuple(options['ngram_range'])
    return options


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


class JoinText(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return X[self.columns].fillna("").astype(str).agg(" ".join, axis=1).tolist()


class PairwiseOverlapFeatures(BaseEstimator, TransformerMixin):
    """Interpretable word-level overlap statistics for two ordered texts.

    Unlike TF-IDF differences, these are dense, signed and standardized on the
    training split only, so the same values are available at inference time.
    """
    NEGATIONS = frozenset({"not", "no", "never", "n't", "without", "hardly",
                           "barely", "none", "nothing", "neither", "nor"})
    TOKEN = re.compile(r"[a-z0-9']+")

    def __init__(self, columns):
        self.columns = columns

    def _rows(self, X):
        rows = []
        for first, second in zip(X[self.columns[0]], X[self.columns[1]]):
            a, b = str(first), str(second)
            ta, tb = set(self.TOKEN.findall(a.lower())), set(self.TOKEN.findall(b.lower()))
            nums_a, nums_b = {t for t in ta if t.isdigit()}, {t for t in tb if t.isdigit()}
            names_a = {t for t in a.split() if t[:1].isupper()}
            names_b = {t for t in b.split() if t[:1].isupper()}
            neg_a, neg_b = bool(ta & self.NEGATIONS), bool(tb & self.NEGATIONS)
            length_a, length_b = len(a.split()), len(b.split())
            span = max(length_a, length_b, 1)
            rows.append([
                len(ta & tb) / (len(ta | tb) or 1),
                len(nums_a & nums_b) / (len(nums_a | nums_b) or 1),
                len(names_a & names_b) / (len(names_a | names_b) or 1),
                float(neg_a != neg_b),
                abs(length_a - length_b) / span,
                (length_a - length_b) / span,
                float(neg_a and neg_b),
            ])
        return np.asarray(rows, dtype=np.float32)

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(self._rows(X))
        return self

    def transform(self, X):
        return self.scaler_.transform(self._rows(X))


class MultiTextFeatures(BaseEstimator, TransformerMixin):
    """Shared training vocabulary, preserving column identity or signed differences."""
    def __init__(self, columns, analyzer="word", representation="difference", max_features=30000,
                 options=None, pairwise_word_features=False):
        self.columns = columns
        self.analyzer = analyzer
        self.representation = representation
        self.max_features = max_features
        self.options = options
        self.pairwise_word_features = pairwise_word_features

    def __setstate__(self, state):
        state.setdefault('options', None)
        state.setdefault('pairwise_word_features', False)
        super().__setstate__(state)

    def fit(self, X, y=None):
        if self.pairwise_word_features and (len(self.columns) != 2 or self.representation != 'difference'
                                           or self.analyzer not in {'char','char_wb'}):
            raise ValueError('Pairwise word features require two text columns, character features and signed differences')
        character = self.analyzer in {'char','char_wb'}
        self.vectorizer_ = TfidfVectorizer(analyzer=self.analyzer if character else 'word',
            ngram_range=(2, 5) if character else (1, 2), min_df=2,
            token_pattern=None if character else r'(?u)\b\w+\b',
            max_features=self.max_features, sublinear_tf=True, dtype=np.float32)
        if self.analyzer == 'combined':
            self.vectorizer_ = FeatureUnion([
                ('word', self.vectorizer_),
                ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2,
                                         max_features=self.max_features, sublinear_tf=True, dtype=np.float32))])
        if self.options:
            vectors = [v for _,v in self.vectorizer_.transformer_list] if self.analyzer=='combined' else [self.vectorizer_]
            for vector in vectors:
                vector.set_params(**checked_vectorizer_options(self.options))
        if self.pairwise_word_features:
            # One shared fitted union keeps character and word differences antisymmetric.
            self.vectorizer_ = FeatureUnion([
                ('character', self.vectorizer_),
                ('word', TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2,
                                       token_pattern=r'(?u)\b\w+\b', max_features=15000,
                                       sublinear_tf=True, dtype=np.float32)),
            ])
        self.vectorizer_.fit(pd.concat([X[c].fillna('').astype(str) for c in self.columns], ignore_index=True))
        return self

    def transform(self, X):
        texts = [X[c].fillna('').astype(str) for c in self.columns]
        vectors = [self.vectorizer_.transform(t) for t in texts]
        stats = [np.asarray([[np.log1p(len(s)), np.log1p(len(s.split())), s.count('?'), s.count('!'),
                              sum(ch.isdigit() for ch in s), sum(ch.isupper() for ch in s)] for s in t], dtype=np.float32)
                 for t in texts]
        if self.representation == 'difference':
            if len(vectors) != 2:
                raise ValueError('Signed text differences require two input columns')
            return sparse.hstack([vectors[0] - vectors[1], sparse.csr_matrix(stats[0] - stats[1])], format='csr')
        return sparse.hstack([*vectors, sparse.csr_matrix(np.hstack(stats))], format='csr')


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


def text_vectorizer(candidate):
    """Text vectorizer shared by hybrid models; capacity is candidate-controlled."""
    analyzer = candidate.get('analyzer', 'word')
    max_features = int(candidate.get('text_max_features', 20000))
    if analyzer == 'combined':
        return FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                     token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, dtype=np.float32)),
            ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2,
                                     max_features=max_features, sublinear_tf=True, dtype=np.float32)),
        ])
    character = analyzer in {'char', 'char_wb'}
    return TfidfVectorizer(analyzer=analyzer if character else 'word',
                           ngram_range=(2, 5) if character else (1, 2), min_df=2,
                           token_pattern=None if character else r"(?u)\b\w+\b",
                           max_features=max_features, sublinear_tf=True, dtype=np.float32)


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
