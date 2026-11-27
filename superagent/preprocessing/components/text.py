"""Train-fitted vocabularies, text columns and ordered-pair features."""

import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


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
