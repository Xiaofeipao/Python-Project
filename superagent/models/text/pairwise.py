"""Ordered-text pair models kept after the 2026-11-25 mechanism comparison.

Only the winning mechanism lives here: a linear classifier over signed
character n-gram differences concatenated with word-level overlap statistics.
The rejected pairwise word-SVM blend and the fine-tuned cross-encoder were
removed rather than left as dormant branches.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from superagent.preprocessing.components.text import MultiTextFeatures, PairwiseOverlapFeatures

DEFAULT_VECTORIZER = {'ngram_range': [3, 6]}
DEFAULT_MAX_FEATURES = 30000


def overlap_classifier(columns, candidate, seed):
    """Build the selected pairwise model; preprocessing is fitted on train only."""
    features = FeatureUnion([
        ('tfidf', MultiTextFeatures(columns, 'char_wb', 'difference', DEFAULT_MAX_FEATURES,
                                    candidate.get('vectorizer_options') or DEFAULT_VECTORIZER)),
        ('overlap', PairwiseOverlapFeatures(columns)),
    ])
    return Pipeline([
        ('preprocess', features),
        ('classifier', LogisticRegression(C=candidate.get('C', 1.), solver='lbfgs', max_iter=1500,
                                          class_weight=candidate.get('class_weight'),
                                          fit_intercept=candidate.get('fit_intercept', True),
                                          random_state=seed)),
    ])


def overlap_xgb_classifier(columns, candidate, config):
    """Nonlinear head over the same signed sparse pairwise representation.

    XGBoost can exploit interactions among the dense overlap statistics and
    the selected character n-gram differences that a linear model cannot.
    """
    from superagent.models.tabular import EncodedXGBoost

    features = FeatureUnion([
        ('tfidf', MultiTextFeatures(columns, 'char_wb', 'difference', DEFAULT_MAX_FEATURES,
                                    candidate.get('vectorizer_options') or DEFAULT_VECTORIZER)),
        ('overlap', PairwiseOverlapFeatures(columns)),
    ])
    return Pipeline([
        ('preprocess', features),
        ('classifier', EncodedXGBoost(
            iterations=candidate.get('iterations', 800), depth=candidate.get('depth', 3),
            learning_rate=candidate.get('learning_rate', .05),
            reg_lambda=candidate.get('reg_lambda', 10), threads=config.threads,
            balanced=candidate.get('balanced', False), device=config.device,
            random_state=config.seed)),
    ])
