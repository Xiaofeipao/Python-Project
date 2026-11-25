"""Bounded parameter proposals: data/model choices, never executable code."""
import math


RULES = {
    'logistic': {'C': (.001, 100.), 'analyzer': ['word', 'char', 'char_wb', 'combined'], 'representation': ['difference', 'separate'], 'fit_intercept':[True,False]},
    'linear_svm': {'C': (.001, 100.), 'analyzer': ['word', 'char', 'char_wb', 'combined'], 'representation': ['difference', 'separate']},
    'text_blend': {'C': (.001, 100.), 'linear_weight': (.1, .9)},
    'pairwise_overlap': {'C': (.001, 100.), 'fit_intercept': [True, False],
                         'class_weight': [None, 'balanced']},
    'minilm_logistic': {'C': (.001, 100.)},
    'catboost': {'iterations': (100, 1600), 'depth': (3, 9), 'learning_rate': (.01, .1), 'l2_leaf_reg': (1., 30.)},
    'xgboost': {'iterations': (100, 2000), 'depth': (3, 8), 'learning_rate': (.01, .1), 'reg_lambda': (1., 50.)},
    'cross_boost_blend': {'gpu_weight': (.05, .75), 'gpu_seeds': (1, 3), 'leaf_weight': (0., .2)},
}
for _family in ('logistic', 'linear_svm'):
    RULES[_family].update(feature_filter=['none','variance','mutual_info'], percentile=(10,100),
                         imputer=['median','mean','most_frequent'], encoding=['onehot','ordinal'],
                         pairwise_word_features=[False,True])
    RULES[_family]['vectorizer_options']={'ngram_range':'two integers 1..7', 'min_df':'integer 1..10',
        'max_features':'integer 1000..100000','use_idf':'boolean','binary':'boolean','stop_words':[None,'english']}


def apply_proposal(base, overrides, profile, trial_memory=None):
    if not isinstance(overrides, dict):
        raise ValueError('Proposal must be a parameter object')
    feature_keys = {'analyzer', 'representation', 'feature_filter', 'percentile','imputer','encoding','vectorizer_options','pairwise_word_features'}
    changed = {k for k, v in overrides.items() if base.get(k) != v}
    if changed & feature_keys and changed - feature_keys:
        raise ValueError('Change features OR hyperparameters in one proposal, not both')
    rules = RULES.get(base['family'], {})
    result = dict(base)
    for key, value in overrides.items():
        if key in {'fit_intercept','pairwise_word_features'} and type(value) is not bool:
            raise ValueError('Boolean proposal setting requires a boolean value')
        if key in {'fit_intercept','pairwise_word_features'} and profile.get('data_type')!='text':
            raise ValueError('Text model settings require text input')
        if key in {'imputer','encoding'} and profile.get('data_type')!='tabular':
            raise ValueError('Tabular feature setting on non-tabular input')
        if key=='vectorizer_options' and profile.get('data_type')!='text':
            raise ValueError('Vectorizer settings require text input')
        if key not in rules:
            raise ValueError('Unapproved parameter')
        allowed = rules[key]
        if key=='vectorizer_options':
            from superagent.preprocessing.transformers import checked_vectorizer_options
            value=checked_vectorizer_options(value)
        elif isinstance(allowed, list):
            if value not in allowed or (key == 'representation' and not base.get('representation')):
                raise ValueError('Unsupported representation change')
        else:
            if type(value) not in (int, float) or not math.isfinite(value) or not allowed[0] <= value <= allowed[1]:
                raise ValueError('Parameter exceeds resource envelope')
            if type(allowed[0]) is int and type(value) is not int:
                raise ValueError('Integer parameter required')
        result[key] = value
    if result.get('representation') == 'difference' and len(profile.get('text_columns') or []) != 2:
        raise ValueError('Difference representation requires two columns')
    if result.get('pairwise_word_features') and (profile.get('data_type')!='text'
            or len(profile.get('text_columns') or [])!=2 or result.get('representation')!='difference'
            or result.get('analyzer') not in {'char','char_wb'}):
        raise ValueError('Pairwise word features require two text columns, character features and signed differences')
    if result.get('analyzer') == 'combined' and not result.get('representation'):
        raise ValueError('Combined features require ordered-text preprocessing')
    if result.get('gpu_weight', .25) + result.get('leaf_weight', 0.) > 1:
        raise ValueError('Invalid blend weights')
    if 'percentile' in overrides and result.get('feature_filter')!='mutual_info':
        raise ValueError('Percentile requires mutual-information selection')
    if trial_memory is not None:
        from superagent.planning.memory import trial_key
        if trial_key(result) in trial_memory:
            raise ValueError('Duplicate family and parameter combination')
    return result
