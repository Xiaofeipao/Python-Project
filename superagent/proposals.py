"""Bounded parameter proposals: data/model choices, never executable code."""
import math


RULES = {
    'logistic': {'C': (.001, 100.), 'analyzer': ['word', 'char', 'combined'], 'representation': ['difference', 'separate']},
    'linear_svm': {'C': (.001, 100.), 'analyzer': ['word', 'char', 'combined'], 'representation': ['difference', 'separate']},
    'text_blend': {'C': (.001, 100.), 'linear_weight': (.1, .9)},
    'catboost': {'iterations': (100, 1600), 'depth': (3, 9), 'learning_rate': (.01, .1), 'l2_leaf_reg': (1., 30.)},
    'xgboost': {'iterations': (100, 2000), 'depth': (3, 8), 'learning_rate': (.01, .1), 'reg_lambda': (1., 50.)},
    'cross_boost_blend': {'gpu_weight': (.05, .75), 'gpu_seeds': (1, 3), 'leaf_weight': (0., .2)},
}


def apply_proposal(base, overrides, profile):
    if not isinstance(overrides, dict):
        raise ValueError('Proposal must be a parameter object')
    rules = RULES.get(base['family'], {})
    result = dict(base)
    for key, value in overrides.items():
        if key not in rules:
            raise ValueError('Unapproved parameter')
        allowed = rules[key]
        if isinstance(allowed, list):
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
    if result.get('analyzer') == 'combined' and not result.get('representation'):
        raise ValueError('Combined features require ordered-text preprocessing')
    if result.get('gpu_weight', .25) + result.get('leaf_weight', 0.) > 1:
        raise ValueError('Invalid blend weights')
    return result
