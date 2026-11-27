"""Bounded parameter proposals: data/model choices, never executable code."""
import math


RULES = {
    'logistic': {'C': (.001, 100.), 'analyzer': ['word', 'char', 'char_wb', 'combined'], 'representation': ['difference', 'separate'], 'fit_intercept':[True,False]},
    'linear_svm': {'C': (.001, 100.), 'analyzer': ['word', 'char', 'char_wb', 'combined'], 'representation': ['difference', 'separate']},
    'text_blend': {'C': (.001, 100.), 'linear_weight': (.1, .9)},
    'pairwise_overlap': {'C': (.001, 100.), 'fit_intercept': [True, False],
                         'class_weight': [None, 'balanced']},
    'pairwise_semantic_blend': {'C': (.001, 100.), 'semantic_weight': (.05, .25),
                                'encoder': ['mpnet', 'bge', 'e5', 'gte']},
    'pairwise_overlap_xgb': {'iterations': (100, 2000), 'depth': (3, 8),
                             'learning_rate': (.01, .1), 'reg_lambda': (1., 50.),
                             'balanced': [True, False]},
    'minilm_logistic': {'C': (.001, 100.)},
    'embedding_logistic': {'C': (.001, 100.), 'class_weight': [None, 'balanced']},
    'embedding_mlp': {'hidden': (32, 512), 'epochs': (5, 100),
                       'learning_rate': (.00001, .01), 'weight_decay': (.0, .1),
                       'dropout': (.0, .6)},
    'text_fasttext': {'embedding_dim': (32, 256), 'hidden': (32, 512),
                       'epochs': (1, 80), 'learning_rate': (.00001, .01),
                       'weight_decay': (.0, .1), 'dropout': (.0, .6),
                       'ngram_max': (1, 3)},
    'text_cnn': {'embedding_dim': (32, 256), 'hidden': (32, 512),
                  'epochs': (1, 80), 'learning_rate': (.00001, .01),
                  'weight_decay': (.0, .1), 'dropout': (.0, .6)},
    'text_rnn': {'embedding_dim': (32, 256), 'hidden': (32, 512),
                  'epochs': (1, 80), 'learning_rate': (.00001, .01),
                  'weight_decay': (.0, .1), 'dropout': (.0, .6)},
    'text_xgboost': {'iterations': (100, 2000), 'depth': (3, 8),
                      'learning_rate': (.01, .1), 'reg_lambda': (1., 50.),
                      'balanced': [True, False]},
    'transformer_finetune': {'epochs': (1, 8), 'learning_rate': (.000001, .0002),
                              'weight_decay': (.0, .1), 'freeze_layers': (0, 1)},
    'hybrid_logistic': {'C': (.001, 100.), 'fit_intercept': [True, False],
                        'class_weight': [None, 'balanced']},
    'hybrid_sgd': {'alpha': (.00001, .1)},
    'hybrid_xgboost': {'iterations': (100, 2000), 'depth': (3, 8), 'learning_rate': (.01, .1),
                       'reg_lambda': (1., 50.), 'analyzer': ['word', 'char', 'char_wb', 'combined'],
                       'balanced': [True, False], 'imputer': ['median', 'mean', 'most_frequent'],
                       'encoding': ['onehot', 'ordinal']},
    'torch_pretrained': {'epochs': (2, 80), 'learning_rate': (.00001, .01),
                         'weight_decay': (.0, .1), 'dropout': (.0, .6),
                         'size': [64, 96, 128, 160], 'tta': [True, False],
                         'pretrained': [True, False], 'augment': [True, False]},
    'image_meta_blend': {'image_weight': (.05, .95)},
    'frozen_image_logistic': {'C': (.001, 100.)},
    'catboost': {'iterations': (100, 1600), 'depth': (3, 9), 'learning_rate': (.01, .1),
                 'l2_leaf_reg': (1., 30.), 'balanced': [True, False]},
    'catboost_cross': {'iterations': (100, 3000), 'depth': (3, 10), 'learning_rate': (.01, .1),
                       'l2_leaf_reg': (1., 30.), 'balanced': [True, False],
                       'max_ctr_complexity': (1, 8), 'one_hot_max_size': (2, 128),
                       'random_strength': (.1, 5.), 'bagging_temperature': (.0, 2.),
                       'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                       'max_leaves': (2, 256)},
    'xgboost': {'iterations': (100, 2000), 'depth': (3, 8), 'learning_rate': (.01, .1),
                'reg_lambda': (1., 50.), 'balanced': [True, False]},
    'lightgbm': {'iterations': (100, 2000), 'num_leaves': (8, 255), 'learning_rate': (.005, .1),
                 'min_child_samples': (5, 80), 'reg_lambda': (0., 50.), 'balanced': [True, False]},
    'cross_boost_blend': {'gpu_weight': (.05, .75), 'gpu_seeds': (1, 3), 'leaf_weight': (0., .2)},
}
for _family in ('logistic', 'sgd', 'catboost', 'lightgbm', 'xgboost',
                'hybrid_logistic', 'hybrid_sgd', 'hybrid_xgboost', 'extra_trees', 'random_forest'):
    if _family in RULES:
        RULES[_family].update(decision_threshold=(.05, .95), calibration=[None, 'sigmoid'])
for _family in ('logistic', 'linear_svm'):
    RULES[_family].update(feature_filter=['none','variance','mutual_info'], percentile=(10,100),
                         imputer=['median','mean','most_frequent'],
                         encoding=['onehot','ordinal','frequency'],
                         pairwise_word_features=[False,True])
    RULES[_family]['vectorizer_options']={'ngram_range':'two integers 1..7', 'min_df':'integer 1..10',
        'max_features':'integer 1000..100000','use_idf':'boolean','binary':'boolean','stop_words':[None,'english']}


def apply_proposal(base, overrides, profile, trial_memory=None):
    if not isinstance(overrides, dict):
        raise ValueError('Proposal must be a parameter object')
    feature_keys = {'analyzer', 'representation', 'feature_filter', 'percentile','imputer','encoding','vectorizer_options','pairwise_word_features', 'augment', 'tta', 'pretrained', 'size'}
    changed = {k for k, v in overrides.items() if base.get(k) != v}
    if changed & feature_keys and changed - feature_keys:
        raise ValueError('Change features OR hyperparameters in one proposal, not both')
    rules = RULES.get(base['family'], {})
    result = dict(base)
    for key, value in overrides.items():
        if key == 'decision_threshold' and (profile.get('metric') not in {'f1_macro', 'f1_weighted', 'balanced_accuracy'}
                or len(profile.get('class_counts', {})) != 2):
            raise ValueError('Threshold tuning is restricted to binary F1/balanced-accuracy objectives')
        if key == 'calibration' and profile.get('metric') not in {'roc_auc', 'log_loss'}:
            raise ValueError('Calibration proposals require a probability-based objective')
        if key == 'calibration' and value and min(profile.get('fitting_class_counts', profile.get('class_counts', {})).values(), default=0) < 2:
            raise ValueError('Calibration needs at least two fitting rows per class')
        if key in {'fit_intercept','pairwise_word_features','balanced','tta','pretrained','augment'} and type(value) is not bool:
            raise ValueError('Boolean proposal setting requires a boolean value')
        if key in {'fit_intercept','pairwise_word_features'} and profile.get('data_type')!='text':
            raise ValueError('Text model settings require text input')
        if key == 'balanced' and not (profile.get('data_type') == 'tabular'
                                      or profile.get('tabular_columns')):
            raise ValueError('Balanced class weights require tabular input')
        if key in {'tta','pretrained','augment'} and profile.get('data_type')!='image':
            raise ValueError('Image model settings require image input')
        if key in {'imputer','encoding'} and not (profile.get('data_type') == 'tabular'
                                                  or profile.get('tabular_columns')):
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
    # Disabling calibration restores the original seed, not a distinct trial.
    if result.get('calibration') is None:
        result.pop('calibration', None)
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
