"""Generic candidate catalog and feedback-guided local search."""

import importlib.util
from superagent.evaluation.metrics import utility
from .allocation import candidate_key, metric_candidates, allocate_candidates


def next_candidate(profile, history, metric):
    tried = {candidate_key(h["candidate"]) for h in history}
    counts = list(profile.get("class_counts", {}).values())
    imbalanced = bool(counts) and max(counts) / max(1, min(counts)) >= 2
    if not history:
        return {"family": "dummy"}, "Establish a class-prior baseline on the fixed validation split"
    if profile["data_type"] == "text":
        initial = [
            {"family": "logistic", "C": 1.0, "analyzer": "word"},
            {"family": "logistic", "C": 1.0, "analyzer": "char"},
            {"family": "nb", "alpha": 1.0, "analyzer": "word"},
        ]
    else:
        initial = [
            {"family": "logistic", "C": 1.0},
            {"family": "extra_trees", "n_estimators": 150, "max_depth": 16, "min_samples_leaf": 1},
            {"family": "random_forest", "n_estimators": 150, "max_depth": 16, "min_samples_leaf": 1},
        ]
    if profile.get("torch_available") and profile["data_type"] in {"tabular", "image"}:
        neural = {"family": "torch_cnn" if profile["data_type"] == "image" else "torch_mlp",
                  "hidden": 128, "epochs": 30 if profile["data_type"] == "image" else 50,
                  "learning_rate": 0.001, "weight_decay": 0.001}
        if profile["data_type"] == "image":
            neural["size"] = 32
        initial.insert(1, neural)
    large_gpu = profile.get("torch_available") and profile.get("compute_profile") == "gpu_large"
    if large_gpu:
        is_image = profile["data_type"] == "image"
        strong = {"family": "torch_resnet" if is_image else "torch_mlp", "hidden": 256,
                  "epochs": 100 if is_image else 120, "learning_rate": 0.001,
                  "weight_decay": 0.01, "early_stopping": True, "dropout": 0.15,
                  "cosine_schedule": True,
                  "architecture": "resnet" if is_image else "residual_mlp"}
        if is_image:
            strong.update(size=32, width=32, augment=True, normalize_images=True)
        elif profile["data_type"] == "text":
            strong["analyzer"] = "char"
        wider = dict(strong, width=48, hidden=384, learning_rate=0.0005) if is_image else dict(strong, hidden=384, depth=4, dropout=0.25)
        ensemble = dict(strong, family="torch_ensemble", ensemble_size=3)
        # Explore multiple neural configurations even when the first CPU trial
        # wins; a single small network is not a sufficient GPU comparison.
        if is_image and importlib.util.find_spec('torchvision') is not None:
            pretrained = [
                {'family': 'torch_pretrained', 'architecture': 'pretrained_resnet18',
                 'size': size, 'epochs': 18, 'learning_rate': 3e-4, 'weight_decay': 1e-4,
                 'dropout': 0.2, 'early_stopping': True, 'cosine_schedule': True,
                 'augment': True, 'imagenet_normalize': True, 'tta': True}
                for size in (96, 128)
            ] + [
                {'family': 'torch_pretrained', 'architecture': 'pretrained_resnet34',
                 'size': 128, 'epochs': 14, 'learning_rate': 2e-4, 'weight_decay': 1e-4,
                 'dropout': 0.2, 'early_stopping': True, 'cosine_schedule': True,
                 'augment': True, 'imagenet_normalize': True, 'tta': True}
            ]
            pretrained.append(dict(pretrained[0], augment=False, tta=False))
            initial[1:1] = pretrained + [strong, wider, ensemble]
        else:
            initial[1:1] = [strong, wider, ensemble]
    if (profile['data_type'] == 'image' and profile.get('torch_available')
            and importlib.util.find_spec('torchvision') is not None):
        initial.insert(1, {'family': 'frozen_image_logistic', 'C': 1., 'size': 96})
    if (profile['data_type'] == 'image' and large_gpu and profile.get('tabular_columns')
            and importlib.util.find_spec('torchvision') is not None):
        initial[0:0] = [
            {'family': 'image_meta_blend', 'image_weight': weight, 'size': 96,
             'epochs': 18, 'learning_rate': 3e-4, 'architecture': 'pretrained_resnet18'}
            for weight in (.7, .5)
        ]
    if profile['data_type'] == 'tabular' and importlib.util.find_spec('catboost') is not None:
        initial[0:0] = [
            {'family': 'catboost', 'iterations': 600, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
            {'family': 'catboost', 'iterations': 900, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 5},
        ]
    if profile['data_type'] == 'tabular':
        extra = []
        if importlib.util.find_spec('catboost') is not None:
            extra += [
                {'family': 'catboost', 'iterations': 1200, 'depth': 8, 'learning_rate': .05, 'l2_leaf_reg': 5},
                {'family': 'catboost', 'iterations': 1800, 'depth': 10, 'learning_rate': .035, 'l2_leaf_reg': 10},
                {'family': 'catboost_ensemble', 'iterations': 1200, 'depth': 8, 'learning_rate': .05, 'l2_leaf_reg': 5},
            ]
        if importlib.util.find_spec('lightgbm') is not None:
            extra[0:0] = [
                {'family': 'lightgbm', 'iterations': 800, 'num_leaves': 31, 'learning_rate': .035,
                 'min_child_samples': 30, 'reg_lambda': 5},
                {'family': 'lightgbm', 'iterations': 1200, 'num_leaves': 63, 'learning_rate': .025,
                 'min_child_samples': 40, 'reg_lambda': 10},
            ]
            if importlib.util.find_spec('catboost') is not None:
                extra += [{'family': 'tabular_blend'}]
        initial[2:2] = extra
        if importlib.util.find_spec('catboost') is not None:
            initial.append({'family': 'catboost_cross', 'iterations': 1800, 'depth': 8,
                            'learning_rate': .04, 'l2_leaf_reg': 10,
                            'max_ctr_complexity': 6, 'one_hot_max_size': 32,
                            'random_strength': 2., 'bagging_temperature': 1.,
                            'grow_policy': 'Lossguide', 'max_leaves': 64})
    if profile['data_type'] == 'text' and len(profile.get('text_columns') or []) > 1:
        ordered = [{'family': 'logistic', 'C': c, 'analyzer': analyzer, 'representation': mode}
                   for mode in ('difference', 'separate') for analyzer in ('word', 'char', 'combined') for c in (.1, 1., 10.)
                   if mode != 'difference' or len(profile['text_columns']) == 2]
        if large_gpu:
            ordered += [{'family': 'torch_mlp', 'architecture': 'mlp', 'hidden': 128, 'epochs': 60,
                         'learning_rate': .001, 'weight_decay': .01, 'early_stopping': True,
                         'analyzer': a, 'representation': 'separate'} for a in ('word', 'char', 'combined')]
        initial[0:0] = ordered
        initial[0:0] = [
            {'family': 'logistic', 'C': 1., 'analyzer': 'char', 'representation': 'separate',
             'class_weight': 'balanced'},
            *[{'family': 'linear_svm', 'C': c, 'analyzer': analyzer, 'representation': mode}
              for mode in ('difference', 'separate') for analyzer in ('char', 'combined') for c in (.1, 1., 10.)
              if mode != 'difference' or len(profile['text_columns']) == 2],
            *[{'family': 'rbf_svm', 'C': c, 'analyzer': 'combined', 'representation': 'separate',
               'svd_components': 128} for c in (1., 10.)],
        ]
        if large_gpu and importlib.util.find_spec('xgboost') is not None:
            initial[1:1] = [
                {'family': 'xgboost', 'iterations': 1000, 'depth': depth, 'learning_rate': .03,
                 'reg_lambda': 10, 'analyzer': 'combined', 'representation': 'separate',
                 'svd_components': 128} for depth in (3, 6)]
    if profile['data_type'] == 'text' and len(profile.get('text_columns') or []) > 1:
        blends = [{'family': 'text_blend', 'linear_weight': .5, 'C': .1}]
        if len(profile['text_columns']) == 2:
            # Keep the original separate-column blend untouched; this new
            # family fuses signed char_wb and word differences for pairs.
            blends.insert(0, {'family': 'text_blend_diff', 'linear_weight': .5, 'C': .1})
        initial[0:0] = blends
    if profile['data_type'] == 'text' and len(profile.get('text_columns') or []) == 2:
        pairwise = {'family':'logistic','C':1.,'analyzer':'char_wb',
                    'representation':'difference','fit_intercept':False,
                    'vectorizer_options':{'ngram_range':[3,6]}}
        overlap = {'family':'pairwise_overlap','C':1.,'fit_intercept':True,
                   'vectorizer_options':{'ngram_range':[3,6]}}
        # Pairwise tasks get the signed overlap anchor before single-text
        # logistic variants; this remains modality/shape-based, not task-specific.
        initial[0:0] = [overlap, pairwise, dict(pairwise,pairwise_word_features=True)]
        if large_gpu and importlib.util.find_spec('xgboost') is not None:
            initial.append({'family': 'pairwise_overlap_xgb', 'iterations': 800, 'depth': 3,
                            'learning_rate': .05, 'reg_lambda': 10})
        if profile.get('text_encoder_available'):
            initial.append({'family': 'pairwise_semantic_blend', 'semantic_weight': .1,
                            'C': 1., 'encoder': 'mpnet', 'batch_size':
                            profile.get('neural_batch_size', 64), 'max_length': 256})
            # Pairwise sparse signal, a small MPNet logistic correction and a
            # stronger BGE MLP head.  The weights come from validation-only
            # blending of the strongest observed components, not task identity.
            fused = {'family': 'pairwise_semantic_mlp_blend', 'semantic_weight': .055,
                     'mlp_weight': .35, 'C': 1., 'encoder_semantic': 'mpnet',
                     'encoder_mlp': 'bge', 'hidden': 384, 'epochs': 40,
                     'learning_rate': .0007, 'weight_decay': .03, 'dropout': .3,
                     'batch_size': profile.get('neural_batch_size', 64), 'max_length': 256}
            initial.insert(0, fused)
            initial.insert(1, dict(fused, mlp_weight=.3))
    if profile['data_type'] == 'text' and profile.get('torch_available'):
        # Trainable text stacks are independent of the TF-IDF vocabulary.  Keep
        # them after the established sparse anchors so a large candidate list
        # cannot displace the verified sparse anchors.
        text_neural = [
            {'family': 'text_fasttext', 'embedding_dim': 128, 'hidden': 192,
             'epochs': 25, 'learning_rate': .002, 'weight_decay': .01,
             'batch_size': profile.get('neural_batch_size', 64), 'ngram_max': 2,
             'dropout': .2, 'max_vocab': 30000},
        ]
        if large_gpu:
            text_neural += [
                {'family': 'text_cnn', 'embedding_dim': 128, 'hidden': 128,
                 'epochs': 24, 'learning_rate': .0015, 'weight_decay': .01,
                 'batch_size': profile.get('neural_batch_size', 64), 'max_length': 256,
                 'dropout': .2},
                {'family': 'text_rnn', 'embedding_dim': 128, 'hidden': 128,
                 'epochs': 24, 'learning_rate': .0015, 'weight_decay': .01,
                 'batch_size': profile.get('neural_batch_size', 64), 'max_length': 256,
                 'dropout': .2},
            ]
        initial.extend(text_neural)
        if large_gpu and importlib.util.find_spec('xgboost') is not None:
            # Sparse or compressed text features with a boosted-tree head.
            initial.extend([
                {'family': 'text_xgboost', 'iterations': 1000, 'depth': 3,
                 'learning_rate': .03, 'reg_lambda': 10, 'analyzer': 'combined',
                 'representation': 'separate'},
                {'family': 'text_xgboost', 'iterations': 1600, 'depth': 4,
                 'learning_rate': .025, 'reg_lambda': 20, 'analyzer': 'combined',
                 'representation': 'separate', 'svd_components': 256},
            ])
    if profile['data_type'] == 'text' and profile.get('text_encoder_available'):
        # Frozen transformer probes: MPNet, BGE, E5 and GTE, plus ModernBERT.
        # Each candidate fixes an encoder/revision through the registry in
        # models.text_encoder; no task-specific branch is introduced here.
        initial.extend([
            {'family': 'embedding_logistic', 'encoder': encoder, 'C': C,
             'batch_size': profile.get('neural_batch_size', 64), 'max_length': 256}
            for encoder in ('mpnet', 'bge', 'e5', 'gte')
            for C in (1., .1)
        ])
        initial.append({'family': 'embedding_logistic', 'encoder': 'modernbert',
                        'C': 1., 'batch_size': 32, 'max_length': 256})
        if large_gpu:
            initial.extend([
                {'family': 'embedding_mlp', 'encoder': encoder, 'hidden': 256,
                 'epochs': 30, 'learning_rate': .001, 'weight_decay': .01,
                 'dropout': .2, 'batch_size': 64, 'max_length': 256}
                for encoder in ('mpnet', 'bge')
            ])
            initial.extend([
                {'family': 'transformer_finetune', 'encoder': encoder, 'epochs': 3,
                 'learning_rate': 2e-5, 'weight_decay': .01, 'batch_size': 16,
                 'max_length': 256, 'freeze_layers': 1}
                for encoder in ('minilm-l12', 'mpnet')
            ])
    if profile['data_type'] == 'text' and profile.get('tabular_columns'):
        hybrid = [{'family': 'hybrid_logistic', 'C': 1., 'analyzer': 'combined'},
                  {'family': 'hybrid_logistic', 'C': .1, 'analyzer': 'word'}]
        if imbalanced:
            hybrid.insert(0, {'family': 'hybrid_logistic', 'C': 1., 'analyzer': 'combined',
                              'class_weight': 'balanced'})
        hybrid.append({'family': 'hybrid_sgd', 'alpha': 1e-4, 'analyzer': 'combined'})
        if importlib.util.find_spec('xgboost') is not None:
            hybrid += [{'family': 'hybrid_xgboost', 'iterations': 800, 'depth': 6,
                        'learning_rate': .05, 'reg_lambda': 10, 'analyzer': 'combined',
                        'text_max_features': 6000},
                       {'family': 'hybrid_xgboost', 'iterations': 1600, 'depth': 4,
                        'learning_rate': .03, 'reg_lambda': 20, 'analyzer': 'combined',
                        'text_max_features': 6000}]
            if imbalanced:
                hybrid.insert(0, {'family': 'hybrid_xgboost', 'iterations': 1000, 'depth': 5,
                                  'learning_rate': .04, 'reg_lambda': 10, 'analyzer': 'combined',
                                  'text_max_features': 6000, 'balanced': True})
        initial[0:0] = hybrid
    if profile['data_type'] == 'tabular' and large_gpu and importlib.util.find_spec('xgboost') is not None:
        initial[0:0] = [
            {'family': 'cross_boost_blend', 'gpu_seeds': 3, 'gpu_weight': .275, 'leaf_weight': 0.},
            {'family': 'cross_boost_blend'},
            {'family': 'cross_boost_blend', 'gpu_weight': .1},
            {'family': 'cross_boost_blend', 'gpu_weight': .5},
            {'family': 'cross_boost_blend', 'gpu_weight': .75},
            {'family': 'cross_boost_blend', 'gpu_seeds': 3, 'gpu_weight': .25, 'leaf_weight': 0.},
            {'family': 'cross_boost_blend', 'gpu_seeds': 3, 'gpu_weight': .5, 'leaf_weight': 0.},
            {'family': 'cross_boost_blend', 'gpu_seeds': 3, 'gpu_weight': .25, 'leaf_weight': .15},
            {'family': 'cross_boost_blend', 'gpu_seeds': 4, 'gpu_weight': .3, 'leaf_weight': 0.},
        ] + [
            {'family': 'xgboost', 'iterations': count, 'depth': depth, 'learning_rate': .03,
             'reg_lambda': regularization} for count, depth, regularization in
            ((2000, 4, 10), (1600, 6, 10), (1200, 8, 20))]
    if profile.get('minilm_available') and profile['data_type'] in {'tabular', 'text'}:
        from superagent.models.text.encoders import MODEL_ID, REVISION
        initial[0:0] = [{'family': 'minilm_logistic', 'C': c, 'encoder': MODEL_ID,
                         'revision': REVISION} for c in (.1, 1., 10.)]
    if profile.get('feature_search_enabled') and profile['data_type'] in {'text','tabular'}:
        base={'family':'logistic','C':.1}
        if profile['data_type']=='text':
            base.update(analyzer='combined' if len(profile.get('text_columns') or [])>1 else 'word')
            if len(profile.get('text_columns') or [])>1:
                base['representation']='separate'
        initial[0:0]=[dict(base,feature_filter='variance'),
                       dict(base,feature_filter='mutual_info',percentile=50,
                            **({'svd_components':64} if profile['data_type']=='text' and base.get('representation') else {})),
                       base]
    if imbalanced and profile['data_type'] == 'tabular':
        balanced = []
        if importlib.util.find_spec('catboost') is not None:
            balanced.append({'family': 'catboost', 'iterations': 1200, 'depth': 8,
                             'learning_rate': .05, 'l2_leaf_reg': 5, 'balanced': True})
        if importlib.util.find_spec('xgboost') is not None:
            balanced.append({'family': 'xgboost', 'iterations': 1600, 'depth': 6,
                             'learning_rate': .03, 'reg_lambda': 10, 'balanced': True})
        initial[0:0] = balanced
    initial = metric_candidates(profile, history, metric) + initial
    for candidate in allocate_candidates(initial, profile, history):
        if candidate_key(candidate) not in tried:
            if candidate.get('calibration') or candidate.get('decision_threshold') is not None:
                observed = [h['score'] for h in history if h['status'] == 'success'
                            and h['candidate']['family'] == candidate['family']]
                best_score = max(observed, key=lambda score: utility(score, metric))
                return candidate, f"Compare a metric-specific {candidate['family']} decision using observed {metric}={best_score:.6g}"
            return candidate, f"Compare a general {candidate['family']} method for {profile['data_type']} data"
    successful = [h for h in history if h["status"] == "success" and h["candidate"]["family"] != "dummy"]
    successful.sort(key=lambda h: utility(h["score"], metric), reverse=True)
    if large_gpu and len(history) % 2 == 0:
        neural_success = [h for h in successful if h["candidate"]["family"].startswith("torch_")]
        if neural_success:
            preferred = neural_success[0]
            successful = [preferred] + [h for h in successful if h is not preferred]
    for best in successful:
        candidate = best["candidate"]
        family = candidate["family"]
        counts = list(profile.get("class_counts", {}).values())
        imbalanced = bool(counts) and max(counts) / min(counts) >= 2
        if family == "catboost":
            variants = [dict(candidate, iterations=1200), dict(candidate, depth=5),
                        dict(candidate, learning_rate=0.025), dict(candidate, l2_leaf_reg=10)]
            if imbalanced:
                variants.insert(0, dict(candidate, balanced=True))
        elif family == 'lightgbm':
            variants = [dict(candidate, num_leaves=value) for value in (15, 63, 127)]
            variants += [dict(candidate, iterations=value) for value in (400, 1600)]
            if imbalanced:
                variants.insert(0, dict(candidate, balanced=True))
        elif family in {'catboost_ensemble', 'tabular_blend', 'minilm_logistic'}:
            variants = []
        elif family == 'embedding_logistic':
            variants = [dict(candidate, C=value) for value in (.03, .3, 3.)]
            variants += [dict(candidate, class_weight='balanced')]
        elif family == 'embedding_mlp':
            variants = [dict(candidate, hidden=value) for value in (128, 384)]
            variants += [dict(candidate, learning_rate=value) for value in (.0003, .003)]
            variants += [dict(candidate, epochs=candidate['epochs'] + 15)]
        elif family in {'text_fasttext', 'text_cnn', 'text_rnn'}:
            variants = [dict(candidate, learning_rate=value) for value in (.0005, .003)]
            variants += [dict(candidate, epochs=candidate['epochs'] + 12)]
            variants += [dict(candidate, dropout=value) for value in (.1, .35)]
        elif family == 'text_xgboost':
            variants = [dict(candidate, depth=value) for value in (5, 7)]
            variants += [dict(candidate, iterations=value) for value in (600, 1800)]
            variants += [dict(candidate, reg_lambda=value) for value in (1., 40.)]
        elif family == 'transformer_finetune':
            variants = [dict(candidate, learning_rate=value) for value in (1e-5, 5e-5)]
            variants += [dict(candidate, epochs=candidate['epochs'] + 2)]
        elif family == 'xgboost':
            variants = [dict(candidate, reg_lambda=value) for value in (1, 20, 50)]
            if imbalanced:
                variants.insert(0, dict(candidate, balanced=True))
        elif family == 'image_meta_blend':
            variants = [dict(candidate, image_weight=value) for value in (.3, .9)]
        elif family == 'frozen_image_logistic':
            variants = [dict(candidate, C=c) for c in (.1, 10.)]
        elif family == 'torch_pretrained':
            variants = [dict(candidate, learning_rate=rate) for rate in (1e-4, 1e-3)]
            variants += [dict(candidate, epochs=candidate["epochs"] + 4),
                         dict(candidate, weight_decay=1e-3)]
        elif family in {'hybrid_logistic'}:
            variants = [dict(candidate, C=value) for value in (.03, .3, 3., 10.)]
            variants += [dict(candidate, class_weight='balanced')]
            variants += [dict(candidate, fit_intercept=not candidate.get('fit_intercept', True))]
        elif family == 'hybrid_sgd':
            variants = [dict(candidate, alpha=value) for value in (1e-5, 1e-3, 1e-2)]
        elif family == 'hybrid_xgboost':
            variants = [dict(candidate, depth=value) for value in (4, 8)]
            variants += [dict(candidate, iterations=value) for value in (400, 1600)]
            variants += [dict(candidate, reg_lambda=value) for value in (1., 40.)]
            variants += [dict(candidate, learning_rate=value) for value in (.02, .08)]
        elif family == 'cross_boost_blend':
            variants = []
        elif family == 'text_blend':
            variants = [dict(candidate, linear_weight=w) for w in (.25, .75)]
            variants += [dict(candidate, C=c) for c in (.03, .3)]
        elif family == 'text_blend_diff':
            variants = [dict(candidate, linear_weight=w) for w in (.25, .75)]
            variants += [dict(candidate, C=c) for c in (.03, .3)]
            variants += [dict(candidate, linear_weight=.4, C=.03),
                         dict(candidate, linear_weight=.6, C=.3)]
        elif family == 'pairwise_overlap':
            variants = [dict(candidate, C=value) for value in (.3, 3.)]
            variants += [dict(candidate, fit_intercept=not candidate.get('fit_intercept', True)),
                         dict(candidate, class_weight='balanced')]
        elif family == 'pairwise_overlap_xgb':
            variants = [dict(candidate, iterations=count) for count in (400, 1200)]
            variants += [dict(candidate, depth=value) for value in (6, 8)]
            variants += [dict(candidate, reg_lambda=value) for value in (1., 30.)]
        elif family == 'pairwise_semantic_blend':
            variants = [dict(candidate, semantic_weight=value) for value in (.05, .2)]
            variants += [dict(candidate, C=value) for value in (.1, 10.)]
            variants += [dict(candidate, encoder=value) for value in ('bge', 'e5')]
        elif family == 'pairwise_semantic_mlp_blend':
            variants = [dict(candidate, mlp_weight=value) for value in (.25, .45)]
            variants += [dict(candidate, semantic_weight=value) for value in (.02, .1)]
            variants += [dict(candidate, hidden=value) for value in (256, 512)]
            variants += [dict(candidate, epochs=candidate.get('epochs', 40) + 10)]
        elif family == 'catboost_cross':
            variants = [dict(candidate, max_ctr_complexity=value) for value in (3, 8)]
            variants += [dict(candidate, one_hot_max_size=value) for value in (8, 64)]
            variants += [dict(candidate, depth=value) for value in (6, 10)]
            variants += [dict(candidate, max_leaves=value) for value in (32, 128)]
        elif family in {"linear_svm", "rbf_svm"}:
            variants = [dict(candidate, C=value) for value in (.01, 100.)]
        elif family == "logistic":
            variants = [dict(candidate, C=value) for value in (0.1, 10.0, 0.01, 100.0)]
            variants += [dict(candidate, class_weight="balanced")]
            variants += [dict(candidate, fit_intercept=not candidate.get('fit_intercept', True))]
        elif family in {"extra_trees", "random_forest"}:
            variants = [dict(candidate, max_depth=depth, min_samples_leaf=leaf)
                        for depth, leaf in ((8, 2), (None, 1), (24, 3), (12, 5))]
            variants += [dict(candidate, class_weight="balanced")]
        elif family == "nb":
            variants = [dict(candidate, alpha=value) for value in (0.1, 10.0, 0.01)]
        elif family in {"torch_mlp", "torch_cnn", "torch_resnet", "torch_ensemble"}:
            variants = [dict(candidate, learning_rate=rate) for rate in (0.003, 0.0003)]
            variants += [dict(candidate, epochs=candidate["epochs"] + 20),
                         dict(candidate, weight_decay=0.01)]
        else:
            variants = [dict(candidate, alpha=value) for value in (0.0001, 0.001, 0.01)]
        if imbalanced and family in {"logistic", "extra_trees", "random_forest"}:
            variants.insert(0, dict(candidate, class_weight="balanced"))
        for variant in allocate_candidates(variants, profile, history):
            if candidate_key(variant) not in tried:
                return variant, f"Refine {family} using observed {metric}={best['score']:.6g}"
    fallback = {"family": "sgd", "alpha": 0.0001}
    if profile["data_type"] == "text":
        fallback["analyzer"] = "char"
    if candidate_key(fallback) not in tried:
        return fallback, "Try a scalable probabilistic linear fallback after earlier results/failures"
    return None, "Search space exhausted"
