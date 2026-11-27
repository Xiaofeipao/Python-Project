"""A deterministic planner driven by data modality and measured validation feedback."""

import json
import importlib.util
import statistics

from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from superagent.preprocessing.transformers import hybrid_preprocessor, preprocessor, tabular_preprocessor
from superagent.evaluation.metrics import utility


def candidate_key(candidate):
    return json.dumps(candidate, sort_keys=True)


def metric_candidates(profile, history, metric):
    """Validation-compared variants, with fixed thresholds retained at refit."""
    families = {'logistic', 'sgd', 'catboost', 'lightgbm', 'xgboost',
                'hybrid_logistic', 'hybrid_sgd', 'hybrid_xgboost', 'extra_trees', 'random_forest'}
    successful = sorted([h for h in history if h['status'] == 'success'
                         and h['candidate']['family'] in families],
                        key=lambda h: utility(h['score'], metric), reverse=True)
    variants, seen = [], set()
    for row in successful:
        candidate = row['candidate']
        if candidate['family'] in seen or len(seen) >= 3:
            continue
        seen.add(candidate['family'])
        base = {k: v for k, v in candidate.items() if k not in {'decision_threshold', 'calibration'}}
        if metric in {'f1_macro', 'f1_weighted', 'balanced_accuracy'} and len(profile.get('class_counts', {})) == 2:
            variants += [dict(base, decision_threshold=t) for t in (.35, .65)]
        elif (metric in {'log_loss', 'roc_auc'}
              and min(profile.get('fitting_class_counts', profile.get('class_counts', {})).values(), default=0) >= 2):
            variants.append(dict(base, calibration='sigmoid'))
    return variants


def allocate_candidates(candidates, profile, history):
    """Allocate optional exploration from modality, scale and remaining time."""
    remaining = profile.get('remaining_search_seconds')
    if remaining is None:
        return candidates
    rows = profile.get('rows', 0)
    costly = {'minilm_logistic', 'torch_pretrained', 'frozen_image_logistic', 'torch_ensemble',
              'image_meta_blend', 'rbf_svm', 'embedding_logistic', 'embedding_mlp',
              'text_fasttext', 'text_cnn', 'text_rnn', 'text_xgboost', 'transformer_finetune'}
    eligible = [c for c in candidates if not (c['family'] in costly and remaining < 60)
                and not (c['family'] == 'rbf_svm' and rows > 20000)]
    # Preserve established general baselines before optional feature/encoder sweeps.
    def priority(c):
        family = c['family']
        anchor = (family == 'cross_boost_blend' and profile['data_type'] == 'tabular'
                  or family in {'logistic', 'pairwise_overlap'} and profile['data_type'] == 'text'
                  and (c.get('analyzer') == 'char_wb' or family == 'pairwise_overlap')
                  and not c.get('pairwise_word_features'))
        # Once the cheap baseline exists, expose the new text families before
        # spending the remaining trials only on close variants of the same
        # sparse linear model.
        novel = {'text_fasttext', 'text_cnn', 'text_rnn', 'text_xgboost',
                 'embedding_logistic', 'embedding_mlp'}
        novel_ready = len(history) >= 2 and family in novel
        return (0 if anchor or novel_ready else 1 if family == 'frozen_image_logistic'
                else 2 if family in costly or c.get('feature_filter') else 1)
    eligible.sort(key=priority)
    limit = min(remaining, profile.get('trial_timeout_seconds', remaining))
    measured = [h for h in history if h['status'] in {'success', 'failed', 'timeout'}]
    affordable = []
    for candidate in eligible:
        relatives = [h for h in measured if h['candidate']['family'] == candidate['family']]
        estimates = []
        for row in relatives:
            elapsed = row.get('wall_seconds_including_startup', row.get('elapsed_seconds', 0))
            factor = max([1.] + [candidate.get(k, 1) / max(1, row['candidate'].get(k, 1))
                                for k in ('iterations', 'epochs', 'gpu_seeds')])
            if candidate.get('calibration') and not row['candidate'].get('calibration'):
                factor *= 3
            estimates.append(elapsed * factor)
        if not estimates or statistics.median(estimates) * 1.25 <= limit:
            affordable.append(candidate)
    eligible = affordable
    # Cover different families before spending the first eight slots tuning one.
    if len(measured) < 8:
        seen = {h['candidate']['family'] for h in history}
        eligible.sort(key=lambda c: c['family'] in seen)
    # After two attempts in one family, expose other families first.
    if len(history) >= 3:
        recent = [h['candidate']['family'] for h in history[-2:]]
        if recent[0] == recent[1]:
            eligible.sort(key=lambda c: c['family'] == recent[0])
    return eligible


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
        ] + [
            {'family': 'xgboost', 'iterations': count, 'depth': depth, 'learning_rate': .03,
             'reg_lambda': regularization} for count, depth, regularization in
            ((2000, 4, 10), (1600, 6, 10), (1200, 8, 20))]
    if profile.get('minilm_available') and profile['data_type'] in {'tabular', 'text'}:
        from superagent.models.text_encoder import MODEL_ID, REVISION
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


def build_model(profile, candidate, config):
    model = _build_model(profile, candidate, config)
    if candidate.get('feature_filter', 'none') != 'none':
        if candidate['family'] not in {'logistic','linear_svm','sgd','extra_trees','random_forest','nb','rbf_svm'}:
            raise ValueError('Feature filters require a numeric feature pipeline')
        from superagent.preprocessing.feature_selection import FeatureFilter
        model.steps.insert(-1, ('feature_filter', FeatureFilter(candidate['feature_filter'],
                           candidate.get('percentile',50),config.seed)))
    if candidate.get('decision_threshold') is not None or candidate.get('calibration'):
        from superagent.models.decision import DecisionClassifier
        model = Pipeline([('preprocess', 'passthrough'), ('classifier', DecisionClassifier(
            model, threshold=candidate.get('decision_threshold'), calibration=candidate.get('calibration'),
            positive_label=config.positive_label))])
    return model


def _build_model(profile, candidate, config):
    family = candidate["family"]
    if family == 'frozen_image_logistic':
        from superagent.models.neural import FrozenImageFeatures
        if profile['data_type'] != 'image':
            raise ValueError('Frozen image features require image input')
        return Pipeline([('preprocess', FrozenImageFeatures(profile['image_column'], config.image_root,
                    size=candidate.get('size', 96), device=config.device, threads=config.threads,
                    batch_size=candidate.get('batch_size', 32), random_state=config.seed,
                    pretrained=candidate.get('pretrained', True))),
                         ('classifier', LogisticRegression(C=candidate.get('C', 1.), max_iter=1200,
                                                          random_state=config.seed))])
    if family == 'hybrid_logistic':
        return Pipeline([('preprocess', hybrid_preprocessor(profile, candidate)),
                         ('classifier', LogisticRegression(C=candidate.get('C', 1.),
                                                           class_weight=candidate.get('class_weight'),
                                                           fit_intercept=candidate.get('fit_intercept', True),
                                                           solver='lbfgs', max_iter=1200,
                                                           random_state=config.seed))])
    if family == 'hybrid_sgd':
        return Pipeline([('preprocess', hybrid_preprocessor(profile, candidate)),
                         ('classifier', SGDClassifier(loss='log_loss', alpha=candidate.get('alpha', 1e-4),
                                                      max_iter=1500, tol=1e-4,
                                                      random_state=config.seed))])
    if family == 'hybrid_xgboost':
        from superagent.models.tabular import EncodedXGBoost
        return Pipeline([('preprocess', hybrid_preprocessor(profile, candidate)),
                         ('classifier', EncodedXGBoost(
                             iterations=candidate.get('iterations', 800),
                             depth=candidate.get('depth', 6),
                             learning_rate=candidate.get('learning_rate', .05),
                             reg_lambda=candidate.get('reg_lambda', 10),
                             threads=config.threads, balanced=candidate.get('balanced', False),
                             device=candidate.get('device', 'auto'),
                             random_state=config.seed))])
    if family == 'image_meta_blend':
        from superagent.models.tabular import ProbabilityBlend
        image = {'family': 'torch_pretrained', 'architecture': candidate.get('architecture', 'pretrained_resnet18'),
                 'size': candidate.get('size', 96), 'epochs': candidate.get('epochs', 18),
                 'learning_rate': candidate.get('learning_rate', 3e-4), 'weight_decay': 1e-4,
                 'dropout': .2, 'early_stopping': True, 'cosine_schedule': True,
                 'augment': True, 'imagenet_normalize': True, 'tta': True,
                 'pretrained': candidate.get('pretrained', True)}
        meta = dict(profile, data_type='tabular', image_column=None, text_columns=None,
                    features=profile['tabular_columns'],
                    numeric_columns=profile['numeric_columns'],
                    categorical_columns=profile['categorical_columns'],
                    datetime_columns=profile.get('datetime_columns', []))
        if importlib.util.find_spec('catboost') is not None:
            meta_candidate = {'family': 'catboost', 'iterations': candidate.get('meta_iterations', 900),
                              'depth': 6, 'learning_rate': .05, 'l2_leaf_reg': 5}
        else:
            meta_candidate = {'family': 'logistic', 'C': 1.}
        members = [build_model(profile, image, config), build_model(meta, meta_candidate, config)]
        weight = candidate.get('image_weight', .7)
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(members, weights=[weight, 1 - weight]))])
    if family == 'embedding_logistic':
        from superagent.models.text_encoder import TextEncoderFeatures
        from sklearn.preprocessing import StandardScaler
        if profile['data_type'] != 'text':
            raise ValueError('Frozen transformer probes require text input')
        columns = profile['text_columns']
        return Pipeline([('preprocess', TextEncoderFeatures(
                             columns, text=True, encoder=candidate.get('encoder', 'mpnet'),
                             device=config.device, batch_size=candidate.get('batch_size', 64),
                             max_length=candidate.get('max_length', 256))),
                         ('scale', StandardScaler()),
                         ('classifier', LogisticRegression(C=candidate.get('C', 1.),
                                                           class_weight=candidate.get('class_weight'),
                                                           max_iter=2500, random_state=config.seed))])
    if family == 'embedding_mlp':
        from superagent.models.text_encoder import TextEncoderFeatures
        from superagent.models.neural import TorchClassifier
        from sklearn.preprocessing import StandardScaler
        if profile['data_type'] != 'text':
            raise ValueError('Frozen transformer probes require text input')
        return Pipeline([('preprocess', TextEncoderFeatures(
                             profile['text_columns'], text=True,
                             encoder=candidate.get('encoder', 'mpnet'), device=config.device,
                             batch_size=candidate.get('batch_size', 64),
                             max_length=candidate.get('max_length', 256))),
                         ('scale', StandardScaler()),
                         ('classifier', TorchClassifier(
                             architecture='mlp', hidden=candidate.get('hidden', 256),
                             epochs=candidate.get('epochs', 30),
                             learning_rate=candidate.get('learning_rate', .001),
                             weight_decay=candidate.get('weight_decay', .01),
                             batch_size=config.neural_batch_size,
                             dropout=candidate.get('dropout', .2), device=config.device,
                             threads=config.threads, random_state=config.seed,
                             mixed_precision=config.mixed_precision,
                             gpu_cache_fraction=config.gpu_cache_fraction))])
    if family == 'transformer_finetune':
        from superagent.models.text_finetune import TransformerFinetuneClassifier
        if profile['data_type'] != 'text':
            raise ValueError('Transformer fine-tuning requires text input')
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', TransformerFinetuneClassifier(
                             profile['text_columns'], encoder=candidate.get('encoder', 'mpnet'),
                             epochs=candidate.get('epochs', 3),
                             learning_rate=candidate.get('learning_rate', 2e-5),
                             weight_decay=candidate.get('weight_decay', .01),
                             batch_size=candidate.get('batch_size', 16),
                             max_length=candidate.get('max_length', 256),
                             freeze_layers=candidate.get('freeze_layers', 0),
                             device=config.device, threads=config.threads,
                             random_state=config.seed))])
    if family == 'text_xgboost':
        from superagent.models.tabular import EncodedXGBoost
        if profile['data_type'] != 'text':
            raise ValueError('Text XGBoost requires text input')
        return Pipeline([('preprocess', preprocessor(profile, candidate, config.image_root)),
                         ('classifier', EncodedXGBoost(
                             iterations=candidate.get('iterations', 1000),
                             depth=candidate.get('depth', 4),
                             learning_rate=candidate.get('learning_rate', .03),
                             reg_lambda=candidate.get('reg_lambda', 10),
                             threads=config.threads, balanced=candidate.get('balanced', False),
                             device=config.device, random_state=config.seed))])
    if family in {'text_fasttext', 'text_cnn', 'text_rnn'}:
        from superagent.models.text_neural import TextNeuralClassifier
        if profile['data_type'] != 'text':
            raise ValueError('Trainable text networks require text input')
        architecture = {'text_fasttext': 'fasttext', 'text_cnn': 'textcnn',
                        'text_rnn': 'textrnn'}[family]
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', TextNeuralClassifier(
                             profile['text_columns'], architecture=architecture,
                             embedding_dim=candidate.get('embedding_dim', 128),
                             hidden=candidate.get('hidden', 128),
                             epochs=candidate.get('epochs', 25),
                             learning_rate=candidate.get('learning_rate', .001),
                             weight_decay=candidate.get('weight_decay', .01),
                             batch_size=candidate.get('batch_size', config.neural_batch_size),
                             max_length=candidate.get('max_length', 256),
                             ngram_max=candidate.get('ngram_max', 2),
                             dropout=candidate.get('dropout', .2), device=config.device,
                             threads=config.threads, random_state=config.seed,
                             max_vocab=candidate.get('max_vocab', 30000)))])
    if family == 'minilm_logistic':
        from superagent.models.text_encoder import MiniLMFeatures
        from sklearn.preprocessing import StandardScaler
        columns = profile['text_columns'] if profile['data_type'] == 'text' else profile['features']
        return Pipeline([('preprocess', MiniLMFeatures(columns, text=profile['data_type'] == 'text',
                                                      device=config.device, batch_size=config.neural_batch_size)),
                         ('scale', StandardScaler()),
                         ('classifier', LogisticRegression(C=candidate['C'], max_iter=2000,
                                                           random_state=config.seed))])
    if family == 'text_blend':
        from superagent.models.tabular import ProbabilityBlend
        weight = candidate.get('linear_weight', .5)
        members = [build_model(profile, {'family': 'logistic', 'C': 1., 'analyzer': 'char',
                    'representation': 'separate', 'class_weight': 'balanced'}, config),
                   build_model(profile, {'family': 'linear_svm', 'C': candidate.get('C', .1),
                    'analyzer': 'combined', 'representation': 'separate'}, config)]
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(members, weights=[weight, 1-weight]))])
    if family == 'text_blend_diff':
        from superagent.models.tabular import ProbabilityBlend
        weight = candidate.get('linear_weight', .5)
        members = [build_model(profile, {'family': 'logistic', 'C': 1., 'analyzer': 'char_wb',
                    'representation': 'difference', 'class_weight': 'balanced'}, config),
                   build_model(profile, {'family': 'linear_svm', 'C': candidate.get('C', .1),
                    'analyzer': 'word', 'representation': 'difference'}, config)]
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(members, weights=[weight, 1-weight]))])
    if family == 'pairwise_overlap':
        from superagent.models.pairwise import overlap_classifier
        return overlap_classifier(profile['text_columns'], candidate, config.seed)
    common = {"random_state": config.seed}
    if family == "dummy":
        estimator = DummyClassifier(strategy="prior", **common)
    elif family == "logistic":
        estimator = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=1200,
                                       class_weight=candidate.get("class_weight"),
                                       fit_intercept=candidate.get('fit_intercept',True), **common)
    elif family == 'linear_svm':
        estimator = CalibratedClassifierCV(LinearSVC(C=candidate['C'], dual='auto', max_iter=10000,
                                                     **common), cv=3, n_jobs=config.threads)
    elif family == 'rbf_svm':
        estimator = CalibratedClassifierCV(SVC(C=candidate['C'], cache_size=512, **common),
                                           cv=3, ensemble=False, n_jobs=config.threads)
    elif family in {"extra_trees", "random_forest"}:
        from superagent.models.tabular import BalancedExtraTrees, BalancedRandomForest

        cls = BalancedExtraTrees if family == "extra_trees" else BalancedRandomForest
        estimator = cls(n_estimators=candidate["n_estimators"], max_depth=candidate["max_depth"],
                        min_samples_leaf=candidate["min_samples_leaf"], n_jobs=config.threads,
                        class_weight=candidate.get("class_weight"), **common)
    elif family == "catboost":
        from superagent.models.tabular import NativeBoostClassifier

        estimator = NativeBoostClassifier(iterations=candidate['iterations'], depth=candidate['depth'],
                                          learning_rate=candidate['learning_rate'],
                                          l2_leaf_reg=candidate['l2_leaf_reg'], threads=config.threads,
                                          balanced=candidate.get('balanced', False), **common)
    elif family in {'lightgbm', 'catboost_ensemble', 'tabular_blend'}:
        from superagent.models.tabular import advanced_estimator

        estimator = advanced_estimator(candidate, config)
    elif family == 'xgboost':
        from superagent.models.tabular import EncodedXGBoost
        estimator = EncodedXGBoost(iterations=candidate['iterations'], depth=candidate['depth'],
                                    learning_rate=candidate['learning_rate'], reg_lambda=candidate['reg_lambda'],
                                    threads=config.threads, balanced=candidate.get('balanced', False), **common)
    elif family == 'cross_boost_blend':
        from superagent.models.tabular import ProbabilityBlend, NativeBoostClassifier, EncodedXGBoost, NativeLeafBoost
        encoded = [Pipeline([('preprocess', preprocessor(profile, {'family': 'xgboost'}, config.image_root)),
                            ('classifier', EncodedXGBoost(iterations=candidate.get('iterations', 1600),
                                                         random_state=config.seed + 101 * index,
                                                         threads=config.threads))])
                   for index in range(candidate.get('gpu_seeds', 1))]
        weight = candidate.get('gpu_weight', .25)
        leaf = candidate.get('leaf_weight', 0.)
        estimator = ProbabilityBlend([
            *[NativeBoostClassifier(iterations=candidate.get('iterations', 1200), depth=8, learning_rate=.05,
                                    l2_leaf_reg=5, threads=config.threads,
                                    random_state=config.seed + 101 * index) for index in range(3)], *encoded,
            *([NativeLeafBoost(random_state=config.seed, threads=config.threads)] if leaf else [])],
            weights=[(1 - weight - leaf) / 3] * 3 + [weight / len(encoded)] * len(encoded)
                    + ([leaf] if leaf else []))
    elif family == "nb":
        estimator = ComplementNB(alpha=candidate["alpha"])
    elif family == "sgd":
        estimator = SGDClassifier(loss="log_loss", alpha=candidate["alpha"], max_iter=1500,
                                  tol=1e-4, **common)
    elif family in {"torch_mlp", "torch_cnn", "torch_resnet", "torch_ensemble", "torch_pretrained"}:
        from superagent.models.neural import TorchClassifier, TorchEnsemble

        default_architecture = ("cnn" if family == "torch_cnn" else
                                "pretrained_resnet18" if family == "torch_pretrained" else "mlp")
        params = dict(
            architecture=candidate.get("architecture", default_architecture),
            epochs=candidate["epochs"], learning_rate=candidate["learning_rate"],
            weight_decay=candidate["weight_decay"], image_size=candidate.get("size", 32),
            threads=config.threads, batch_size=config.neural_batch_size,
            mixed_precision=config.mixed_precision, gpu_cache_fraction=config.gpu_cache_fraction,
            depth=candidate.get("depth", 3), width=candidate.get("width", 32),
            dropout=candidate.get("dropout", 0.1), early_stopping=candidate.get("early_stopping", False),
            augment=candidate.get("augment", False), cosine_schedule=candidate.get("cosine_schedule", False),
            normalize_images=candidate.get("normalize_images", False),
            imagenet_normalize=candidate.get("imagenet_normalize", False),
            pretrained=candidate.get("pretrained", True), tta=candidate.get("tta", False),
        )
        if family != "torch_pretrained":
            params["hidden"] = candidate["hidden"]
        if family == "torch_ensemble":
            estimator = TorchEnsemble(member_params=params, size=candidate.get("ensemble_size", 3),
                                      device=config.device, **common)
        else:
            estimator = TorchClassifier(**params, device=config.device, **common)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    if family in {"catboost", "lightgbm", "catboost_ensemble", "tabular_blend", "cross_boost_blend"}:
        from superagent.models.tabular import NativeCategories

        return Pipeline([("preprocess", NativeCategories(
            profile.get('tabular_columns') or profile['features'],
            profile['numeric_columns'], profile.get('datetime_columns', []))),
                         ("classifier", estimator)])
    return Pipeline([("preprocess", preprocessor(profile, candidate, config.image_root)),
                     ("classifier", estimator)])
