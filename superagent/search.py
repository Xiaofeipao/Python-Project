"""A deterministic planner driven by data modality and measured validation feedback."""

import json
import importlib.util

from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from .features import preprocessor
from .metrics import utility


def candidate_key(candidate):
    return json.dumps(candidate, sort_keys=True)


def next_candidate(profile, history, metric):
    tried = {candidate_key(h["candidate"]) for h in history}
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
        initial[1:1] = [strong, wider, ensemble]
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
        initial.insert(0, {'family': 'text_blend', 'linear_weight': .5, 'C': .1})
    if profile['data_type'] == 'tabular' and large_gpu and importlib.util.find_spec('xgboost') is not None:
        initial[0:0] = [{'family': 'cross_boost_blend'}] + [
            {'family': 'cross_boost_blend', 'gpu_weight': weight} for weight in (.1, .5, .75)] + [
            {'family': 'xgboost', 'iterations': count, 'depth': depth, 'learning_rate': .03,
             'reg_lambda': regularization} for count, depth, regularization in
            ((2000, 4, 10), (1600, 6, 10), (1200, 8, 20))]
        initial[1:1] = [
            {'family': 'cross_boost_blend', 'gpu_seeds': 3, 'gpu_weight': weight,
             'leaf_weight': leaf} for weight, leaf in ((.25, 0.), (.5, 0.), (.25, .15))]
    for candidate in initial:
        if candidate_key(candidate) not in tried:
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
        elif family == 'lightgbm':
            variants = [dict(candidate, num_leaves=value) for value in (15, 63, 127)]
            variants += [dict(candidate, iterations=value) for value in (400, 1600)]
        elif family in {'catboost_ensemble', 'tabular_blend'}:
            variants = []
        elif family == 'xgboost':
            variants = [dict(candidate, reg_lambda=value) for value in (1, 20, 50)]
        elif family == 'cross_boost_blend':
            variants = []
        elif family == 'text_blend':
            variants = [dict(candidate, linear_weight=w) for w in (.25, .75)]
            variants += [dict(candidate, C=c) for c in (.03, .3)]
        elif family in {"linear_svm", "rbf_svm"}:
            variants = [dict(candidate, C=value) for value in (.01, 100.)]
        elif family == "logistic":
            variants = [dict(candidate, C=value) for value in (0.1, 10.0, 0.01, 100.0)]
            variants += [dict(candidate, class_weight="balanced")]
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
        for variant in variants:
            if candidate_key(variant) not in tried:
                return variant, f"Refine {family} using observed {metric}={best['score']:.6g}"
    fallback = {"family": "sgd", "alpha": 0.0001}
    if profile["data_type"] == "text":
        fallback["analyzer"] = "char"
    if candidate_key(fallback) not in tried:
        return fallback, "Try a scalable probabilistic linear fallback after earlier results/failures"
    return None, "Search space exhausted"


def build_model(profile, candidate, config):
    family = candidate["family"]
    if family == 'text_blend':
        from .tabular_models import ProbabilityBlend
        weight = candidate.get('linear_weight', .5)
        members = [build_model(profile, {'family': 'logistic', 'C': 1., 'analyzer': 'char',
                    'representation': 'separate', 'class_weight': 'balanced'}, config),
                   build_model(profile, {'family': 'linear_svm', 'C': candidate.get('C', .1),
                    'analyzer': 'combined', 'representation': 'separate'}, config)]
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(members, weights=[weight, 1-weight]))])
    common = {"random_state": config.seed}
    if family == "dummy":
        estimator = DummyClassifier(strategy="prior", **common)
    elif family == "logistic":
        estimator = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=1200,
                                       class_weight=candidate.get("class_weight"), **common)
    elif family == 'linear_svm':
        estimator = CalibratedClassifierCV(LinearSVC(C=candidate['C'], dual='auto', max_iter=10000,
                                                     **common), cv=3, n_jobs=config.threads)
    elif family == 'rbf_svm':
        estimator = CalibratedClassifierCV(SVC(C=candidate['C'], cache_size=512, **common),
                                           cv=3, ensemble=False, n_jobs=config.threads)
    elif family in {"extra_trees", "random_forest"}:
        from .tabular_models import BalancedExtraTrees, BalancedRandomForest

        cls = BalancedExtraTrees if family == "extra_trees" else BalancedRandomForest
        estimator = cls(n_estimators=candidate["n_estimators"], max_depth=candidate["max_depth"],
                        min_samples_leaf=candidate["min_samples_leaf"], n_jobs=config.threads,
                        class_weight=candidate.get("class_weight"), **common)
    elif family == "catboost":
        from .tabular_models import NativeBoostClassifier

        estimator = NativeBoostClassifier(iterations=candidate['iterations'], depth=candidate['depth'],
                                          learning_rate=candidate['learning_rate'],
                                          l2_leaf_reg=candidate['l2_leaf_reg'], threads=config.threads, **common)
    elif family in {'lightgbm', 'catboost_ensemble', 'tabular_blend'}:
        from .tabular_models import advanced_estimator

        estimator = advanced_estimator(candidate, config)
    elif family == 'xgboost':
        from .tabular_models import EncodedXGBoost
        estimator = EncodedXGBoost(iterations=candidate['iterations'], depth=candidate['depth'],
                                    learning_rate=candidate['learning_rate'], reg_lambda=candidate['reg_lambda'],
                                    threads=config.threads, **common)
    elif family == 'cross_boost_blend':
        from .tabular_models import ProbabilityBlend, NativeBoostClassifier, EncodedXGBoost, NativeLeafBoost
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
    elif family in {"torch_mlp", "torch_cnn", "torch_resnet", "torch_ensemble"}:
        from .torch_models import TorchClassifier, TorchEnsemble

        params = dict(
            architecture=candidate.get("architecture", "cnn" if family == "torch_cnn" else "mlp"), hidden=candidate["hidden"],
            epochs=candidate["epochs"], learning_rate=candidate["learning_rate"],
            weight_decay=candidate["weight_decay"], image_size=candidate.get("size", 32),
            threads=config.threads, batch_size=config.neural_batch_size,
            mixed_precision=config.mixed_precision, gpu_cache_fraction=config.gpu_cache_fraction,
            depth=candidate.get("depth", 3), width=candidate.get("width", 32),
            dropout=candidate.get("dropout", 0.1), early_stopping=candidate.get("early_stopping", False),
            augment=candidate.get("augment", False), cosine_schedule=candidate.get("cosine_schedule", False),
            normalize_images=candidate.get("normalize_images", False),
        )
        if family == "torch_ensemble":
            estimator = TorchEnsemble(member_params=params, size=candidate.get("ensemble_size", 3),
                                      device=config.device, **common)
        else:
            estimator = TorchClassifier(**params, device=config.device, **common)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    if family in {"catboost", "lightgbm", "catboost_ensemble", "tabular_blend", "cross_boost_blend"}:
        from .tabular_models import NativeCategories

        return Pipeline([("preprocess", NativeCategories(profile['features'], profile['numeric_columns'])),
                         ("classifier", estimator)])
    return Pipeline([("preprocess", preprocessor(profile, candidate, config.image_root)),
                     ("classifier", estimator)])
