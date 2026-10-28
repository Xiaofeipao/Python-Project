"""A deterministic planner driven by data modality and measured validation feedback."""

import json

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

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
        if family == "logistic":
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
    common = {"random_state": config.seed}
    if family == "dummy":
        estimator = DummyClassifier(strategy="prior", **common)
    elif family == "logistic":
        estimator = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=1200,
                                       class_weight=candidate.get("class_weight"), **common)
    elif family in {"extra_trees", "random_forest"}:
        cls = ExtraTreesClassifier if family == "extra_trees" else RandomForestClassifier
        estimator = cls(n_estimators=candidate["n_estimators"], max_depth=candidate["max_depth"],
                        min_samples_leaf=candidate["min_samples_leaf"], n_jobs=config.threads,
                        class_weight=candidate.get("class_weight"), **common)
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
    return Pipeline([("preprocess", preprocessor(profile, candidate, config.image_root)),
                     ("classifier", estimator)])
