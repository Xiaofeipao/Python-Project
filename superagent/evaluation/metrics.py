"""Metric direction, probability handling, and JSON-compatible evaluations."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score


def utility(score, metric):
    return -score if metric == "log_loss" else score


def score_model(model, X, y, metric, positive_label=None):
    if metric == "accuracy":
        score = accuracy_score(y, model.predict(X))
    elif metric == "balanced_accuracy":
        score = balanced_accuracy_score(y, model.predict(X))
    elif metric in {"f1_macro", "f1_weighted"}:
        score = f1_score(y, model.predict(X), average=metric.split("_", 1)[1], zero_division=0)
    else:
        probabilities = model.predict_proba(X)
        classes = [str(c) for c in model.classes_]
        if metric == "log_loss":
            score = log_loss(y, probabilities, labels=classes)
        elif len(classes) == 2:
            positive = positive_label if positive_label is not None else classes[-1]
            index = classes.index(positive)
            score = roc_auc_score(np.asarray(y) == positive, probabilities[:, index])
        else:
            score = roc_auc_score(y, probabilities, labels=classes, multi_class="ovr", average="macro")
    score = float(score)
    if not np.isfinite(score):
        raise ValueError(f"Nonfinite {metric} score")
    return score
