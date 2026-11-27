"""Candidate identity, metric variants and resource-aware allocation."""

import json
import statistics
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
              'text_fasttext', 'text_cnn', 'text_rnn', 'text_xgboost', 'transformer_finetune',
              'pairwise_semantic_blend', 'pairwise_semantic_mlp_blend', 'pairwise_overlap_xgb'}
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
                 'embedding_logistic', 'embedding_mlp', 'pairwise_semantic_blend',
                 'pairwise_semantic_mlp_blend', 'pairwise_overlap_xgb', 'catboost_cross'}
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
