"""Validation-only search memory and conservative family allocation."""
import hashlib
import json
import statistics
from superagent.evaluation.metrics import utility


def trial_key(candidate):
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def family_stats(history, metric):
    result = {}
    for row in history:
        item = result.setdefault(row['candidate']['family'], {'scores': [], 'failures': 0, 'seconds': 0.})
        item['seconds'] += row.get('elapsed_seconds', 0)
        if row['status'] == 'success':
            item['scores'].append(utility(row['score'], metric))
        elif row['status'] in {'failed', 'timeout'}:
            item['failures'] += 1
    best = max((max(v['scores']) for v in result.values() if v['scores']), default=None)
    for item in result.values():
        values = item.pop('scores')
        item.update(successes=len(values), best_utility=max(values) if values else None,
                    mean_utility=statistics.mean(values) if values else None,
                    std=statistics.pstdev(values) if values else None)
        item['retired'] = item['failures'] >= 2 or (len(values) >= 3 and best is not None and
                         max(values) + 2 * item['std'] + .01 < best)
    return result


def state(history, metric):
    return {'trial_memory': {trial_key(h['candidate']): h['candidate'] for h in history},
            'families': family_stats(history, metric), 'completed_trials': len(history)}
