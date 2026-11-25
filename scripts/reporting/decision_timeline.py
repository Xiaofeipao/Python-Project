"""Render recorded events without inventing decisions or changing metrics."""

import argparse
import json
from pathlib import Path


def render(source):
    rows = ['# Decision timeline', '', 'Only recorded events are included; missing LLM reflections are not inferred.', '']
    for line in Path(source).read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        if item['event'] in {'data_analyzed', 'modality_inferred', 'llm_plan_selected', 'trial_finished',
                              'feedback_assessed', 'partial_checkpoint_finished', 'selection_completed', 'test_read_started', 'run_finished'}:
            fields = {k: item[k] for k in ('candidate', 'score', 'diagnosis', 'next_action', 'abandoned_strategy',
                       'status', 'inferred', 'outcome', 'utility_gain', 'validation_score') if k in item}
            rows.append(f"- {item['timestamp_utc']} **{item['event']}**: {json.dumps(fields, ensure_ascii=False)}")
    return '\n'.join(rows) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.events), encoding='utf-8')
