"""Run configured released tasks serially; print scores only after both finish."""

import json
import argparse
import os
import subprocess
import sys
from pathlib import Path
from test_numpy_task import convert_split

ROOT = Path(__file__).resolve().parents[1]


def invoke(folder, entry, *args):
    process = subprocess.run([sys.executable, str(entry), *map(str, args)], cwd=ROOT,
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
    (folder / f'{args[0]}.console.log').write_text(process.stdout + process.stderr, encoding='utf-8')
    if process.returncode:
        raise RuntimeError(f'{args[0]} failed; inspect {folder}')
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/released_gpu_batch_20261111')
    parser.add_argument('--tasks', nargs='+', choices=['coupon', 'headlines'], default=['coupon', 'headlines'])
    parser.add_argument('--max-trials', type=int, default=36)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['SUPERAGENT_USAGE_ROOT'] = str(ROOT / '.model_usage')
    policy = {'max_trials': args.max_trials, 'time_budget_seconds': 2400, 'trial_timeout_seconds': 600,
              'threads': 4, 'device': 'cuda', 'compute_profile': 'gpu_large', 'llm_mode': 'disabled'}
    (output / 'policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    frozen = invoke(output, ROOT / 'main.py', 'freeze', '--policy', output / 'policy.json',
                    '--output', output / 'frozen_agent')
    entry = output / 'frozen_agent/main.py'
    inputs = [
        ('coupon', ROOT / 'in_vehicle_coupon_recommendation', 'train.csv', 'val.csv', 'test.csv',
         {'data_type': 'tabular', 'target_column': 'label', 'id_column': 'id'}, .7922743397713835),
        ('headlines', ROOT / 'Released_Task2-Headlines_CTR', 'train.csv', 'val.csv', 'test.csv',
         {'data_type': 'text', 'target_column': 'label_pairwise', 'text_columns': ['headline_A', 'headline_B']},
         .6605816831683168),
    ]
    results = {'purpose': 'Development comparison; reused test data is not independent final evaluation',
               'runtime_sha256': frozen['runtime_sha256'], 'tasks': {}}
    for name, data, train, val, test, fields, baseline in inputs:
        if name not in args.tasks:
            continue
        folder = output / name
        folder.mkdir()
        arrays = (data / 'C_train.npy').exists() or (data / 'N_train.npy').exists()
        source = data
        if arrays:
            data = folder / 'converted'
            data.mkdir()
            for split in ('train', 'val'):
                convert_split(source, split, data / f'{split}.csv')
        config = {**fields, 'train_csv': str(data / train), 'validation_csv': str(data / val),
                  'metric': 'accuracy', 'evaluation_stage': 'released',
                  'evaluation_id': f'released-{name}', 'frozen_manifest': frozen['manifest']}
        path = folder / 'task.json'
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        try:
            summary = invoke(folder, entry, 'train', '--config', path, '--output', folder / 'agent')
            if arrays:
                convert_split(source, 'test', data / test)
            invoke(folder, entry, 'predict', '--model', folder / 'agent/model.joblib',
                   '--input', data / test, '--output', folder / 'agent/predictions.csv')
            evaluation = invoke(folder, entry, 'evaluate', '--model', folder / 'agent/model.joblib',
                                '--input', data / test, '--output', folder / 'test_evaluation.json')
            results['tasks'][name] = {'status': 'completed', 'baseline_accuracy': baseline,
                                      'accuracy': evaluation['score'], 'improved': evaluation['score'] > baseline,
                                      'training': summary, 'evaluation': evaluation}
        except Exception as error:
            results['tasks'][name] = {'status': 'failed', 'error': str(error)}
        (output / 'result.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in result.items() if k not in {'training', 'evaluation'}}
                      for name, result in results['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
