"""Prepare all pinned text snapshots and explicit ImageNet weights without training."""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from superagent.common.json_io import write_json
from superagent.models.cache import encoder_cache_path, text_cache_root, torch_cache_root
from superagent.models.text.encoders import ENCODER_REGISTRY

PATTERNS = ['*.json', '*.txt', '*.model', '*.safetensors', '*.bin']


def inventory(directory):
    result = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or '.cache' in path.relative_to(directory).parts:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        result.append({'path': path.relative_to(directory).as_posix(),
                       'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    return result


def prepare_text(name, spec, offline):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    target = encoder_cache_path(spec)
    # Reuse prior materialized snapshots without moving or deleting originals.
    legacy = Path.home() / '.cache' / 'superagent_models' / target.name
    if legacy.is_dir() and not target.exists():
        shutil.copytree(legacy, target)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(spec['model_id'], revision=spec['revision'], local_dir=str(target),
                      allow_patterns=PATTERNS, local_files_only=offline)
    AutoConfig.from_pretrained(str(target), local_files_only=True, trust_remote_code=False)
    AutoTokenizer.from_pretrained(str(target), local_files_only=True, trust_remote_code=False)
    files = inventory(target)
    if not any(row['path'] in {'model.safetensors', 'pytorch_model.bin'}
               or row['path'].endswith('.index.json') for row in files):
        raise ValueError(f'No model weights in {target}')
    return {'name': name, 'model_id': spec['model_id'], 'revision': spec['revision'],
            'source': 'https://huggingface.co/' + spec['model_id'] + '/tree/' + spec['revision'],
            'path': str(target), 'files': files, 'status': 'ready'}


def prepare_images(offline):
    import torch
    from torchvision import models

    original = Path(torch.hub.get_dir()) / 'checkpoints'
    torch.hub.set_dir(str(torch_cache_root() / 'hub'))
    target = Path(torch.hub.get_dir()) / 'checkpoints'
    target.mkdir(parents=True, exist_ok=True)
    result = []
    for name, weights in (('resnet18', models.ResNet18_Weights.IMAGENET1K_V1),
                          ('resnet34', models.ResNet34_Weights.IMAGENET1K_V1)):
        filename = Path(urlparse(weights.url).path).name
        path = target / filename
        if not path.exists() and (original / filename).is_file():
            shutil.copy2(original / filename, path)
        if offline and not path.is_file():
            raise FileNotFoundError(path)
        # The publisher prefix is also checked on pre-existing local weights.
        weights.get_state_dict(progress=True, check_hash=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        prefix = path.stem.rsplit('-', 1)[-1]
        if not digest.startswith(prefix):
            raise ValueError(f'Publisher hash mismatch: {path}')
        result.append({'name': name, 'weight_enum': str(weights), 'source': weights.url,
                       'path': str(path), 'bytes': path.stat().st_size,
                       'sha256': digest, 'status': 'ready'})
        print(f'READY {name}: {path}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Check existing local resources only')
    parser.add_argument('--text', nargs='*', choices=sorted(ENCODER_REGISTRY),
                        help='Selected encoder keys; omit to prepare all seven')
    parser.add_argument('--skip-images', action='store_true')
    args = parser.parse_args()
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'text_root': str(text_cache_root()), 'torch_root': str(torch_cache_root()),
              'models': [], 'errors': []}
    manifest = ROOT / '.pretrained_cache' / 'manifest.json'
    for name in args.text if args.text is not None else ENCODER_REGISTRY:
        try:
            item = prepare_text(name, ENCODER_REGISTRY[name], args.offline)
            report['models'].append(item)
            print(f'READY {name}: {item["path"]}', flush=True)
        except Exception as exc:
            report['errors'].append({'name': name, 'error': str(exc)})
            print(f'FAILED {name}: {exc}', flush=True)
        write_json(manifest, report)
    if not args.skip_images:
        try:
            report['models'].extend(prepare_images(args.offline))
        except Exception as exc:
            report['errors'].append({'name': 'image_weights', 'error': str(exc)})
            print(f'FAILED image weights: {exc}', flush=True)
    report['complete'] = not report['errors']
    write_json(manifest, report)
    print(f'Manifest: {manifest}', flush=True)
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
