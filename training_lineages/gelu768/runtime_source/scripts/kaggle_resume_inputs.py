"""Validate a pinned completed remote parent; never copy its optimizer locally."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verified_parent(input_root,record,cache_sha,hardware):
    matches=[p for p in Path(input_root).rglob('session_result.json') if sha(p)==record['terminal_sha256']]
    if len(matches)!=1:raise ValueError('Exactly one pinned completed parent terminal required')
    terminal_path=matches[0];terminal=json.loads(terminal_path.read_text());root=terminal_path.parent
    run=record['run'];step=record['step']
    if (terminal.get('complete') is not True or terminal.get('training_complete') is not True or
        terminal['iterations'].get(run)!=step or terminal['training_source_sha256']!=record['source_sha256'] or
        terminal['driver_sha256']!=record['driver_sha256'] or terminal['cache_manifest_sha256']!=cache_sha):
        raise ValueError('Parent completion, count, source, driver or cache mismatch')
    required={f'run-{run}/{name}' for name in ('last_training.pt','model.pt','run_config.json','metrics.json')}
    if set(record['files'])!=required:raise ValueError('Incomplete or unexpected parent file specification')
    for name,digest in record['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe parent path')
        if terminal['artifact_sha256'].get(name)!=digest or sha(root/name)!=digest:
            raise ValueError('Parent artifact hash mismatch: '+name)
    directory=root/f'run-{run}'
    metadata=json.loads((directory/'run_config.json').read_text())
    metrics=json.loads((directory/'metrics.json').read_text())
    if metadata['torch']!=hardware['torch'] or metadata['cuda']!=hardware['cuda']:
        raise ValueError('Parent/new runtime software mismatch')
    if (metadata['seed']!=record['seed'] or metadata['batch_size']!=64 or
        metrics['completed_steps']!=step or metrics['optimizer_updates']+metrics['skipped_updates']!=step or
        metrics['sampled_examples']!=64*step):
        raise ValueError('Parent metadata/counters mismatch')
    return directory
