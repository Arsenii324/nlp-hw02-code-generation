"""Recover bounded diagnostics only for the declared two-cache300k experiment."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate(root,manifest,kernel):
    root=Path(root);m=json.loads(Path(manifest).read_text());t=json.loads((root/'session_result.json').read_text())
    if not t['complete'] or not t['training_complete'] or not t['inference_complete']:raise ValueError('Run incomplete; inspect logs without promotion')
    if t['iterations']!={'b033':300000,'b034':300000}:raise ValueError('Wrong completed horizon')
    if (t['driver_sha256']!=m['driver_sha256'] or t['training_source_sha256']!=m['source_sha256'] or
        t['cache_manifest_sha256']!=m['cache_manifest_sha256'] or t['cache_manifests_sha256']!=m['cache_manifests_sha256'] or
        t['controlled_data_transition']!=m['data_transition'] or t['experiment_manifest_sha256']!=sha(manifest)):
        raise ValueError('Terminal provenance mismatch')
    verified=[]
    for name in t['artifact_sha256']:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe terminal artifact name')
    required={'training_source.bin','run_driver.py','experiment-manifest.json','data-transition.json','session_events.jsonl'}
    for name in ('b033','b034'):
        required.update('run-'+name+'/'+f for f in ('run_config.json','metrics.json','train.jsonl','validation.json'))
        required.update('decoding-'+name+'/'+f for f in ('decoding_profile.json','beam_test_timing.json','beam_test_predictions.jsonl','beam4-alpha06/validation.json','beam4-alpha06/validation_predictions.jsonl'))
    if not required.issubset(t['artifact_sha256']) or not all((root/n).is_file() for n in required):raise ValueError('Missing/hash-unbound consumed diagnostics')
    for name,expected in t['artifact_sha256'].items():
        p=root/name
        if p.is_file():
            if sha(p)!=expected:raise ValueError('Artifact hash mismatch: '+name)
            verified.append(name)
    if any(p.suffix in ('.pt','.npy') for p in root.rglob('*')):raise ValueError('Large-file recovery forbidden')
    if sha(root/'training_source.bin')!=m['source_sha256'] or sha(root/'run_driver.py')!=m['driver_sha256']:
        raise ValueError('Recovered source/driver mismatch')
    with zipfile.ZipFile(root/'training_source.bin') as z:
        if z.testzip() is not None:raise ValueError('Source CRC failure')
        names=z.namelist()
        if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe source members')
        beam_sha=hashlib.sha256(z.read('src/nlp_hw02/beam.py')).hexdigest()
        profiler_sha=hashlib.sha256(z.read('scripts/profile_decoding_gpu.py')).hexdigest()
    runs={}
    for spec in m['lanes']:
        name=spec['id'].lower();directory=root/('run-'+name)
        meta=json.loads((directory/'run_config.json').read_text());metrics=json.loads((directory/'metrics.json').read_text())
        expected_rows=2000000 if name=='b033' else 6000000
        if (meta['config']!=spec['model_config'] or meta['parameters']!=17524256 or meta['seed']!=42 or meta['batch_size']!=64 or
            meta['torch']!='2.10.0+cu128' or meta['cuda']!='12.8' or meta['data']['train_rows']!=expected_rows or
            meta['optimizer']['orthogonalizer']!='newton_schulz' or meta['schedule']!='warmup=200, inverse_sqrt decay'):
            raise ValueError('Lane recipe/data/software mismatch')
        if (metrics['completed_steps']!=300000 or metrics['sampled_examples']!=19200000 or metrics['resumed_from_step']!=240000 or
            metrics['new_iterations']!=60000 or metrics['optimizer_updates']+metrics['skipped_updates']!=300000 or
            not 0<metrics['selected_step']<=300000 or meta['resumption']['step']!=240000):raise ValueError('Lane counters/roles mismatch')
        transition=meta['resumption'].get('controlled_data_transition')
        if name=='b033' and transition is not None:raise ValueError('Control underwent a data intervention')
        if name=='b034' and (not transition or transition['new_cache_manifest_sha256']!=m['cache_manifests_sha256'][name] or
            transition['absolute_transition_step']!=240000 or transition['exact_training_example_continuation'] is not False):
            raise ValueError('Intervention provenance absent')
        profile=root/('decoding-'+name);p=json.loads((profile/'decoding_profile.json').read_text());timing=json.loads((profile/'beam_test_timing.json').read_text())
        if (p.get('synthetic_fixture_executed') is not False or p['checkpoint_sha256']!=t['artifact_sha256']['run-'+name+'/model.pt'] or p['beam_source_sha256']!=beam_sha or p['profiler_sha256']!=profiler_sha or
            p['selected_step']!=metrics['selected_step'] or p['gpu_count']!=1 or 'T4' not in p['gpu'] or
            timing['rows']!=1919 or timing['gpu_count']!=1 or timing['seconds']>180 or timing['beam_width']!=4 or
            timing['batch_size']!=256 or timing['length_penalty']!=.6):raise ValueError('Profile identity/budget mismatch')
        test=[json.loads(line) for line in (profile/'beam_test_predictions.jsonl').read_text().splitlines()]
        val=[json.loads(line) for line in (profile/'beam4-alpha06/validation_predictions.jsonl').read_text().splitlines()]
        if [r['row_index'] for r in test]!=list(range(1919)) or len(val)!=1792 or not (all(isinstance(r['code'],str) for r in test) and all(isinstance(r['prediction'],str) for r in val)):raise ValueError('Prediction rows/order mismatch')
        validation=profile/'beam4-alpha06/validation.json'
        if sha(validation)!='73ac2a8b8ea0c16c2eec38c219b62f46857cdfe3de7fab3cab64754aa8b4c24e':raise ValueError('Fixed validation bytes differ')
        full=json.loads(validation.read_text());pilot=metrics['validation_generation_indices']
        if len(pilot)!=256 or len(set(pilot))!=256 or any(type(i) is not int or not 0<=i<2048 for i in pilot):raise ValueError('Invalid pilot indices')
        selected=[r for i,r in enumerate(full) if i not in set(pilot)]
        if [{k:v for k,v in r.items() if k!='prediction'} for r in val]!=selected:raise ValueError('Validation question/reference/order mismatch')
        events=[json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
        gate=[r for r in events if r['event']=='parent_initial_validation_verified']
        if len(gate)!=1 or gate[0]['step']!=240000 or gate[0]['expected']!=m['parent_last_validation_loss']:raise ValueError('InitialCEgate evidence missing')
        runs[name]={'completed_steps':300000,'selected_step':metrics['selected_step'],'primary_CE':metrics['best_validation_loss'],
                    'train_rows':expected_rows,'new_draws':3840000,'beam_T4_seconds':timing['seconds'],
                    'optimizer_updates':metrics['optimizer_updates'],'skipped_updates':metrics['skipped_updates'],
                    'checkpoint_sha256_remote_declared_unrecovered':p['checkpoint_sha256']}
    if (root/'decoding-b033/beam4-alpha06/validation.json').read_bytes()!=(root/'decoding-b034/beam4-alpha06/validation.json').read_bytes():raise ValueError('Quality split differs between lanes')
    return {'utc':datetime.now(timezone.utc).isoformat(),'kernel':kernel,'kernel_version':int(kernel.rsplit('/',1)[1]),'verified_artifacts':verified,'terminal_sha256':sha(root/'session_result.json'),
            'complete':True,'runs':runs,'no_weights_optimizer_arrays_downloaded':True,'numeric_quality_review_pending':True,'no_submission':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kernel',required=True);p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--evidence',required=True);p.add_argument('--launch',required=True);a=p.parse_args()
    if not re.fullmatch(r'[\w-]+/[\w-]+/[1-9][0-9]*',a.kernel):p.error('Explicit owner/slug/version required')
    launch=json.loads(Path(a.launch).read_text())
    if launch['kernel']!=a.kernel or launch['experiment_manifest_sha256']!=sha(a.manifest):raise ValueError('Recovery differs from recorded launch')
    root=Path(a.root)
    if root.exists():raise ValueError('Fresh recovery root required')
    command=['/Users/a2mogus/anaconda3/bin/kaggle','kernels','output',a.kernel,'--file-pattern',r'\.(json|jsonl|log|py)$|training_source\.bin$','--page-size','200','-q','-p',str(root)]
    subprocess.run(command,check=True,timeout=300)
    result=validate(root,a.manifest,a.kernel);Path(a.evidence).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
