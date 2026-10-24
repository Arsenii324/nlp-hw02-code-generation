"""Freeze loop/depth or a pinned NS/Polar continuation; never launch a runtime."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from successor_driver_patch import patched_driver
from nlp_hw02.model import ModelConfig,Seq2Seq

ROOT=Path(__file__).resolve().parents[1]
BASE_SOURCE_SHA='6f69330a1493bea8246d0bd4afa8e353a92843b285d692ccb166f8323c1f0d74'
BASE_DRIVER_SHA='8eaa692d0570f73a1779bdc7dc3eed6704c01614e947722e4a1d26b0e9d6b677'
OVERLAYS=('src/nlp_hw02/model.py','src/nlp_hw02/optim.py','src/nlp_hw02/optim_batched.py','src/nlp_hw02/train.py')
ADDITIONS=('src/nlp_hw02/optim_polar.py','src/nlp_hw02/resuming.py',
    'scripts/check_looped_model.py','scripts/check_training_resumption.py','scripts/check_polar_muon.py',
    'scripts/kaggle_resume_inputs.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_record(terminal_path):
    terminal_path=Path(terminal_path);terminal=json.loads(terminal_path.read_text())
    if not terminal.get('complete') or not terminal.get('training_complete') or terminal['iterations']['b013']!=60000:
        raise ValueError('Complete 60k B013 parent required before freezing resume stage')
    expected=json.loads((ROOT/'artifacts/kaggle/xsa-retry-tools/experiment-manifest.json').read_text())
    if terminal['training_source_sha256']!=expected['source_sha256'] or terminal['driver_sha256']!=expected['driver_sha256']:
        raise ValueError('Unexpected B013 training source/driver')
    metadata_path=terminal_path.parent/'run-b013/run_config.json'
    metadata=json.loads(metadata_path.read_text())
    if sha(metadata_path)!=terminal['artifact_sha256']['run-b013/run_config.json']:
        raise ValueError('Recovered parent metadata hash mismatch')
    return {'kernel':'arsen4ikvar/nlp-hw02-xsa-retry-20261002','version':1,'run':'b013','step':60000,'seed':42,
        'terminal_sha256':sha(terminal_path),'source_sha256':terminal['training_source_sha256'],
        'driver_sha256':terminal['driver_sha256'],
        'files':{f'run-b013/{name}':terminal['artifact_sha256'][f'run-b013/{name}']
            for name in ('last_training.pt','model.pt','run_config.json','metrics.json')}},ModelConfig(**metadata['config'])


def xsa_parent_record(terminal_path,run):
    terminal_path=Path(terminal_path);terminal=json.loads(terminal_path.read_text())
    specifications={
        'b014':('arsen4ikvar/nlp-hw02-xsa-retry-20261002','xsa-retry-tools',True,False),
        'b015':('arsen4ikvar/nlp-hw02-model-followup-20261002','model-followup-tools-20261002',False,True)}
    kernel,tools,fixed,learned=specifications[run]
    expected=json.loads((ROOT/'artifacts/kaggle'/tools/'experiment-manifest.json').read_text())
    if (terminal.get('complete') is not True or terminal.get('training_complete') is not True or
        terminal['iterations'].get(run)!=60000 or
        terminal['training_source_sha256']!=expected['source_sha256'] or
        terminal['driver_sha256']!=expected['driver_sha256']):
        raise ValueError('Complete, source-pinned 60k XSA parent required: '+run)
    directory=terminal_path.parent/f'run-{run}'
    for name in ('run_config.json','metrics.json'):
        if sha(directory/name)!=terminal['artifact_sha256'][f'run-{run}/{name}']:
            raise ValueError('Recovered XSA parent metadata hash mismatch: '+run+'/'+name)
    metadata=json.loads((directory/'run_config.json').read_text())
    metrics=json.loads((directory/'metrics.json').read_text())
    config=ModelConfig(**metadata['config'])
    required=ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,
        exclusive_self_attention=fixed,learnable_xsa=learned)
    if (config!=required or metadata['torch']!='2.10.0+cu128' or metadata['cuda']!='12.8' or
        metadata['seed']!=42 or metadata['batch_size']!=64 or
        metrics['completed_steps']!=60000 or metrics['sampled_examples']!=60000*64 or
        metrics['optimizer_updates']+metrics['skipped_updates']!=60000):
        raise ValueError('XSA parent recipe/software/counters mismatch: '+run)
    return {'kernel':kernel,'version':1,'run':run,'step':60000,'seed':42,
        'terminal_sha256':sha(terminal_path),'source_sha256':terminal['training_source_sha256'],
        'driver_sha256':terminal['driver_sha256'],
        'files':{f'run-{run}/{name}':terminal['artifact_sha256'][f'run-{run}/{name}']
            for name in ('last_training.pt','model.pt','run_config.json','metrics.json')}},config


def continuation_parent_record(terminal_path,run):
    """Bind the specific ordinary/learned120k parents without downloading moments."""
    specifications={
        'b019':('arsen4ikvar/nlp-hw02-polar-resume-20261002','tools-polar-resume-20261002',False),
        'b024':('arsen4ikvar/nlp-hw02-xsa-continuation-20261003','tools-xsa-continuation-20261003',True)}
    if run not in specifications:raise ValueError('Only declared B019/B024120k parents supported')
    kernel,tools,learned=specifications[run]
    expected=json.loads((ROOT/'artifacts/kaggle'/tools/'experiment-manifest.json').read_text())
    terminal_path=Path(terminal_path);terminal=json.loads(terminal_path.read_text())
    if (terminal.get('complete') is not True or terminal.get('training_complete') is not True or
        terminal['iterations'].get(run)!=120000 or terminal['training_source_sha256']!=expected['source_sha256'] or
        terminal['driver_sha256']!=expected['driver_sha256']):
        raise ValueError('Complete, source-pinned120k parent required: '+run)
    directory=terminal_path.parent/f'run-{run}'
    for name in ('run_config.json','metrics.json'):
        if sha(directory/name)!=terminal['artifact_sha256'][f'run-{run}/{name}']:
            raise ValueError('Recovered parent metadata hash mismatch: '+run+'/'+name)
    metadata=json.loads((directory/'run_config.json').read_text());metrics=json.loads((directory/'metrics.json').read_text())
    cfg=ModelConfig(**metadata['config'])
    required=ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,learnable_xsa=learned)
    opt=metadata['optimizer']
    if (cfg!=required or metadata['torch']!='2.10.0+cu128' or metadata['cuda']!='12.8' or
        metadata['seed']!=42 or metadata['batch_size']!=64 or metadata['schedule']!='warmup=200, inverse_sqrt decay' or
        opt['kind']!='hybrid grouped FP16 Muon+AdamW' or opt['orthogonalizer']!='newton_schulz' or opt['muon_ns_steps']!=5 or
        metrics['completed_steps']!=120000 or metrics['sampled_examples']!=7680000 or
        metrics['optimizer_updates']+metrics['skipped_updates']!=120000):
        raise ValueError('120k parent recipe/software/counters mismatch: '+run)
    return {'kernel':kernel,'version':1,'run':run,'step':120000,'seed':42,
        'terminal_sha256':sha(terminal_path),'source_sha256':terminal['training_source_sha256'],
        'driver_sha256':terminal['driver_sha256'],
        'files':{f'run-{run}/{name}':terminal['artifact_sha256'][f'run-{run}/{name}']
            for name in ('last_training.pt','model.pt','run_config.json','metrics.json')}},cfg


def learned_180k_parent_record(terminal_path):
    """Bind B028's full last180k state; selected inference state remains179k."""
    terminal_path=Path(terminal_path);terminal=json.loads(terminal_path.read_text())
    expected=json.loads((ROOT/'artifacts/kaggle/tools-training-180k-20261003/experiment-manifest.json').read_text())
    run='b028';step=180000
    if (terminal.get('complete') is not True or terminal.get('training_complete') is not True or
        terminal['iterations'].get(run)!=step or terminal['training_source_sha256']!=expected['source_sha256'] or
        terminal['driver_sha256']!=expected['driver_sha256'] or terminal['cache_manifest_sha256']!=expected['cache_manifest_sha256']):
        raise ValueError('Complete source/driver/cache-pinned learned180k parent required')
    directory=terminal_path.parent/f'run-{run}'
    for name in ('run_config.json','metrics.json'):
        if sha(directory/name)!=terminal['artifact_sha256'][f'run-{run}/{name}']:
            raise ValueError('Recovered learned180k parent metadata hash mismatch: '+name)
    metadata=json.loads((directory/'run_config.json').read_text());metrics=json.loads((directory/'metrics.json').read_text())
    cfg=ModelConfig(**metadata['config']);opt=metadata['optimizer']
    required=ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,learnable_xsa=True)
    if (cfg!=required or metadata['torch']!='2.10.0+cu128' or metadata['cuda']!='12.8' or
        metadata['seed']!=42 or metadata['batch_size']!=64 or metadata['schedule']!='warmup=200, inverse_sqrt decay' or
        opt['kind']!='hybrid grouped FP16 Muon+AdamW' or opt['orthogonalizer']!='newton_schulz' or opt['muon_ns_steps']!=5 or
        metrics['completed_steps']!=step or metrics['sampled_examples']!=step*64 or
        metrics['optimizer_updates']+metrics['skipped_updates']!=step):
        raise ValueError('Learned180k parent recipe/software/counters mismatch')
    return {'kernel':'arsen4ikvar/nlp-hw02-training-180k-20261003','version':1,'run':run,'step':step,'seed':42,
        'terminal_sha256':sha(terminal_path),'source_sha256':terminal['training_source_sha256'],
        'driver_sha256':terminal['driver_sha256'],
        'files':{f'run-{run}/{name}':terminal['artifact_sha256'][f'run-{run}/{name}']
            for name in ('last_training.pt','model.pt','run_config.json','metrics.json')}},cfg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['loop','resume','adjacent','xsa-resume','long-resume','xsa-polar-resume'])
    parser.add_argument('--parent-terminal',type=Path);parser.add_argument('--tag',default='20261002')
    parser.add_argument('--fixed-parent-terminal',type=Path);parser.add_argument('--learned-parent-terminal',type=Path)
    parser.add_argument('--ordinary-parent-terminal',type=Path)
    args=parser.parse_args()
    if not args.tag.isalnum():raise ValueError('Use an alphanumeric unique stage tag')
    if (args.mode in ('resume','xsa-polar-resume'))!=(args.parent_terminal is not None):parser.error('Single-parent resume requires --parent-terminal')
    if args.mode=='xsa-resume':
        if args.fixed_parent_terminal is None or args.learned_parent_terminal is None:
            parser.error('XSA continuation requires both distinct parent terminals')
    elif args.mode=='long-resume':
        if args.ordinary_parent_terminal is None or args.learned_parent_terminal is None or args.fixed_parent_terminal is not None:
            parser.error('180k continuation requires ordinary and learned120k parents')
    elif args.fixed_parent_terminal is not None or args.learned_parent_terminal is not None:
        parser.error('XSA parent arguments require xsa-resume')
    if args.mode!='long-resume' and args.ordinary_parent_terminal is not None:
        parser.error('Ordinary parent argument requires long-resume')
    is_loop=args.mode in ('loop','adjacent')
    label={'loop':'loop-depth','resume':'polar-resume','adjacent':'adjacent-sharing','xsa-resume':'xsa-continuation','long-resume':'training-180k','xsa-polar-resume':'xsa-polar-240k'}[args.mode]+'-'+args.tag
    tools=ROOT/'artifacts/kaggle'/('tools-'+label);kernel=ROOT/'kaggle'/label
    if tools.exists() or kernel.exists():raise ValueError('Refuse to overwrite frozen staging')
    parent=None;parents={}
    if args.mode=='xsa-resume':
        fixed,fixed_cfg=xsa_parent_record(args.fixed_parent_terminal,'b014')
        learned,learned_cfg=xsa_parent_record(args.learned_parent_terminal,'b015')
        parents={'b023':fixed,'b024':learned};base=fixed_cfg
    elif args.mode=='long-resume':
        fixed,fixed_cfg=continuation_parent_record(args.ordinary_parent_terminal,'b019')
        learned,learned_cfg=continuation_parent_record(args.learned_parent_terminal,'b024')
        parents={'b027':fixed,'b028':learned};base=fixed_cfg
    elif args.mode=='resume':parent,base=parent_record(args.parent_terminal)
    elif args.mode=='xsa-polar-resume':parent,base=learned_180k_parent_record(args.parent_terminal)
    else:base=ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,
        loop_repeats=2,residual_scale=.5,loop_order='adjacent' if args.mode=='adjacent' else 'stack')
    if args.mode in ('xsa-resume','long-resume'):configs=[asdict(fixed_cfg),asdict(learned_cfg)]
    elif args.mode=='adjacent':configs=[asdict(base),{**asdict(base),'loop_share_scope':'attention'}]
    elif args.mode=='loop':configs=[asdict(base),{**asdict(base),'share_loop_weights':False}]
    else:configs=[asdict(base)]*2
    ids={'loop':['B017','B018'],'resume':['B019','B020'],'adjacent':['B021','B022'],'xsa-resume':['B023','B024'],'long-resume':['B027','B028'],'xsa-polar-resume':['B031','B032']}[args.mode]
    optimizers=['muon_grouped','muon_grouped_pe'] if args.mode in ('resume','xsa-polar-resume') else ['muon_grouped']*2
    lanes=[];counts={}
    for device,(name,cfg,optimizer) in enumerate(zip(ids,configs,optimizers)):
        lanes.append({'id':name,'device':device,'seed':42,'model_config':cfg,'optimizer':optimizer,
            'iterations':30000 if is_loop else 240000 if args.mode=='xsa-polar-resume' else 180000 if args.mode=='long-resume' else 120000,
            'resume_from_step':0 if is_loop else 180000 if args.mode=='xsa-polar-resume' else 120000 if args.mode=='long-resume' else 60000,
            'allow_orthogonalizer_change':args.mode in ('resume','xsa-polar-resume') and device==1})
        model=Seq2Seq(ModelConfig(**cfg));counts[name.lower()]=sum(p.numel() for p in model.parameters());del model
    recovered=ROOT/'artifacts/kaggle/b011-b012-v1'
    if sha(recovered/'training_source.bin')!=BASE_SOURCE_SHA or sha(recovered/'run_driver.py')!=BASE_DRIVER_SHA:
        raise ValueError('Preserved baseline source/driver identity mismatch')
    driver=patched_driver((recovered/'run_driver.py').read_text(),args.mode)
    tools.mkdir();kernel.mkdir()
    source=tools/'training_source.bin'
    additions=ADDITIONS+(('scripts/check_adjacent_sharing.py',) if args.mode=='adjacent' else ())
    if args.mode in ('xsa-resume','long-resume'):additions+=('scripts/check_xsa_resumption.py',)
    with zipfile.ZipFile(recovered/'training_source.bin') as old,zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archived path')
            new.writestr(name,(ROOT/name).read_bytes() if name in OVERLAYS else old.read(name))
        for name in additions:
            if name in old.namelist():raise ValueError('Addition already exists in preserved source')
            new.writestr(name,(ROOT/name).read_bytes())
    with zipfile.ZipFile(source) as new,zipfile.ZipFile(recovered/'training_source.bin') as old:
        assert new.testzip() is None and set(new.namelist())==set(old.namelist())|set(additions)
        changed=[name for name in old.namelist() if old.read(name)!=new.read(name)]
        assert set(changed)==set(OVERLAYS)
        members={name:hashlib.sha256(new.read(name)).hexdigest() for name in new.namelist()}
    (kernel/'run.py').write_text(driver)
    record=json.loads((ROOT/'artifacts/kaggle/xsa-retry-tools/experiment-manifest.json').read_text())
    record.update({'lanes':lanes,'source_sha256':sha(source),'driver_sha256':sha(kernel/'run.py'),
        'parameter_counts':counts,'GPU_sessions_concurrent_limit':1,
        'changed_source_members':list(OVERLAYS),'added_source_members':list(additions),
        'source_member_sha256':members,'base_source_sha256':BASE_SOURCE_SHA,'base_driver_sha256':BASE_DRIVER_SHA,
        'new_iterations_per_lane':30000 if is_loop else 60000,'startup_bound_seconds':240,
        'note':'Unlaunched finite successor. Available GPU slot required; no third kernel. Distinct fresh outputs, actual1000new-step admission, complete-only evaluations, near-finish priority, no filler/retries. '+
            ('Adjacent full-block versus shared self/cross-attention and cloned independent FFNs/norms; same0.5branches/initial outputs/30k data, different capacity/optimizer cost.' if args.mode=='adjacent' else
             'Whole-stack2 shared versus cloned8+8, same0.5residual scales/initial outputs/30k data; not equal capacity.' if args.mode=='loop' else
             'B014 fixed and B015 learned XSA60k last weights/gates/moments/scaler/RNG/absolute schedule;60k new to120k; groupedNS unchanged, priorbest preserved.' if args.mode=='xsa-resume' else
             'B019 ordinary and B024 learnedXSA120k full last weights/gates/moments/scaler/RNG/absolute inverse-sqrt schedule;60k new to180k; groupedNS unchanged, priorbest preserved.' if args.mode=='long-resume' else
             'Same B028learnedXSA180k full last weights/gates/moments/scaler/RNG/absolute schedule,60k new to240k; NS5 versus pinnedPE5FP16 only, explicit method change; selected179k priorbest preserved, optionaltelemetry failure does not kill workers.' if args.mode=='xsa-polar-resume' else
             'Same B01360k last weights/momentum/scaler/RNG/absolute schedule,60k new; NS5 versus pinnedPE5FP16 only. Explicit orthogonalizer change, priorbest preserved.')})
    record.pop('concurrency_authorized_until_UTC',None)
    if parent:record['resume_parent']=parent
    if parents:record['resume_parents']=parents
    for directory in (tools,kernel):(directory/'experiment-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    test=ROOT/'artifacts/kaggle/xsa-retry-tools/test.parquet'
    expected=json.loads((ROOT/'artifacts/kaggle/xsa-retry-tools/training_checksums.json').read_text())['test.parquet']
    assert sha(test)==expected;shutil.copyfile(test,tools/'test.parquet')
    (tools/'training_checksums.json').write_text(json.dumps({name:sha(tools/name) for name in ('training_source.bin','test.parquet')},indent=2)+'\n')
    dataset='arsen4ikvar/nlp-hw02-tools-'+label;identity='arsen4ikvar/nlp-hw02-'+label
    title='NLP HW02 '+label.replace('-',' ').title()
    (tools/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':title+' Tools','licenses':[{'name':'other'}]},indent=2)+'\n')
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':identity,'title':title,
        'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,
        'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':[dataset],'competition_sources':[],
        'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2']+([parent['kernel']] if parent else [])+
            [value['kernel'] for value in parents.values()],
        **({'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
            'docker_image_pinning_type':'original'} if args.mode in ('xsa-resume','long-resume','xsa-polar-resume') else {})},indent=2)+'\n')
    (ROOT/'reports/evidence'/f'{label}_prelaunch_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'mode':args.mode,'tools':str(tools),'kernel':str(kernel),'tools_bytes':sum(p.stat().st_size for p in tools.iterdir()),
        'source_sha256':record['source_sha256'],'driver_sha256':record['driver_sha256'],'parameters':counts,'launched':False}))


if __name__=='__main__':main()
