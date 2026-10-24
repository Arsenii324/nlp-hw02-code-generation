"""Build immutable local private/offline T4x2 stage. Never uploads/allocates."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

WORKTREE=Path(__file__).resolve().parents[1]
PARENT_SHA='d09a8611bd6ef8ab2d0711d6999aa419efe50454d1bb12252069c59bc20c8a85'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def stage(project,tag):
    project=Path(project).resolve();frozen=project/'artifacts/kaggle/b033-b034-v1'
    metadata=frozen/'run-b034';weights=project/'artifacts/kaggle/b033-b034-weights-20261022/run-b034/model.pt'
    terminal=json.loads((frozen/'session_result.json').read_text());prior=json.loads((frozen/'experiment-manifest.json').read_text())
    if sha(weights)!=PARENT_SHA:raise ValueError('Incorrect B034 weights')
    if sha(frozen/'training_source.bin')!=terminal['training_source_sha256']:raise ValueError('Frozen source SHA mismatch')
    metrics=json.loads((metadata/'metrics.json').read_text());config=json.loads((metadata/'run_config.json').read_text())
    if metrics['selected_step']!=300000 or metrics['completed_steps']!=300000 or config['data']['train_rows']!=6000000:raise ValueError('Incorrect B034 lineage')
    for name in ('run_config.json','metrics.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
        if sha(metadata/name)!=terminal['artifact_sha256']['run-b034/'+name]:raise ValueError('Frozen metadata SHA mismatch: '+name)
    transition=prior['data_transition']['b034']
    cpu=project/'artifacts/kaggle/cache-6m-v1/cache/cache_manifest.json'
    if sha(cpu)!=transition['new_cache_manifest_sha256']:raise ValueError('Cache manifest SHA mismatch')
    cache_manifest=json.loads(cpu.read_text())
    if cache_manifest['checksums']!=transition['new_cache_checksums']:raise ValueError('Cache checksums mismatch')
    tools=WORKTREE/('artifacts/kaggle/tools-gated-b034-'+tag);kernel=WORKTREE/('kaggle/gated-b034-'+tag)
    if tools.exists() or kernel.exists():raise ValueError('Fresh immutable tag required')
    tools.mkdir(parents=True);kernel.mkdir(parents=True);parent=tools/'parent-b034';parent.mkdir()
    # APFS clone only: no silent70MB full-copy fallback on this shared low-disk Mac.
    subprocess.run(['cp','-c',str(weights),str(parent/'model.pt')],check=True)
    for name in ('run_config.json','metrics.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json'):
        shutil.copyfile(metadata/name,parent/name)
    changes=['src/nlp_hw02/model.py','src/nlp_hw02/optim.py']
    additions=['scripts/train_gated_warmstart.py','scripts/kaggle_gated_warmstart.py','scripts/build_kaggle_gated_warmstart.py']
    overlay={name:(WORKTREE/name).read_bytes() for name in changes+additions}
    for name,body in overlay.items():ast.parse(body,filename=name)
    with zipfile.ZipFile(frozen/'training_source.bin') as old:
        original={name:old.read(name) for name in old.namelist()}
    # Existing frozen cached trainer/optimizer/helpers and decoding remain byte-pinned.
    if '--skip-synthetic-check' not in original['scripts/profile_decoding_gpu.py'].decode():raise ValueError('Frozen profiler lacks real-only option')
    with zipfile.ZipFile(tools/'training_source.bin','w',zipfile.ZIP_DEFLATED) as out:
        for name in sorted(set(original)|set(overlay)):out.writestr(name,overlay.get(name,original.get(name)))
    with zipfile.ZipFile(tools/'training_source.bin') as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or archive.testzip() is not None:raise ValueError('Source archive invalid')
        for name in names:
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archive')
            if name not in overlay and archive.read(name)!=original[name]:raise ValueError('Unexpected source change')
    shutil.copyfile(WORKTREE/'scripts/kaggle_gated_warmstart.py',kernel/'run.py')
    test=project/'artifacts/kaggle/tools-data-coverage-300k-20261004a/test.parquet'
    shutil.copyfile(test,tools/'test.parquet')
    inputs={str(p.relative_to(tools)):sha(p) for p in tools.rglob('*') if p.is_file()}
    experiment={'schema':'gated_b034_warmstart_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'driver_sha256':sha(kernel/'run.py'),'input_sha256':inputs,
        'frozen_source_sha256':sha(frozen/'training_source.bin'),'changed_source_members':changes,'added_source_members':additions,
        'cache_manifest_sha256':sha(cpu),'cache_checksums':cache_manifest['checksums'],
        'parent_model_sha256':PARENT_SHA,'parent_selected_step':300000,'parent_validation_loss':metrics['best_validation_loss'],
        'lanes':[{'id':'control','device':0,'steps':45000},{'id':'treatment','device':1,'steps':30000}],
        'primary_comparison':'both best CE-selected checkpoints through absolute330k,30k new steps each',
        'secondary_comparison':'control best through345k versus treatment best through330k; useful similar wall-time control',
        'optimizer_state':'reset both lanes; grouped FP16 NS5 Muon+AdamW; no decay repeat gates/XSA',
        'rng_state':'reset both lanes seed42, same6M row index stream through30k; dropout streams may differ',
        'lr_schedule':'absolute step300k onward,6e-4*min(step/200,sqrt(200/max(step,200)))',
        'outer_timeout_seconds':18000,'tail_reserve_seconds':1500,'admission_new_steps':1000,
        'evaluation_bound_seconds':300,'planned_evaluations':3,
        'topology':{'base_depth':'4+4','effective_depth':'8+8','order':'0,0,1,1,2,2,3,3',
            'sharing':'self/cross attention aliases; separate FFNs/norms','base_branch_scale':1.,
            'repeat_branch_scale':.5,'repeat_wrapper':'h+g*(Block(h)-h)','g_init':0.,'g_weight_decay':0.,
            'expected_unique_parameters':26992168,'expected_unique_xsa_scalars':32},
        'not_full_state_continuation':True,'synthetic_fixtures':False,'network':False,
        'quality_and_180s_claim':'Unmeasured until actual remote training/oneT4 inference; stage is implementation only.'}
    (tools/'experiment-manifest.json').write_text(json.dumps(experiment,indent=2)+'\n')
    slug='nlp-hw02-tools-gated-b034-'+tag
    (tools/'dataset-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/'+slug,'title':slug.replace('-',' '),'licenses':[{'name':'other'}]},indent=2)+'\n')
    previous=json.loads((project/'kaggle/data-coverage-300k-20261004a/kernel-metadata.json').read_text())
    kernel_meta={**previous,'id':'arsen4ikvar/nlp-hw02-gated-b034-'+tag,'title':'nlp hw02 gated b034 '+tag,
        'dataset_sources':['arsen4ikvar/'+slug],
        'kernel_sources':['arsen4ikvar/nlp-hw02-cache-6m-20261004c'],
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False}
    (kernel/'kernel-metadata.json').write_text(json.dumps(kernel_meta,indent=2)+'\n')
    return {'tools':str(tools),'kernel':str(kernel),'experiment':experiment,'kernel_metadata':kernel_meta,
        'source_members':len(names),'archive_sha256':sha(tools/'training_source.bin'),'local_only':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args();print(json.dumps(stage(args.project_root,args.tag),indent=2))
