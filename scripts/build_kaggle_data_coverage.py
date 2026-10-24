"""SpecificB031240k/verified6M-cache continuation stage; no allocation here."""
from datetime import datetime,timezone
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(body,old,new):
    if body.count(old)!=1:raise ValueError('Unexpected frozen driver marker: '+old[:80])
    return body.replace(old,new)


def driver(body):
    body=once(body,"    tools_files=list(Path('/kaggle/input').rglob('training_checksums.json'))","    tools_files=[p for p in Path('/kaggle/input').rglob('training_checksums.json') if all((p.parent/n).is_file() for n in ('training_source.bin','data-transition.json','experiment-manifest.json'))]")
    start=body.index("    manifests=list(Path('/kaggle/input').rglob('cache_manifest.json'))")
    end=body.index("    env={**os.environ",start)
    replacement='''    all_manifests=list(Path('/kaggle/input').rglob('cache_manifest.json'))
    caches={}
    for name,expected in experiment['cache_manifests_sha256'].items():
        candidates=[p for p in all_manifests if hashlib.sha256(p.read_bytes()).hexdigest()==expected]
        assert len(candidates)==1,'Exactly one content-pinned cache for '+name
        caches[name]=candidates[0].parent
    cache=caches['b033'];new_cache=caches['b034'];manifests=[cache/'cache_manifest.json']
    cache_digest=hashlib.sha256(manifests[0].read_bytes()).hexdigest()
    assert cache_digest==experiment['cache_manifest_sha256']
    RUN_STATE.update({'training_source_sha256':hashes['training_source.bin'],
        'cache_manifest_sha256':cache_digest,'cache_manifests_sha256':experiment['cache_manifests_sha256'],
        'cpu_kernel_version':experiment['cpu_kernel_version'],'cpu_kernel_versions':experiment['cpu_kernel_versions'],
        'controlled_data_transition':experiment['data_transition']})
    import numpy as np
    for name,root in caches.items():
        cm=json.loads((root/'cache_manifest.json').read_text());assert cm['complete']
        rows=2000000 if name=='b033' else 6000000
        for filename,expected in cm['checksums'].items():
            if filename.endswith('.json'):assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==expected,filename
        for filename,width,dtype in [('train_source.npy',256,np.uint16),('train_target.npy',192,np.uint16),('train_row_ids.npy',2,np.uint32)]:
            a=np.load(root/filename,mmap_mode='r');assert a.shape==(rows,width) and a.dtype==dtype;del a
    verified=subprocess.run([sys.executable,str(source/'scripts/verify_training_caches.py'),
        '--old',str(cache),'--new',str(new_cache),'--manifest',str(tools/'experiment-manifest.json')],
        capture_output=True,text=True,check=True,timeout=150)
    emit(json.loads(verified.stdout))
    from nlp_hw02.training_arrays import available_memory_bytes,array_resident_bytes
    sizes={n:array_resident_bytes(p) for n,p in caches.items()};available=available_memory_bytes()
    required=2*sum(sizes.values())+2*1024**3
    assert available>=required,'Insufficient RAM for both resident caches/pagecache/headroom'
    emit({'event':'resident_arrays_admission','array_bytes_by_lane':sizes,'available_bytes':available,
          'required_bytes':required,'reserved_copies':2,'note':'Two different immutable caches; arrays stay remote.'})
'''
    body=body[:start]+replacement+body[end:]
    start=body.index("    subprocess.run([sys.executable,str(source/'scripts/check_baseline.py')]")
    end=body.index('    parent_directory=None',start)
    body=body[:start]+'    selected_muon=None\n'+body[end:]
    marker="    emit({'event':'resume_parent_verified','parent':experiment['resume_parent'],'seconds':time.monotonic()-STARTED})"
    extra="\n    shutil.copyfile(tools/'data-transition.json',WORKING/'data-transition.json')\n    from nlp_hw02.data_transition import validate_transition\n    _,transition_record=validate_transition(tools/'data-transition.json',parent_directory,new_cache)\n    emit({'event':'controlled_data_transition_verified',**transition_record})"
    body=once(body,marker,marker+extra)
    body=once(body,"        run=WORKING/f'run-{name}'","        lane_cache=caches[name]\n        assert spec['optimizer']=='muon_grouped' and not spec['allow_orthogonalizer_change']\n        run=WORKING/f'run-{name}'")
    body=once(body,"data_dir={str(cache)!r},max_steps=","data_dir={str(lane_cache)!r},max_steps=")
    marker='        suffix=f",resume_dir={str(parent_directory)!r},allow_orthogonalizer_change={spec.get(\'allow_orthogonalizer_change\',False)!r}"'
    replacement='''        assert resumed_step==240000 and iterations==300000
        transition_path=str(tools/'data-transition.json') if name=='b034' else None
        suffix=f",resume_dir={str(parent_directory)!r},allow_orthogonalizer_change=False,data_transition_manifest={transition_path!r},expected_initial_validation={experiment['parent_last_validation_loss']!r}"
'''.rstrip()
    body=once(body,marker,replacement)
    body=once(body,"if diagnostic in ('xsa_gates','loop_norms','update_rms')", "if diagnostic in ('xsa_gates','loop_norms','update_rms','parent_initial_validation_verified')")
    body=once(body,"            'cache_manifest_sha256':hashlib.sha256(manifests[0].read_bytes()).hexdigest(),", "            'cache_manifest_sha256':cache_digest,'cache_manifests_sha256':experiment['cache_manifests_sha256'],\n            'controlled_data_transition':experiment['data_transition'],")
    body=once(body,"               WORKING/'experiment-manifest.json',WORKING/'run_driver.py']", "               WORKING/'experiment-manifest.json',WORKING/'run_driver.py',WORKING/'data-transition.json']")
    body=once(body,"'--run',str(run),'--output',str(out),'--test-parquet',str(tools/'test.parquet')],","'--run',str(run),'--output',str(out),'--test-parquet',str(tools/'test.parquet'),'--skip-synthetic-check'],")
    ast.parse(body)
    if 'subprocess.run([sys.executable,str(source/\'scripts/check_' in body:raise ValueError('Fixture execution retained')
    return body


def stage(project,tag='20261004'):
    project=Path(project);oldroot=project/'artifacts/kaggle/b031-b032-v1';cpu=project/'artifacts/kaggle/cache-6m-v1'
    terminal=json.loads((oldroot/'session_result.json').read_text());meta=json.loads((oldroot/'run-b031/run_config.json').read_text());metrics=json.loads((oldroot/'run-b031/metrics.json').read_text())
    if (not terminal['complete'] or not terminal['training_complete'] or terminal['iterations']['b031']!=240000 or
        terminal['training_source_sha256']!='6c840a501233f425099b6d9a313f532f66800591aa15c7d6030b166cf0465dea' or
        terminal['driver_sha256']!='86414d20d5988fc56f7eaaa21cdcb895b0946cd260189db0316a3b45c0752148' or
        sha(oldroot/'training_source.bin')!=terminal['training_source_sha256'] or sha(oldroot/'run_driver.py')!=terminal['driver_sha256']):raise ValueError('Wrong frozenB031 parent')
    for n in ('run_config.json','metrics.json','data_stats.json'):
        if sha(oldroot/'run-b031'/n)!=terminal['artifact_sha256']['run-b031/'+n]:raise ValueError('Parent metadata hash mismatch')
    required=json.loads((oldroot/'experiment-manifest.json').read_text())['lanes'][0]['model_config']
    opt=meta['optimizer']
    if (meta['config']!=required or meta['parameters']!=17524256 or meta['torch']!='2.10.0+cu128' or
        meta['cuda']!='12.8' or meta['seed']!=42 or meta['batch_size']!=64 or
        meta['schedule']!='warmup=200, inverse_sqrt decay' or opt['kind']!='hybrid grouped FP16 Muon+AdamW' or
        opt['orthogonalizer']!='newton_schulz' or opt['muon_ns_steps']!=5 or metrics['completed_steps']!=240000 or
        metrics['sampled_examples']!=15360000 or metrics['optimizer_updates']+metrics['skipped_updates']!=240000):raise ValueError('Parent recipe/counts mismatch')
    prepared=json.loads((cpu/'prepare_result.json').read_text());coverage=json.loads((cpu/'cache/coverage_audit.json').read_text())
    if not prepared['complete'] or not prepared['cache_ready'] or coverage!=prepared['coverage'] or coverage['new_rows_not_in_old']!=4000000 or coverage['intersection_rows']!=2000000:raise ValueError('CPU cache not ready')
    for n,h in prepared['artifact_sha256'].items():
        if sha(cpu/n)!=h:raise ValueError('CPU artifact hash mismatch')
    if prepared['source_sha256']!='7f6e71dab848c437d2b5dda392954c677a15a34585efa6a1926c2e741cea5a0d':raise ValueError('WrongCPUsource')
    parent={'kernel':'arsen4ikvar/nlp-hw02-xsa-polar-240k-20261003a','version':1,'run':'b031','step':240000,'seed':42,
        'terminal_sha256':sha(oldroot/'session_result.json'),'source_sha256':terminal['training_source_sha256'],'driver_sha256':terminal['driver_sha256'],
        'files':{'run-b031/'+n:terminal['artifact_sha256']['run-b031/'+n] for n in ('last_training.pt','model.pt','run_config.json','metrics.json')}}
    new_cm=json.loads((cpu/'cache/cache_manifest.json').read_text())
    transition={'schema':'nlp_hw02_controlled_cache_transition_v1','step':240000,'seed':42,'batch_size':64,
        'parent_file_sha256':{n:terminal['artifact_sha256']['run-b031/'+n] for n in ('last_training.pt','model.pt','run_config.json','metrics.json','data_stats.json')},
        'old_cache_manifest_sha256':terminal['cache_manifest_sha256'],'new_cache_manifest_sha256':sha(cpu/'cache/cache_manifest.json'),
        'CPU_terminal_sha256':sha(cpu/'prepare_result.json'),'CPU_source_sha256':prepared['source_sha256'],
        'new_cache_checksums':new_cm['checksums'],'new_small_file_sha256':{n:sha(cpu/'cache'/n) for n in ('data_stats.json','coverage_audit.json','validation.json','validation_legacy.json','tokenizer.json')}}
    tools=project/('artifacts/kaggle/tools-data-coverage-300k-'+tag);kernel=project/('kaggle/data-coverage-300k-'+tag)
    if tools.exists() or kernel.exists():raise ValueError('Fresh immutable stage required')
    tools.mkdir(parents=True);kernel.mkdir(parents=True)
    overlays=('src/nlp_hw02/train.py','src/nlp_hw02/data_transition.py','scripts/profile_decoding_gpu.py')
    with zipfile.ZipFile(oldroot/'training_source.bin') as z,zipfile.ZipFile(tools/'training_source.bin','w',zipfile.ZIP_DEFLATED) as out:
        for n in z.namelist():out.writestr(n,(ROOT/n).read_bytes() if n in overlays else z.read(n))
        out.writestr(overlays[1],(ROOT/overlays[1]).read_bytes())
        out.writestr('scripts/verify_training_caches.py',(ROOT/'scripts/verify_training_caches.py').read_bytes())
    with zipfile.ZipFile(tools/'training_source.bin') as z:
        names=z.namelist()
        if len(names)!=41 or len(names)!=len(set(names)) or z.testzip() is not None:raise ValueError('Archive members/CRC mismatch')
        for n in names:
            if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('Unsafe source member')
    (kernel/'run.py').write_text(driver((oldroot/'run_driver.py').read_text()))
    (tools/'data-transition.json').write_text(json.dumps(transition,indent=2)+'\n')
    shutil.copyfile(project/'artifacts/kaggle/tools-xsa-polar-240k-20261003a/test.parquet',tools/'test.parquet')
    checksums={n:sha(tools/n) for n in ('training_source.bin','data-transition.json','test.parquet')};(tools/'training_checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    experiment={'source_sha256':checksums['training_source.bin'],'driver_sha256':sha(kernel/'run.py'),'cpu_kernel_version':2,'cpu_kernel_versions':{'b033':2,'b034':1},
        'cache_manifest_sha256':terminal['cache_manifest_sha256'],'cache_manifests_sha256':{'b033':terminal['cache_manifest_sha256'],'b034':transition['new_cache_manifest_sha256']},
        'lanes':[{'id':n,'device':i,'seed':42,'model_config':meta['config'],'optimizer':'muon_grouped','iterations':300000,'resume_from_step':240000,'allow_orthogonalizer_change':False} for i,n in enumerate(('B033','B034'))],
        'batch_size':64,'outer_timeout_seconds':18000,'GPU_sessions_concurrent_limit':1,'include_b001_timing':False,'profile_decoding':True,
        'canonical_bundle_timing':False,'resident_training_arrays':True,'evaluation_bound_seconds':300,'planned_evaluation_count':4,'tail_reserve_seconds':1500,
        'parameter_counts':{'b033':17524256,'b034':17524256},'resume_parent':parent,'parent_last_validation_loss':metrics['final_validation_loss'],
        'data_transition':{'b033':None,'b034':transition},'changed_source_members':['src/nlp_hw02/train.py','scripts/profile_decoding_gpu.py'],'added_source_members':['src/nlp_hw02/data_transition.py','scripts/verify_training_caches.py'],
        'note':'SameB031last240k fullstate/NS5/absoluteLR to300k,2Mcontrolvsnested6Mintervention. Future example mapping changes onlyintervention. No syntheticfixture execution; actualparent/hardware/initialCE/new-step/RAMgates. No localweights/optimizer/arrays or submission.'}
    (tools/'experiment-manifest.json').write_text(json.dumps(experiment,indent=2)+'\n')
    slug='nlp-hw02-tools-data-coverage-300k-'+tag;(tools/'dataset-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/'+slug,'title':slug.replace('-',' '),'licenses':[{'name':'other'}]},indent=2)+'\n')
    slug='nlp-hw02-data-coverage-300k-'+tag;metadata={'id':'arsen4ikvar/'+slug,'title':slug.replace('-',' '),'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':['arsen4ikvar/nlp-hw02-tools-data-coverage-300k-'+tag],'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2','arsen4ikvar/nlp-hw02-cache-6m-20261004c',parent['kernel']],'competition_sources':[],'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','docker_image_pinning_type':'original'}
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return {'utc':datetime.now(timezone.utc).isoformat(),'tools':str(tools),'kernel':str(kernel),'experiment':experiment,'transition':transition,'metadata':metadata}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--tag',default='20261004');a=p.parse_args();print(json.dumps(stage(a.project_root,a.tag),indent=2))
