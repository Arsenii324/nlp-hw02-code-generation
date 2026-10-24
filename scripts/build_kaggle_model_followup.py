"""Freeze the second GPU pair from an immutable source plus explicit overlays."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OVERLAYS=('src/nlp_hw02/model.py','src/nlp_hw02/optim.py','src/nlp_hw02/train.py')
CHECK='scripts/check_model_followup.py'
BASE_SHA='6f69330a1493bea8246d0bd4afa8e353a92843b285d692ccb166f8323c1f0d74'
DRIVER_SHA='8eaa692d0570f73a1779bdc7dc3eed6704c01614e947722e4a1d26b0e9d6b677'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    tools=ROOT/'artifacts/kaggle/model-followup-tools-20261002'
    kernel=ROOT/'kaggle/model-followup-20261002'
    assert not tools.exists() and not kernel.exists(),'Refuse to overwrite frozen staging'
    recovered=ROOT/'artifacts/kaggle/b011-b012-v1'
    assert sha(recovered/'training_source.bin')==BASE_SHA
    assert sha(recovered/'run_driver.py')==DRIVER_SHA
    tools.mkdir();kernel.mkdir()
    source=tools/'training_source.bin'
    with zipfile.ZipFile(recovered/'training_source.bin') as old, zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            content=(ROOT/name).read_bytes() if name in OVERLAYS else old.read(name)
            new.writestr(name,content)
        new.writestr(CHECK,(ROOT/CHECK).read_bytes())
    with zipfile.ZipFile(source) as new,zipfile.ZipFile(recovered/'training_source.bin') as old:
        assert new.testzip() is None and set(new.namelist())==set(old.namelist())|{CHECK}
        changed=[name for name in old.namelist() if old.read(name)!=new.read(name)]
        assert set(changed)==set(OVERLAYS)
        members={name:hashlib.sha256(new.read(name)).hexdigest() for name in new.namelist()}
    driver=(recovered/'run_driver.py').read_text()
    preflight="    subprocess.run([sys.executable,str(source/'scripts/check_baseline.py')],env=env,check=True,timeout=90)"
    assert driver.count(preflight)==1
    driver=driver.replace(preflight,preflight+"\n    subprocess.run([sys.executable,str(source/'scripts/check_model_followup.py')],env=env,check=True,timeout=90)")
    marker="                      'array_loading_forwarded':False,'exit_observed_seconds':None})"
    assert driver.count(marker)==1
    driver=driver.replace(marker,"                      'last_gate_forwarded':-1,\n"+marker)
    marker="                        if record.get('event') in ('initial_validation','validation') and step>lane['last_validation_forwarded']:"
    assert driver.count(marker)==1
    hook="                        if record.get('event')=='xsa_gates' and step>lane['last_gate_forwarded']:\n                            emit({'event':'lane_xsa_gates','run':lane['name'],'step':step,'tanh_alpha':record['tanh_alpha']})\n                            lane['last_gate_forwarded']=step\n"
    driver=driver.replace(marker,hook+marker)
    compile(driver,'model-followup-driver','exec');(kernel/'run.py').write_text(driver)
    previous=json.loads((ROOT/'artifacts/kaggle/xsa-retry-tools/experiment-manifest.json').read_text())
    config={'d_model':384,'ff':1536,'heads':4,'encoder_layers':4,'decoder_layers':4}
    record={**previous,'source_sha256':sha(source),'driver_sha256':sha(kernel/'run.py'),
        'base_source_sha256':BASE_SHA,'base_driver_sha256':DRIVER_SHA,
        'source_member_sha256':members,'changed_source_members':list(OVERLAYS),
        'added_source_members':[CHECK],
        'lanes':[{'id':'B015','device':0,'seed':42,'model_config':{**config,'learnable_xsa':True},'optimizer':'muon_grouped','iterations':60000},
                 {'id':'B016','device':1,'seed':42,'model_config':{**config,'activation':'swiglu','ff':1024},'optimizer':'muon_grouped','iterations':60000}],
        'GPU_sessions_concurrent_limit':2,'outer_timeout_seconds':18000,
        'parameter_counts':{'b015':17524256,'b016':17528320},
        'control_kernel':'arsen4ikvar/nlp-hw02-xsa-retry-20261002/1',
        'control_run':'B013',
        'note':'User authorized second concurrent GPU kernel until00:00UTC reset. LearnableXSA zero32gates/no-decayAdam; ordinarySwiGLUFF1024 matchedFFmatrixbudget. Same cache/order/seed/groupedMuon/60k. Fresh, no extra targets/test-quality tuning. Source overlay limited to model/optimizer/gate diagnostics plus check; immutable driver gains CPU preflight and gate forwarding only. Four complete-only evals/1500s tail, healthy near-finish priority.'}
    for directory in (tools,kernel):(directory/'experiment-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    # Reuse already-verified question-only Parquet; no raw/train array copy.
    test=ROOT/'artifacts/kaggle/xsa-retry-tools/test.parquet'
    expected=json.loads((ROOT/'artifacts/kaggle/xsa-retry-tools/training_checksums.json').read_text())['test.parquet']
    assert sha(test)==expected;shutil.copyfile(test,tools/'test.parquet')
    checks={name:sha(tools/name) for name in ('training_source.bin','test.parquet')}
    (tools/'training_checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    dataset='arsen4ikvar/nlp-hw02-model-followup-tools-20261002'
    identity='arsen4ikvar/nlp-hw02-model-followup-20261002'
    (tools/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':'NLP HW02 private model followup tools','licenses':[{'name':'other'}]},indent=2)+'\n')
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':identity,'title':'NLP HW02 Model Followup 20261002',
        'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,
        'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':[dataset],'competition_sources':[],'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2']},indent=2)+'\n')
    assert {p.name for p in tools.iterdir()}=={'training_source.bin','test.parquet','training_checksums.json','experiment-manifest.json','dataset-metadata.json'}
    (ROOT/'reports/evidence/model_followup_prelaunch_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'tools':str(tools),'kernel':str(kernel),'tools_bytes':sum(p.stat().st_size for p in tools.iterdir()),
        'source_sha256':record['source_sha256'],'driver_sha256':record['driver_sha256'],
        'parameters':record['parameter_counts'],'changed_source_members':record['changed_source_members']}))


if __name__=='__main__':main()
