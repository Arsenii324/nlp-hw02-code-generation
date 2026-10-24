"""Stage a longer, larger ordinary/XSA pair from immutable completed source."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    tools=ROOT/'artifacts/kaggle/xsa-retry-tools';kernel=ROOT/'kaggle/xsa-retry'
    for directory in (tools,kernel):
        assert not directory.exists(),'Preserve staged inputs'
        directory.mkdir(parents=True)
    previous=json.loads((ROOT/'reports/evidence/b011_b012_prelaunch_manifest.json').read_text())
    recovered=ROOT/'artifacts/kaggle/b011-b012-v1'
    assert sha(recovered/'training_source.bin')==previous['source_sha256']
    assert sha(recovered/'run_driver.py')==previous['driver_sha256']
    shutil.copyfile(recovered/'training_source.bin',tools/'training_source.bin')
    shutil.copyfile(recovered/'run_driver.py',kernel/'run.py')
    with zipfile.ZipFile(tools/'training_source.bin') as archive:assert archive.testzip() is None
    raw=Path(os.environ.get('NLP_HW02_DATA_ROOT',ROOT.parent/'data/tinysms_qa_12m'))/'data/test/test.parquet'
    expected=next(r['sha256'] for r in json.loads((ROOT/'assignment/dataset_manifest.json').read_text())['files'] if r['path']=='data/test/test.parquet')
    assert sha(raw)==expected;shutil.copyfile(raw,tools/'test.parquet')
    checks={name:sha(tools/name) for name in ('training_source.bin','test.parquet')}
    (tools/'training_checksums.json').write_text(json.dumps(checks,indent=2))
    config={'d_model':384,'ff':1536,'heads':4,'encoder_layers':4,'decoder_layers':4}
    record={**previous,'lanes':[{'id':'B013','device':0,'seed':42,'model_config':{**config,'exclusive_self_attention':False},'optimizer':'muon_grouped','iterations':60000},
        {'id':'B014','device':1,'seed':42,'model_config':{**config,'exclusive_self_attention':True},'optimizer':'muon_grouped','iterations':60000}],
        'outer_timeout_seconds':18000,'parameter_counts':{'b013':17524224,'b014':17524224},
        'note':'Matched17.52M ordinary vs XSA,60k iterations, identical data/order/seed/optimizer/initialization recipe. Test only fixed timing, never quality tuning. Immutable B011/B012 source/driver reused; fresh training, no unverified resume. One allocation, no filler/blind retry.'}
    record.pop('predecessor_failed_kernel',None)
    for directory in (tools,kernel):(directory/'experiment-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    dataset='arsen4ikvar/nlp-hw02-xsa-retry-tools-20261002'
    (tools/'dataset-metadata.json').write_text(json.dumps({'id':dataset,'title':'NLP HW02 private XSA retry tools','licenses':[{'name':'other'}]},indent=2))
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/nlp-hw02-xsa-retry-20261002','title':'NLP HW02 larger XSA60k comparison','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[dataset],'competition_sources':[],'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2']},indent=2))
    (ROOT/'reports/evidence/xsa_retry_prelaunch_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'tools_bytes':sum(p.stat().st_size for p in tools.iterdir()),**record},indent=2))

if __name__=='__main__':main()
