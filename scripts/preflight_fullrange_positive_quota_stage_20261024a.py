"""Static actual-stage metadata/hash readback; no runtime/model or provider calls."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/kaggle/fullrange-cooldown-20261024a'
TOOLS = STAGE / 'tools'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
build = json.loads((STAGE/'build_receipt.json').read_text())
manifest_path = TOOLS/'fullrange_bucket_cooldown_manifest_a.json'
m = json.loads(manifest_path.read_text())
assert len(build['tools_sha256']) == 45
assert all(sha(TOOLS/name) == digest for name,digest in build['tools_sha256'].items())
assert all(sha(TOOLS/name) == digest for name,digest in m['input_sha256'].items())
assert sha(STAGE/'kernel/run.py') == build['driver_sha256'] == 'fb7b52c0ed3d823b2710db8dcef918cf807ef169770ed7f56b3be4183de83f08'
assert sha(TOOLS/'fullrange_bucket_cooldown_source.bin') == '6adf394525266ebeb91f5c2fc168cd730458a067b106cd60e1a363157e7e5a4c'
with zipfile.ZipFile(TOOLS/'fullrange_bucket_cooldown_source.bin') as archive:
    members = m['source_members_sha256']
    assert len(members) == len(archive.namelist()) == len(set(archive.namelist())) == 80
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name,digest in members.items())
    assert archive.read('scripts/kaggle_fullrange_bucket_cooldown_c.py') == (STAGE/'kernel/run.py').read_bytes()
assert m['bounds'] == {'platform':28800,'parent':28740,'work':27240,'worker':27235,'cleanup':27600,'hash':28695,'tail':1500}
assert len(m['runs']) == 2
for role in m['runs']:
    c = json.loads((TOOLS/role['contract_path']).read_text())
    assert c['new_steps'] == 196608 and c['rows'] == 11811850 and c['parent_absolute_step'] == 345000
    assert c['parent_artifact_sha256']['last_training.pt'] == 'f9d4f20ca116b4d71089cbafeecab9c6cfb210752bd572e8da5bcca9627cfb0d'
    assert c['parent_artifact_sha256']['model.pt'] == 'eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9'
    assert c['runtime_archive_sha256'] == build['runtime_archive_sha256'] and c['runtime_source_sha256'] == members
    assert c['cooldown_plan']['root_cap_NEW_iterations'] == 196608
    assert c['bounds'] == m['bounds']
policy = m['actual_quota_admission']['admission_policy']
assert policy == build['admission_policy']
assert policy['mode'] == 'user_authorized_positive_quota_overrun'
assert policy['actual_unreserved_seconds'] > 0
assert policy['current_quota_funds_declared_cap_and_reserves'] is False
assert policy['proof_quota_reserve_funded_by_current_balance'] is False
assert policy['provider_acceptance_established'] is False
quota = json.loads((TOOLS/'receipts/root_resource_snapshot.json').read_text())
g = quota['gpuQuota']
actual = float(g['totalTimeAllowed'][:-1])-float(g['timeUsed'][:-1])-float(g['timeReserved'][:-1])
assert actual == policy['actual_unreserved_seconds']
auth = json.loads((TOOLS/'receipts/positive_quota_overrun_authorization.json').read_text())
assert auth['root_owned'] is auth['user_authorized'] is True and auth['actual_unreserved_seconds'] == actual
assert auth['root_decision_sha256'] == sha(TOOLS/'receipts/root_decision.json')
assert auth['actual_resource_snapshot_sha256'] == sha(TOOLS/'receipts/root_resource_snapshot.json')
meta = json.loads((STAGE/'kernel/kernel-metadata.json').read_text())
assert len(meta) == 16 and 'id_no' not in meta and meta['id'].split('/')[1] == meta['title']
assert len(meta['title']) <= 50 and meta['is_private'] is True and meta['enable_gpu'] is True
assert meta['enable_internet'] is meta['enable_tpu'] is False
assert meta['kernel_sources'] == ['arsen4ikvar/nlp-hw02-gated-b034-20261022b','arsen4ikvar/nlp-hw02-full-cache-20261023a']
assert meta['dataset_sources'] == ['arsen4ikvar/nlp-hw02-fullrange-8h-tools-20261024a']
assert meta['competition_sources'] == meta['model_sources'] == []
assert not list(STAGE.rglob('*.pt')) and not list(STAGE.rglob('*.npy'))
ast.parse((STAGE/'kernel/run.py').read_text())
result = {'schema':'fullrange_positive_quota_actual_stage_metadata_preflight_v1','utc':datetime.now(timezone.utc).isoformat(),
 'complete':True,'manifest_sha256':sha(manifest_path),'build_receipt_sha256':sha(STAGE/'build_receipt.json'),
 'driver_sha256':build['driver_sha256'],'metadata_sha256':sha(STAGE/'kernel/kernel-metadata.json'),
 'source_sha256':build['runtime_archive_sha256'],'tools_files':45,'runtime_members':80,
 'root_bound_NEW_cap':196608,'actual_H':None,'actual_quota_observed_utc':quota['observed_utc'],
 'actual_unreserved_seconds':actual,'full8h_cap_and_reserves_funded_by_current_balance':False,
 'provider_acceptance_established':False,'fresh_positive_quota_slot_deadline_gate_required':True,
 'all_tools_and_archive_hashes_passed':True,'no_PT_NPY_model_forward_or_provider_calls':True,
 'preflight_source_sha256':sha(Path(__file__)),
 'limits':['Actual remote arrays/fullstate/CPU-RAM/GPU/admission remain runtime checks.','Read-only observed build quota is not fresh postreadback launch admission.','No dataset upload or provider launch was performed.']}
output = ROOT/'reports/evidence/fullrange_positive_quota_launch_20261023c/actual_stage_metadata_preflight.json'
assert not output.exists()
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'sha256':sha(output),**{k:result[k] for k in ['manifest_sha256','metadata_sha256','build_receipt_sha256']}},indent=2))
