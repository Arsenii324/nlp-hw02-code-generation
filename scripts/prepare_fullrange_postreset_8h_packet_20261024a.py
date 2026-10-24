"""Actual input-byte bindings, provisional root bounds; quota/build deliberately pending."""
from pathlib import Path
import hashlib,json,shutil,zipfile
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';OLD=R/'reports/evidence/fullrange_bucket_cooldown_launch_20261023c';E=R/'reports/evidence/fullrange_postreset_launch_20261024a';E.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ref(p):return {'path':str(p.resolve()),'sha256':sha(p)}
p=read(W/'configs/proposals/fullrange_bucket_cooldown_actual_input_packet_20261023c.bound.json');bounds={'platform':28800,'parent':28740,'work':27240,'worker':27235,'cleanup':27600,'hash':28695,'tail':1500};cap=196608
assert cap%64==0 and bounds['parent']==bounds['platform']-60 and bounds['work']==bounds['parent']-bounds['tail'] and bounds['worker']==bounds['work']-5 and bounds['cleanup']==bounds['work']+360 and bounds['hash']==bounds['parent']-45
# Byte-exact existing accepted source tuple. This does not admit resource usage.
source=E/'source_acceptance.json';assert not source.exists();shutil.copyfile(OLD/'source_acceptance.json',source);assert sha(source)==sha(OLD/'source_acceptance.json')
results=read(OLD/'current_results_decision.json');results.update(authorize_local_stage_build=False,requested_NEW_cap=cap,bounds=bounds,pre_or_post_reset='post_reset',readiness='PROVISIONAL_PREPARED_INPUT_BYTES_ONLY_WAITING_RESET_QUOTA_AND_N24_RESULT_DECISION',post_reset_plan=ref(R/'docs/plans/2026-10-23-post-reset-measured-budget.md'),actual_pilot_recovery=ref(OLD/'diagnostic_recovery_retry1.json'),actual_pilot_root_recovery_acceptance=ref(OLD/'root_recovery_acceptance.json'),actual_N24_result=None,actual_quota_snapshot=None,scope='Root authorized preparation only. Larger cap/runtime admission needs fresh actual postreset quota; no forecast passed as actual, no allocation authority.')
resultsp=E/'current_results_decision_PREPARED.json';save(resultsp,results)
root=read(OLD/'root_binding.json');root.update(verified_actual_values=False,requested_NEW_cap=cap,bounds=bounds,current_results_decision_sha256=sha(resultsp),source_review_receipt_sha256=sha(source),build_quota_snapshot_receipt_sha256=None,build_resource_admission={'snapshot_receipt_sha256':None,'actual_unreserved_seconds':None,'other_active_residual_bound_seconds':None,'concrete_proof_reserve_seconds':None},actual_quota_admission=None,preparation_only=True,not_before_reset='2026-10-24T00:00:00Z',conditional_on_actual_N24_and_fresh_root_decision=True,scope='Immutable actual inputs/provisional8h bound; H remains real1024-prefix admitted. Unchanged builderB cannot build until actual qualifying quota/root flags are filled in a fresh accepted successor.')
rootp=E/'root_binding_PREPARED.json';save(rootp,root)
p.update(verified_actual_values=False,actual_H=None,root_decision=ref(rootp),current_results_decision=ref(resultsp),source_acceptance=ref(source),root_resource_snapshot={'path':None,'sha256':None},dataset_id='arsen4ikvar/nlp-hw02-fullrange-cooldown-tools-20261024a',kernel_id='arsen4ikvar/nlp-hw02-fullrange-cooldown-20261024a',pre_or_post_reset='post_reset',preparation_status='ACTUAL_INPUT_BYTES_VERIFIED_PROVISIONAL_8H_WAITING_REAL_RESET_QUOTA_AND_N24_FINAL_CHOICE',provisional_bounds=bounds,provisional_NEW_cap=cap,future_build_output_root=str(R/'artifacts/kaggle/fullrange-cooldown-20261024a'),prior_resource_snapshot_NOT_current_admission=ref(OLD/'resource_snapshot.json'))
packet=W/'configs/proposals/fullrange_bucket_cooldown_postreset_input_packet_20261024a.prepared.json';save(packet,p)
# Shared exact small/reference bytes only. No PT/fullstate/NPY read or builder invocation.
checks={}
def verify_ref(name,x):
 f=Path(x['path']);assert f.suffix not in ('.pt','.npy') and f.is_file() and sha(f)==x['sha256'];checks[name]=x['sha256']
for k in ('source_archive','runtime_freeze','runtime_independent_review','stage_builder_freeze','stage_builder_independent_review','parent_quality','parent_cpu_integrity','current_selection','kernel_metadata_origin','test_parquet'):verify_ref(k,p[k])
for k in ('terminal','manifest'):verify_ref('parent.'+k,p['parent'][k])
for n,x in p['parent']['small_roles'].items():verify_ref('parent.'+n,x)
for k,x in p['historical_cache'].items():verify_ref('historical_cache.'+k,x)
for k in ('terminal','manifest','coverage'):verify_ref('fullcache.'+k,p['fullcache'][k])
for k,x in p['fullcache']['small_files'].items():verify_ref('fullcache.'+k,x)
sf=read(p['runtime_freeze']['path']);bf=read(p['stage_builder_freeze']['path']);assert all(sha(W/n)==v for n,v in sf['source_sha256'].items()) and all(sha(W/n)==v for n,v in bf['source_sha256'].items());members={**sf['foundation_members_sha256'],**sf['source_sha256'],p['freeze_member_path']:p['runtime_freeze']['sha256']}
with zipfile.ZipFile(p['source_archive']['path']) as z:assert len(z.namelist())==len(set(z.namelist()))==len(members)==80 and set(z.namelist())==set(members) and all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in members.items())
terminal=read(p['parent']['terminal']['path']);roles={n:terminal['artifact_sha256']['run-control/'+n] for n in ('last_training.pt','model.pt','run_config.json','metrics.json','tokenizer.json','validation.json','validation_legacy.json','data_stats.json')};assert roles['last_training.pt']=='f9d4f20ca116b4d71089cbafeecab9c6cfb210752bd572e8da5bcca9627cfb0d' and roles['model.pt']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9'
full=read(p['fullcache']['terminal']['path']);cm=read(p['fullcache']['manifest']['path']);assert full['cache_ready'] and full['artifact_hashes_complete'] and all(full['artifact_sha256']['cache/'+n]==v for n,v in cm['checksums'].items())
assert not Path(p['future_build_output_root']).exists()
out=E/'prepared_packet_metadata_preflight.json';save(out,{'schema':'fullrange_postreset_prepared_actual_inputs_preflight_v1','status':'ACTUAL_INPUT_BYTES_PASS_BUILD_PENDING_REAL_RESET_QUOTA_AND_N24_CHOICE','packet_path':str(packet),'packet_sha256':sha(packet),'root_binding_sha256':sha(rootp),'current_results_sha256':sha(resultsp),'source_acceptance_sha256':sha(source),'verified_small_input_refs':checks,'all80_source_members_exact':True,'parent8roles_terminal_bound':roles,'fullcache7terminal_map_explicit':True,'provisional_bounds':bounds,'requested_NEW_cap':cap,'actual_H':None,'actual_quota_snapshot':None,'actual_stage_built':False,'builder_source_unchanged':True,'PT_fullstate_arrays_read':False,'provider_action':False,'required_next':['Root actual N24 decision','Fresh actual postreset qualifying quota, residual sessions/proof reserve and accepted root wrappers','One unchanged-builderB localbuild','Wholeactualstage/private source+mount readback/fresh launchquota/slots']});print(json.dumps({'packet':str(packet),'packet_sha256':sha(packet),'preflight_sha256':sha(out),'stage_built':False,'actual_quota':None}))
