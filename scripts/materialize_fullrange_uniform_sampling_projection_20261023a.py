"""Actual authorized local projection + stdlib metadata inspection; no tensor imports."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022'
OUT=W/'artifacts/fullrange-uniform-sampling-projection-20261023a'
DIAG=R/'artifacts/kaggle/fullrange-paired-v1-a-diagnostics';TOOLS=R/'artifacts/kaggle/fullrange-paired-20261023a/tools'
E=R/'reports/evidence/full_cache_training_launch_20261023a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def require(ok,msg):
 if not ok:raise ValueError(msg)
require(not OUT.exists(),'Fresh projection required; retain failures, no rerun')
dr=E/'diagnostic_recovery.json';sr=E/'selected_model_recovery.json';ir=R/'reports/evidence/fullrange_actual_whole_independent_review_20261023.json'
require(sha(dr)=='0ecb86b7da113499ef8c389efb3011451dc9eaee08cf86275d551c5475799ef7','Accepted raw diagnostic recovery changed')
require(sha(sr)=='16d4ac38ba205f8caaff5bd16445467a713410ad5adece70ec390e3ca19f460e','Accepted raw model recovery changed')
require(sha(ir)=='36953e42573e1a0ee7efc708f586e11c09cae8ac07129ade06eba1aaa7b47e45','Accepted whole actual audit changed')
d=read(dr);s=read(sr);terminal=read(DIAG/'session_result.json');terminal_sha=sha(DIAG/'session_result.json');require(terminal_sha==d['terminal_sha256']==s['terminal_sha256']=='e1ae001d203a82ca223fe7751dd5f377c3bfd9fe2951a646e34824b68bbcc521','Actual terminal differs')
for n,v in d['diagnostic_download']['downloaded'].items():require(sha(DIAG/n)==v['sha256']==terminal['artifact_sha256'][n],'Actual diagnostic bytes differ:'+n)
require(len(d['diagnostic_download']['downloaded'])==129 and d['diagnostic_download']['complete'] is True and d['identity_stable_through_recovery'] is True and s['complete'] is True and not s['fullstate_downloaded'] and not s['arrays_downloaded'],'Actual recovery subset incomplete')
driver='9e2e6eedb23adb7a8b1ebba75c15f7510ea78a6bf018d389c881e78d02c81f19'
for x in (d['identity_before'],d['identity_after_diagnostics']):require(x['numeric_id']==140324445 and x['current_version_number']==1 and x['source_sha256']==driver,'Diagnostic version/source identity differs')
for x in (s['identity_before'],s['identity_after']):require(x['numeric_id']==140324445 and x['current_version_number']==1 and x['driver_sha256']==driver,'Selected model version/source differs')
# Original immutable tools are all small metadata/source/question files, no states/arrays.
m=read(TOOLS/'full_cache_transition_manifest.json');require(sha(TOOLS/'full_cache_transition_manifest.json')==terminal['manifest_sha256'],'Actual tools manifest differs')
for n,v in m['input_sha256'].items():require(sha(TOOLS/n)==v,'Actual tools input differs:'+n)
OUT.mkdir(parents=True);prov=[]
def copy(source,target,kind,expected=None,clone=False):
 source=Path(source);target=Path(target);require(source.suffix not in ('.npy',) and source.name!='last_training.pt','No arrays/fullstate')
 expected=expected or sha(source);require(sha(source)==expected,'Source differs:'+str(source));target.parent.mkdir(parents=True,exist_ok=True);require(not target.exists(),'Fresh target only')
 if clone:subprocess.run(['/bin/cp','-c',str(source),str(target)],check=True)
 else:shutil.copyfile(source,target)
 require(sha(target)==expected,'Copied projection differs');prov.append({'source':str(source.relative_to(R)),'target':str(target.relative_to(OUT)),'sha256':expected,'bytes':target.stat().st_size,'origin':kind,'APFS_clone_requested':clone})
tr=OUT/'training';tools=OUT/'tools';inputs=OUT/'inputs';proposals=OUT/'proposals'
base=['session_result.json','full_cache_transition_manifest.json','input_preflight.json','preflight_handoff.json','resolved_inputs/path_resolution.json','resolved_inputs/resolution_handoff.json','resolved_inputs/path_selectors.json']
base += ['resolved_inputs/'+p+name+'.json' for name in ('cache6m','cachefull') for p in ('staged-contract-','resolved-contract-')]
base += ['cache-coordination/'+name+suffix+'.json' for name in ('cache6m','cachefull') for suffix in ('-prefix-request','-decision-ack','-postprefix_continue')]
base += ['cache-coordination/common_decision.json','cache-coordination/common_release.json']
for n in base:copy(DIAG/n,tr/n,'actual_recovered_diagnostic',terminal_sha if n=='session_result.json' else terminal['artifact_sha256'][n])
for p in TOOLS.rglob('*'):
 if p.is_file():copy(p,tools/p.relative_to(TOOLS),'retained_original_frozen_source_tools')
originals=R/'artifacts/kaggle/cache-6m-v1/cache'
cpus={}
for name in ('cache6m','cachefull'):
 pointer=read(DIAG/('run-'+name+'/latest_checkpoint.json'));slot=pointer['slot'];require(slot=='durable-slot0','Actual latest pointer, not prior summary, controls slot')
 cpu_path=E/('selected_'+name+'_cpu_tensor_integrity.json');cpu=read(cpu_path);cpus[name]=(cpu_path,cpu)
 require(cpu['complete'] and cpu['terminal_sha256']==terminal_sha and cpu['pointer_sha256']==sha(DIAG/('run-'+name+'/latest_checkpoint.json')) and cpu['slot']==slot and cpu['model_sha256']==pointer['files_sha256']['model.pt'],'Actual CPU receipt/pointer differs')
 for suffix in ('latest_checkpoint.json','worker_result.json'):copy(DIAG/('run-'+name+'/'+suffix),tr/('run-'+name+'/'+suffix),'actual_recovered_diagnostic',terminal['artifact_sha256']['run-'+name+'/'+suffix])
 for role in ('run_config.json','metrics.json','tokenizer.json','data_stats.json'):copy(DIAG/('run-'+name+'/'+slot+'/'+role),tr/('run-'+name+'/'+slot+'/'+role),'actual_recovered_selected_slot_metadata',pointer['files_sha256'][role])
 model=R/s['recovered'][name]['path'];copy(model,tr/('run-'+name+'/'+slot+'/model.pt'),'actual_recovered_selected_weights_APFS_clone',pointer['files_sha256']['model.pt'],True)
 for role in ('validation.json','validation_legacy.json'):
  copy(originals/role,tr/('run-'+name+'/'+slot+'/'+role),'retained_pinned_original_identical_bytes_NOT_fresh_remote_recovery',pointer['files_sha256'][role])
 copy(cpu_path,proposals/('raw_'+name+'_CPU_integrity.json'),'unchanged_raw_CPU_receipt')
common=W/'artifacts/trained-gated-sampling-inputs-20261023a'
input_names={'validation_questions':'validation_questions.json','parity_pairs':'actual_first4_parity_pairs.json','test_parquet':'test.parquet','legacy_selector_source':'sampling_candidate.py'}
for n in input_names.values():copy(common/n,inputs/n,'existing_pinned_goldfree_sampling_or_fourpair_audit_input')
# Root explicitly authorized exact normalized recovery acceptance after raw rehash.
recovery={'schema':'fullrange_training_recovery_a_v1','root_accepted':True,'complete':True,'family':'full_cache_isqrt_a','version':1,'current_version_before':1,'current_version_after':1,'source_before_sha256':d['identity_before']['source_sha256'],'source_after_sha256':d['identity_after_diagnostics']['source_sha256'],'training_terminal_sha256':terminal_sha,'training_source_sha256':terminal['source_sha256'],'training_manifest_sha256':terminal['manifest_sha256'],'numeric_id':140324445,'raw_diagnostic_recovery':{'path':str(dr.relative_to(R)),'sha256':sha(dr)},'raw_selected_weights_recovery':{'path':str(sr.relative_to(R)),'sha256':sha(sr)},'independent_actual_whole_audit':{'path':str(ir.relative_to(R)),'sha256':sha(ir)},'acceptance_basis':'Root explicit exact-normalization authorization after accepted raw diagnostic/selected recovery and independent whole audit; recomputed all actual hashes/identities','scope':'Accepted version/source/terminal recovery facts only. All fullstate/arrays stay remote; historical overwritten-pointer bytes not recovered. Missing validation roles are same-hash retained-original materializations, not new downloads.'}
save(tr/'normalized_recovery.json',recovery)
# Only accepted stdlib metadata module: no producer or tensor/model imports.
module=W/'scripts/fullrange_sampling_contract_a.py';require(sha(module)=='f7f2c40f79df83aced13ffc2e82ab7bf0d266a51e7406fe1170bc09abddc0389','Accepted metadata inspector changed')
loader=importlib.util.spec_from_file_location('actual_fullrange_metadata_inspection_only',module);adapter=importlib.util.module_from_spec(loader);loader.loader.exec_module(adapter)
require('torch' not in sys.modules and not any(n=='nlp_hw02' or n.startswith('nlp_hw02.') for n in sys.modules),'No Torch/model import allowed')
returned={}
for name in ('cache6m','cachefull'):
 spec=read(W/'configs/fullrange_sampling_producer_a.template.json');cpu_path,cpu=cpus[name]
 spec.update(verified_actual_values=True,proposal_status='ACTUAL_VALUES_METADATA_INSPECTED_ROOT_CPU_ACCEPTANCE_AND_STAGE_PENDING',input_root=str(inputs),selected={'model_sha256':cpu['model_sha256'],'config':cpu['config'],'step':cpu['step'],'validation_loss':cpu['validation_loss']},generation=adapter.GENERATION,backend_policy=adapter.BACKEND_POLICY,samples_per_question=24)
 spec['reviewed_source']['sha256']=sha(W/'configs/fullrange_sampling_source_a_20261023.json')
 spec['training'].update(family='full_cache_isqrt_a',artifact_root=str(tr),tools_root=str(tools),run_id=name,terminal={'path':'session_result.json','sha256':terminal_sha},recovery={'path':'normalized_recovery.json','sha256':sha(tr/'normalized_recovery.json')},pointer_sha256=cpu['pointer_sha256'])
 spec['inputs']={k:{'path':v,'sha256':sha(inputs/v)} for k,v in input_names.items()}
 save(proposals/(name+'_producer_spec_proposal.json'),spec)
 try:identity,files,fixed_inputs,archive=adapter.inspect_training_metadata(spec)
 except BaseException as e:
  save(proposals/(name+'_metadata_inspection_FAILURE.json'),{'type':type(e).__name__,'error':str(e),'candidate_source_edited':False,'root_acceptance_fabricated':False});raise
 require('torch' not in sys.modules and not any(n=='nlp_hw02' or n.startswith('nlp_hw02.') for n in sys.modules),'Metadata function imported model/Torch')
 require(all(cpu[k]==identity[k] for k in ('model_sha256','model_source_sha256','config','step','validation_loss','seed','dataset_revision','parameters','run_id','slot','pointer_sha256')),'Actual generic CPU facts disagree with metadata identity')
 identity_path=proposals/(name+'_returned_metadata_identity.json');save(identity_path,identity)
 audit={'schema':'fullrange_actual_local_metadata_inspection_a_v1','complete':True,'run_id':name,'inspector_sha256':sha(module),'inspector_freeze_sha256':spec['reviewed_source']['sha256'],'returned_identity_sha256':sha(identity_path),'proposal_spec_sha256':sha(proposals/(name+'_producer_spec_proposal.json')),'accepted_normalized_recovery_sha256':sha(tr/'normalized_recovery.json'),'all_seven_non_fullstate_roles_checked':True,'last_training_bytes_read':False,'torch_or_model_imported':False,'producer_executed':False,'actual_tensor_CPU_receipt_sha256':sha(cpu_path),'CPU_fields_equal_metadata_identity':True}
 audit_path=proposals/(name+'_actual_metadata_audit.json');save(audit_path,audit)
 integrity=read(W/'configs/fullrange_selected_checkpoint_integrity_a.template.json');integrity.update(complete=False,root_accepted=False,actual_CPU_tensor_audit_sha256=sha(cpu_path),actual_metadata_protocol_audit_sha256=sha(audit_path),identity=identity,selected_tensor_facts={'parameters':cpu['parameters'],'checkpoint_fields':cpu['checkpoint_fields'],'finite_selected_state_tensors':cpu['finite_state_tensors'],'strict_alias_check':cpu['strict_alias_check'],'unique_xsa_scalars':cpu['unique_xsa_scalars']},proposal_facts_complete=True,proposal_status='ROOT_REVIEW_REQUIRED_NOT_ACCEPTED')
 save(proposals/(name+'_complete_integrity_PROPOSAL.json'),integrity);returned[name]={'identity_sha256':sha(identity_path),'metadata_audit_sha256':sha(audit_path),'integrity_proposal_sha256':sha(proposals/(name+'_complete_integrity_PROPOSAL.json')),'model_sha256':identity['model_sha256'],'selected_step':identity['step'],'endpoint':identity['absolute_completed_step'],'H':identity['admitted_NEW_horizon'],'slot':identity['slot'],'root_accepted':False}
save(OUT/'projection_provenance.json',{'schema':'fullrange_uniform_sampling_projection_materialization_a_v1','complete':True,'authorizer':'Root explicit LOCAL projection and accepted metadata-only inspector task; exact recovery normalization separately authorized','file_copies':prov,'normalized_recovery_sha256':sha(tr/'normalized_recovery.json'),'original_diagnostic_download_count':130,'actual_projection_diagnostic_subset_count':len(base)+12,'no_arrays_or_fullstate':True,'two_selected_weight_APFS_clones_only':True,'no_model_forward_or_producer':True,'raw_receipt_hashes_preserved':{'diagnostic':sha(dr),'selected_weights':sha(sr)},'returned':returned})
save(R/'reports/evidence/fullrange_uniform_sampling_projection_a_20261023.json',{'schema':'fullrange_uniform_sampling_projection_a_v1','status':'LOCAL_ACTUAL_METADATA_PASS_CPU_COMPLETE_INTEGRITY_PROPOSALS_UNACCEPTED','projection':str(OUT.relative_to(R)),'provenance_sha256':sha(OUT/'projection_provenance.json'),'normalized_recovery_sha256':sha(tr/'normalized_recovery.json'),'returned':returned,'Torch_model_or_producer_calls':False,'provider_action':False})
print(json.dumps({'projection':str(OUT.relative_to(R)),'returned':returned,'normalized_recovery_sha256':sha(tr/'normalized_recovery.json')}))
