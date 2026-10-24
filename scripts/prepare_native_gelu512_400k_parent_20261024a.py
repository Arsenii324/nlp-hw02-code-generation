"""One local actual8-role opaque SHA projection; no tensor deserialization or provider calls."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
R=Path(__file__).resolve().parents[1]
BACKUP=R/'reports/evidence/native_offnode_monitor_20261024e/attempts/gelu/20261024T015335-62923/receipt.json'
ORIGINAL=R/'artifacts/paid-native-backups-57422123/gelu/20261024T015335-62923'
OUT=R/'artifacts/native-gelu512-400k-parent-20261024a'
PC=R/'artifacts/paid-wide-gelu-launch-20261023b/contract.json'
SOURCE=R/'artifacts/paid-wide-scratch-20261023b/source.bin'
ROLES={'last_training.pt','model.pt','metrics.json','run_config.json','tokenizer.json','data_stats.json','validation.json','validation_legacy.json'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ref(p):return {'path':str(p.relative_to(R)),'sha256':sha(p)}
def localref(p):return {'path':str(p.relative_to(OUT)),'sha256':sha(p)}
def main():
 assert not OUT.exists();receipt_bytes=BACKUP.read_bytes();receipt=json.loads(receipt_bytes);pointer=read(ORIGINAL/'latest_checkpoint.json');metrics=read(ORIGINAL/'metrics.json');metadata=read(ORIGINAL/'run_config.json');snapshot=read(ORIGINAL/'snapshot_receipt.json');pc=read(PC)
 assert receipt['complete'] is snapshot['complete'] is True and receipt['reason']=='terminal' and receipt['job']=='gelu' and (R/receipt['local_path']).resolve()==ORIGINAL.resolve()
 assert pointer['schema']=='wide_scratch_durable_pointer_a_v1' and pointer['slot'] in ('durable-slot0','durable-slot1') and set(pointer['files_sha256'])==ROLES and pointer['counter_origin']==0 and pointer['absolute_step']==snapshot['absolute_step']==receipt['absolute_step']==400000
 assert sha(ORIGINAL/'latest_checkpoint.json')==snapshot['pointer_sha256']==receipt['pointer_sha256']=='f5e40db9216611d30a96e898cdb773189211b6a1642ecbea588ea998618b80c1'
 assert sha(PC)==pointer['contract_sha256']==metadata['contract_sha256']=='d73de8b21e1b4c0615fa9db9efef874fb6cf5f9e7cfb6d9116e6a2cb2cc93c69'
 assert sha(SOURCE)==pc['runtime_archive_sha256']==metadata['runtime_archive_sha256']=='97494fb88adab12d244716a782b079b74f3072ae11f5f17166c5f22159c857bd'
 assert pc['verified_actual_values'] is True and pc['seed']==42 and pc['counter_origin']==0 and pc['lr_fraction']==1. and pc['cache_manifest_sha256']=='8ce0939534bdb5f2b791d19d0ab13f6b4c79402b8627fecad4e9902cd08bb13c'
 assert metadata['schema']=='wide_scratch_run_a_v1' and metadata['parameters']==30705728 and metadata['unique_xsa_scalars']==64 and metadata['config']==pc['config'] and metadata['seed']==42 and metadata['counter_origin']==0 and metadata['no_parent_model_moments_or_data_lineage'] is True
 assert metrics==pointer['metrics'] and metrics['schema']=='wide_scratch_metrics_a_v1' and metrics['absolute_completed_step']==metrics['completed_steps']==metrics['lr_absolute_step']==400000 and metrics['optimizer_updates']+metrics['skipped_updates']==400000 and metrics['sampled_examples']==25600000
 assert metadata['recipe']==pc['recipe']==metrics['sampler_checkpoint']['recipe'] and metadata['recipe_sha256']==pc['recipe_sha256']==pointer['recipe_sha256']==metrics['sampler_checkpoint']['recipe_sha256']
 assert metrics['sampler_checkpoint']['at_consumed_window_boundary'] is True and metrics['sampler_checkpoint']['pending_rows']==0 and metrics['sampler_checkpoint']['absolute_step']==400000
 with zipfile.ZipFile(SOURCE) as z:
  assert len(z.namelist())==len(set(z.namelist()))==64 and set(z.namelist())==set(pc['runtime_source_sha256'])
  assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in pc['runtime_source_sha256'].items())
 verified={}
 for n in sorted(ROLES|{'latest_checkpoint.json','snapshot_receipt.json'}):
  p=ORIGINAL/n;assert p.is_file() and not p.is_symlink();h=sha(p);size=p.stat().st_size;assert h==receipt['files'][n]['sha256'] and size==receipt['files'][n]['bytes']
  if n in ROLES:assert h==snapshot['files'][n]['sha256']==pointer['files_sha256'][n]
  verified[n]={'sha256':h,'bytes':size}
 assert BACKUP.read_bytes()==receipt_bytes,'Receipt changed during projection preflight'
 OUT.mkdir();(OUT/'roles').mkdir();(OUT/'evidence').mkdir()
 # APFS clone bytes only; never import torch or read .pt tensors.
 for n in sorted(ROLES):subprocess.run(['cp','-c',str(ORIGINAL/n),str(OUT/'roles'/n)],check=True)
 for p,target in [(ORIGINAL/'latest_checkpoint.json',OUT/'latest_checkpoint.json'),(ORIGINAL/'snapshot_receipt.json',OUT/'evidence/snapshot_receipt.json'),(PC,OUT/'evidence/native_parent_contract.json'),(SOURCE,OUT/'evidence/source.bin')]:subprocess.run(['cp','-c',str(p),str(target)],check=True)
 (OUT/'evidence/backup_receipt.json').write_bytes(receipt_bytes)
 for n in ROLES:assert sha(OUT/'roles'/n)==verified[n]['sha256'] and (OUT/'roles'/n).stat().st_size==verified[n]['bytes']
 proposal={'schema':'wide_origin0_resume_parent_acceptance_a_v1','root_accepted':False,'complete':True,'architecture':'gelu512','counter_origin':0,'pointer_sha256':sha(OUT/'latest_checkpoint.json'),'contract_sha256':sha(OUT/'evidence/native_parent_contract.json'),'runtime_archive_sha256':sha(OUT/'evidence/source.bin'),'files_sha256':pointer['files_sha256'],'absolute_step':400000,'selected_step_saved_metadata':metrics['selected_step'],'best_validation_loss_saved_metadata':metrics['best_validation_loss'],'endpoint_validation_loss_saved_metadata':metrics['final_validation_loss'],'validation_scope':'OPAQUE8ROLE_SHA_AND_SAVED_FAMILY_METADATA_ONLY','fullstate_tensors_moments_scaler_or_RNG_loaded':False,'model_tensors_or_CPU_constructor_called':False,'strict_remote_restore_before_any_gradient_required':True,'training_horizon_completion_not_independently_accepted_here':True,'backup_receipt':localref(OUT/'evidence/backup_receipt.json'),'snapshot_receipt':localref(OUT/'evidence/snapshot_receipt.json'),'root_acceptance_pending':True}
 write(OUT/'evidence/root_parent_acceptance_PROPOSAL.json',proposal)
 binding={'schema':'wide_origin0_resume_parent_a_v1','verified_actual_values':False,'architecture':'gelu512','artifact_root':str(OUT),'root_parent_acceptance':localref(OUT/'evidence/root_parent_acceptance_PROPOSAL.json'),'pointer':localref(OUT/'latest_checkpoint.json'),'parent_contract':localref(OUT/'evidence/native_parent_contract.json'),'parent_archive':localref(OUT/'evidence/source.bin'),'projection':{n:localref(OUT/'roles'/n) for n in sorted(ROLES)}};write(OUT/'parent_binding_PROPOSAL.json',binding)
 packet={'schema':'native_gelu512_actual_400k_parent_packet_a_v1','root_accepted':False,'complete_opaque_projection':True,'endpoint_absolute_step':400000,'selected_CEbest_step_saved_metadata':metrics['selected_step'],'endpoint_fullstate_sha256':pointer['files_sha256']['last_training.pt'],'selected_model_sha256':pointer['files_sha256']['model.pt'],'parent_binding':ref(OUT/'parent_binding_PROPOSAL.json'),'parent_pointer':ref(OUT/'latest_checkpoint.json'),'native_parent_contract':ref(OUT/'evidence/native_parent_contract.json'),'root_parent_acceptance':ref(OUT/'evidence/root_parent_acceptance_PROPOSAL.json'),'native_archive':ref(OUT/'evidence/source.bin'),'parent_metadata':{n:ref(OUT/'roles'/n) for n in sorted(ROLES-{'model.pt','last_training.pt'})},'parent_selector':{'pointer_basename':'latest_checkpoint.json','pointer_sha256':sha(OUT/'latest_checkpoint.json'),'roles':{n:'roles/'+n for n in sorted(ROLES)}},'parent_dataset_id':None,'Kaggle_B_source_freeze_sha256':'adf01761bc8f5eac2d122bf537548d7f03e770d9b4feec9cc482644642116b8c','root_source_acceptance':None,'root_decision':None,'resource_snapshot':None,'root_NEW_cap':None,'common_NEW_horizon':None,'platform_seconds':None,'strict_fullpopulation_binding_ABI':'Use root-accepted fresh copy of parent_binding_PROPOSAL.json; same unchanged wide_origin0_resume_a loader. No state reset or best-weight-for-moment substitution.','remaining_root_bindings':['Accept opaque role/source/family metadata facts separately from Arch selected-model CPU/status acceptance.','Write fresh root-accepted parent acceptance copy and verified parent-binding copy with its actual new acceptance SHA; preserve both proposals.','Kaggle B root source/decision/resource/cap/budget/new IDs and exact private parent asset upload/readback/version remain unbound.','Fullpopulation root fixedH/seeds/data/runtime/owner binding and strict runtime moments/scaler/RNG/savedLR/endpoint-CE check remain separate.'],'no_tensor_or_model_or_array_inspection':True,'provider_actions':False}
 write(OUT/'parent_packet_ROOT_PENDING.json',packet)
 result={'schema':'native_gelu512_400k_parent_projection_receipt_a_v1','complete':True,'root_accepted':False,'backup_receipt_sha256':hashlib.sha256(receipt_bytes).hexdigest(),'source_snapshot':str(ORIGINAL.relative_to(R)),'source_files_verified':verified,'APFS_cloned8_roles':True,'target_files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},'endpoint_step':400000,'selected_step_saved_metadata':metrics['selected_step'],'saved_optimizer_updates':metrics['optimizer_updates'],'saved_skips':metrics['skipped_updates'],'snapshot_or_copy_remote_repeated':False,'tensor_or_array_deserialization':False,'CPU_model_inspection_repeated':False,'root_pending_packet_sha256':sha(OUT/'parent_packet_ROOT_PENDING.json'),'preparation_script_sha256':sha(Path(__file__))};write(OUT/'preparation_receipt.json',result);print(json.dumps({'output':str(OUT.relative_to(R)),'packet_sha256':sha(OUT/'parent_packet_ROOT_PENDING.json'),'acceptance_PROPOSAL_sha256':sha(OUT/'evidence/root_parent_acceptance_PROPOSAL.json'),'binding_PROPOSAL_sha256':sha(OUT/'parent_binding_PROPOSAL.json'),'receipt_sha256':sha(OUT/'preparation_receipt.json')}))
if __name__=='__main__':main()
