"""Actual local half415656 projection; no producer/model/provider invocation."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,importlib.util
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';S=R/'artifacts/kaggle/cooldown-paired-n24-20261023a/tools';D=R/'artifacts/kaggle/own-checkpoint-control-prepared-20261023a';E=R/'reports/evidence/own_checkpoint_control_preparation_20261023'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def ref(p):return {'path':str(p.resolve()),'sha256':sha(p)}
assert not D.exists() and not E.exists();D.mkdir(parents=True);E.mkdir()
files={};source_map={}
for p in (S/'training_projection').rglob('*'):
 if not p.is_file():continue
 n=p.relative_to(S/'training_projection')
 if n.parts[0]=='run-cooldown-full' or p.suffix=='.pt':continue
 target=D/'roles/control/training'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha(target)==sha(p);files['training/'+str(n)]=ref(target);source_map[str(target.relative_to(D))]=ref(p)
model=S/'training_projection/run-cooldown-half/durable-slot1/model.pt';target=D/'roles/control/training/run-cooldown-half/durable-slot1/model.pt';target.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['cp','-c',str(model),str(target)],check=True,capture_output=True);assert sha(target)=='edb1b328116abb82240b094093108e533a0ce026a283f7f5463602631f253089';model_relative=str(target.relative_to(D/'roles/control'));files[model_relative]=ref(target);source_map[str(target.relative_to(D))]=ref(model)
for p in (S/'training_tools').iterdir():
 if p.is_file():
  t=D/'roles/control/training_tools'/p.name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t);assert sha(t)==sha(p);files['training_tools/'+p.name]=ref(t);source_map[str(t.relative_to(D))]=ref(p)
inputs={}
for n in ['validation_questions.json','actual_first4_parity_pairs.json','test.parquet','sampling_candidate.py','integrity/cooldown-half.json']:
 p=S/'inputs'/n;t=D/'inputs'/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t);assert sha(t)==sha(p);inputs[n]=ref(t);source_map[str(t.relative_to(D))]=ref(p)
original=json.loads((S/'contracts/cooldown-half.json').read_text());spec=json.loads(json.dumps(original));spec['training']['artifact_root']=str(D/'roles/control/training');spec['training']['tools_root']=str(D/'roles/control/training_tools');spec['input_root']=str(D/'inputs')
# Exact accepted stdlib-only verifier, metadata+streamed hashes; no Torch/import model.
sys.dont_write_bytecode=True;module_path=W/'scripts/cooldown_sampling_contract_b.py';before={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()};sp=importlib.util.spec_from_file_location('accepted_cooldown_control_metadata',module_path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);identity,roles,actualinputs,archive=m.verify_training(spec);after={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()};assert before==after and 'torch' not in sys.modules and identity['step']==415656
# Stage contract keeps only the three explicit runtime-resolved path fields unbound.
staged=json.loads(json.dumps(original));write(D/'control_producer_contract.json',staged)
selection=R/'reports/evidence/cooldown_half_fixed24_selection_20261023.json';proof=R/'reports/evidence/cooldown_half_actual_proof_root_review_20261023.json';quality=R/'reports/evidence/cooldown_pair_n24_actual_quality_independent_review_20261023.json';timing=R/'artifacts/kaggle/cooldown-paired-n24-v1-a-diagnostics/run-cooldown-half/validation_n24_timing.json'
proofrefs={}
for k,p in [('selection',selection),('proof_root',proof),('quality_independent',quality),('validation_timing',timing)]:
 t=D/'provenance'/p.name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t);assert sha(t)==sha(p);proofrefs[k]=ref(t)
references={'validation_prediction_sha256':json.loads(timing.read_text())['prediction_sha256'],'test_prediction_sha256':'f8c78da5c4dbfc6fea37779046268e092a11ce266815ee48dabd782c96136d6b','test_ordered_codes_sha256':'015304d69d1a7904adfeca977415fe168c218a87df4ecb92fc4f78064fd82de4'}
pc={'schema':'proved_canonical_fixed24_control_v1','root_accepted':False,'complete':True,'evidence_kind':'proved_cooldown_control','model_sha256':identity['model_sha256'],'step':identity['step'],'quality_fixed1529':1117,'root_cpu_integrity_sha256':identity['root_cpu_integrity_sha256'],'selection_receipt':proofrefs['selection'],'references':references,'provenance':proofrefs,'identity':identity,'purpose':'Explicit positive-control requalification authorized by root; already proved, never relabeled as Colab/unproved. New wrapper awaits root content acceptance.'};write(D/'positive_control_lineage_PREPARED.json',pc)
packet=json.loads((W/'configs/own_checkpoint_control_pair_input_a.template.json').read_text());packet['paired_source_freeze_sha256']=sha(W/'configs/own_checkpoint_control_pair_source_a_20261023.json');packet['roles']['control']={'producer_contract':ref(D/'control_producer_contract.json'),'expected_identity':identity,'model':ref(target),'model_relative_path':model_relative,'files':files};packet['shared_inputs']=inputs;packet['positive_control_lineage']=ref(D/'positive_control_lineage_PREPARED.json');packet['image_origin']=ref(R/'kaggle/data-coverage-300k-20261004a/kernel-metadata.json');write(D/'paired_input_packet_CONTROL_PREPARED.json',packet)
report={'schema':'own_checkpoint_positive_control_local_projection_v1','complete':True,'producer_verifier_sha256':sha(module_path),'verifier_call':'Unchanged accepted stdlib verify_training on actual projected bytepaths; no Torch/model/forward/import override','identity':identity,'projection_unchanged_by_inspection':before==after,'torch_imported':False,'model_APFS_clones':1,'model_bytes':target.stat().st_size,'selected_role_files':len(files),'shared_inputs':len(inputs),'projected_files_sha256':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'copy_provenance':source_map,'artifact_root':str(D),'packet':ref(D/'paired_input_packet_CONTROL_PREPARED.json'),'positive_control_lineage':ref(D/'positive_control_lineage_PREPARED.json'),'guard_review':{'proof_registry_present':False,'reuse_guard':'Only exact inherited344k CURRENT_MODEL_SHA eed1 is rejected by frozen B; actualhalf415656 edb1 passes unchanged. No empty fake registry/override flag.','positive_control_authorization':'Root explicitly requested repeat measurement as a positive control, not new qualification claim.'},'excluded':'All peer weights, all last_training/fullstates, cache arrays; peer completion/protocol remains validated from actual terminal+both contracts/coordination sidecars.','remaining_NULL':'Snapshot model/CPU/lineage/sourceacceptance/screen decision/IDs remain unbound. New control wrapper root_accepted=false pending root content review; no stage build/provider action.'};write(E/'projection_and_metadata_preflight.json',report);print(json.dumps({'root':str(D),'packet_sha256':sha(D/'paired_input_packet_CONTROL_PREPARED.json'),'wrapper_sha256':sha(D/'positive_control_lineage_PREPARED.json'),'receipt_sha256':sha(E/'projection_and_metadata_preflight.json'),'model_bytes':target.stat().st_size,'rolefiles':len(files),'sharedinputs':len(inputs)}))
