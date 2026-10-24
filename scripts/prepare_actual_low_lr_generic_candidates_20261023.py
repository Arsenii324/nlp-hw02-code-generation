"""Bind two actual own lowLR checkpoints to unmodified genericB; no model/provider calls."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,importlib.util
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';E=R/'reports/evidence/low_lr_generic_candidates_20261023a';P=R/'artifacts/kaggle/half-lowlr-candidate-inputs-20261023a';D=R/'artifacts/kaggle/half-lowlr-v1-a-diagnostics/recovery-current-v1';T=R/'artifacts/kaggle/half-lowlr-20261023a/tools';LE=R/'reports/evidence/half_low_lr_launch_20261023a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(p.resolve()),'sha256':sha(p)}
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(v,indent=2)+'\n')
def copy(p,t):
 assert not t.exists();t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t);assert sha(p)==sha(t)
assert not P.exists() and not E.exists();P.mkdir(parents=True);E.mkdir();family=LE/'actual_family_validation_ROOT_ACCEPTED.json';fa=json.loads(family.read_text());assert fa['root_accepted'] and fa['complete'] and fa['independent_pass'];cpu_paths={k:LE/('selected_'+k+'_cpu_ROOT_ACCEPTED.json') for k in ['low-lr-eighth','low-lr-quarter']}
freeze_path=W/'configs/own_checkpoint_fixed24_source_b_20261023.json';freeze=json.loads(freeze_path.read_text());assert sha(freeze_path)=='50aa2424b66448ca323a8d138ce1d75723740cd967e71e994c386ea28d2a1af4'
for n,h in freeze['source_sha256'].items():assert sha(W/n)==h;copy(W/n,P/n)
copy(freeze_path,P/'configs'/freeze_path.name)
inputsource=R/'artifacts/kaggle/cooldown-paired-n24-20261023a/tools/inputs';inputnames=['validation_questions.json','actual_first4_parity_pairs.json','test.parquet','sampling_candidate.py']
for n in inputnames:copy(inputsource/n,P/'inputs'/n)
roles={};contracts={};lineages={};before={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()}
for role,cp in cpu_paths.items():
 cpu=json.loads(cp.read_text());pointer_path=D/('run-'+role)/'latest_checkpoint.json';pointer=json.loads(pointer_path.read_text());assert cpu['root_accepted'] and cpu['complete'] and cpu['model_sha256']==pointer['files_sha256']['model.pt'] and cpu['step']==pointer['metrics']['selected_step'];base=P/'roles'/role;base.mkdir(parents=True);projection={}
 for n,h in pointer['files_sha256'].items():
  if n=='last_training.pt':continue
  target=base/'selected'/n;target.parent.mkdir(parents=True,exist_ok=True)
  if n=='model.pt':
   src=R/'artifacts/kaggle/half-lowlr-v1-a-selected-models'/('run-'+role)/pointer['slot']/n;subprocess.run(['cp','-c',str(src),str(target)],check=True,capture_output=True)
  else:src=D/('run-'+role)/pointer['slot']/n;copy(src,target)
  assert sha(target)==h;projection[n]={'path':'selected/'+n,'sha256':h}
 evidence_sources={'source':T/'half_low_lr_manifest_a.json','archive':T/'source.bin','parent':T/'parent-terminal-provenance.json','pointer':pointer_path,'contract':T/('contract-'+role+'.json'),'family_validation':family};evidence={}
 for key,src in evidence_sources.items():
  target=base/'evidence'/({'archive':'source.bin'}.get(key,key+'.json'));copy(src,target);evidence[key]={'path':str(target.relative_to(base)),'sha256':sha(target)}
 copy(cp,base/'root_cpu_integrity.json')
 lineage={'schema':'own_checkpoint_root_lineage_v1','root_accepted':True,'complete':True,'own_training_from_scratch':True,'family_validation_accepted':True,'training_family':'half_low_lr_pair_a_v1','model_sha256':cpu['model_sha256'],'model_source_sha256':cpu['model_source_sha256'],'step':cpu['step'],'validation_loss':cpu['validation_loss'],'dataset_revision':cpu['dataset_revision'],'checkpoint_role':'selected_ce_best','absolute_durable_step':pointer['absolute_step'],'training_complete':True,'run_id':role,'required_software':{'torch':'2.10.0+cu128','cuda':'12.8'},'projection_sha256':{k:v['sha256'] for k,v in projection.items()},'evidence':evidence,'acceptance_basis':{'root_family_receipt_sha256':sha(family),'root_CPU_receipt_sha256':sha(cp),'authority':'Root explicitly authorized actual local candidate binding from these accepted CPU/family receipts. This normalized lineage adds no tensor/protocol validation claim; independent actual projection review follows.','original_reset_provenance':'Own B034 weights-only optimizer/scaler/RNG reset300000 retained; actual D-half endpoint417384 moments resumed; no foreign weights/no invented optimizer lineage.'}}
 write(base/'root_lineage.json',lineage);lineages[role]=ref(base/'root_lineage.json');spec=json.loads((W/'configs/own_checkpoint_fixed24_producer_b.template.json').read_text());spec.update(verified_actual_values=True,artifact_root=str(base),input_root=str(P/'inputs'));spec['root_cpu_integrity']={'path':'root_cpu_integrity.json','sha256':sha(base/'root_cpu_integrity.json')};spec['root_lineage']={'path':'root_lineage.json','sha256':sha(base/'root_lineage.json')};spec['projection']=projection;spec['runtime_source_archive']=evidence['archive'];spec['reviewed_source']['sha256']=sha(freeze_path);write(P/'contracts'/(role+'.json'),spec);contracts[role]=spec
# Import only unchanged stdlib metadata adapter at actual materialized root. No producer/Torch import.
sys.dont_write_bytecode=True;path=P/'scripts/own_checkpoint_fixed24_contract_b.py';sp=importlib.util.spec_from_file_location('actual_generic_B_metadata',path);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);pre={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()}
for role,spec in contracts.items():
 identity,files,inputs,archive=mod.verify_training(spec);assert identity['model_sha256']==fa['roles'][role]['model_sha256'];filesmap={str(p.relative_to(P/'roles'/role)):ref(p) for p in (P/'roles'/role).rglob('*') if p.is_file()};roles[role]={'producer_contract':ref(P/'contracts'/(role+'.json')),'expected_identity':identity,'model':ref(files['model.pt']),'model_relative_path':'selected/model.pt','files':filesmap}
post={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()};assert pre==post and 'torch' not in sys.modules
packet=json.loads((W/'configs/own_checkpoint_pair_input_c.template.json').read_text());packet.update(verified_actual_values=False,paired_source_freeze_sha256=sha(W/'configs/own_checkpoint_pair_source_c_20261023.json'),roles=roles,shared_inputs={n:ref(P/'inputs'/n) for n in inputnames},image_origin=ref(R/'kaggle/data-coverage-300k-20261004a/kernel-metadata.json'),dataset_id='arsen4ikvar/nlp-hw02-lowlr-n24-tools-20261023a',kernel_id='arsen4ikvar/nlp-hw02-lowlr-n24-paired-20261023a');write(P/'input_packet_ACTUAL_CANDIDATES_PREPARED.json',packet)
report={'schema':'low_lr_generic_actual_candidates_v1','complete':True,'root':str(P),'source_freeze_sha256':sha(freeze_path),'root_family_sha256':sha(family),'root_CPU_receipts':{k:ref(v) for k,v in cpu_paths.items()},'normalization_basis':'Actual root accepted CPU +family receipts; no foreign-family coercion and no model forward','producer_contracts':{k:ref(P/'contracts'/(k+'.json')) for k in roles},'root_lineages':lineages,'roles':{k:v['expected_identity'] for k,v in roles.items()},'actual_metadata_verifier_sha256':sha(path),'metadata_only_verifier_passed_both':True,'torch_imported':False,'files_unchanged_by_verifier':pre==post,'files_sha256':post,'models_APFS_cloned':2,'fullstates_arrays_copied_or_read':False,'packet':ref(P/'input_packet_ACTUAL_CANDIDATES_PREPARED.json'),'stage_built':False,'provider_action':False,'Colab_path_contract':'Same source/projection layout; patch ONLY each spec.artifact_root and input_root to actual VM paths, leave SHA receipts/source and policy untouched. Execute scripts/probe_own_checkpoint_fixed24_b.py --contract ... --output fresh ... --deadline absolute; separate sequential roles on oneT4 are not a coldnotebook proof.'};write(E/'actual_candidates_metadata_preflight.json',report);print(json.dumps({'root':str(P),'receipt_sha256':sha(E/'actual_candidates_metadata_preflight.json'),'packet_sha256':sha(P/'input_packet_ACTUAL_CANDIDATES_PREPARED.json'),'contracts':report['producer_contracts'],'lineages':lineages}))
