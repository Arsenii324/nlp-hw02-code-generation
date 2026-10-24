from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';O=W/'artifacts/fullrange-C-sampling-projection-20261023c';E=R/'reports/evidence/fullrange_bucket_cooldown_launch_20261023c';Q=E/'pair_local_preparation';Q.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def save(p,x):assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ref(p):return {'path':str(p.resolve()),'sha256':sha(p)}
fr=W/'configs/fullrange_pair_sampling_source_b_20261023.json';ind=R/'reports/evidence/fullrange_actual_schema_source_correction_independent_review_20261023.json';old=R/'reports/evidence/fullrange_pair_sampling_root_source_acceptance_20261023.json'
save(Q/'root_source_wrapper_B.json',{'freeze_sha256':sha(fr),'decision':'ACCEPT_FULLRANGE_PAIR_SOURCE_B_FOR_LOCAL_BUILD_ONLY','independent_sha256':sha(ind),'basis':'Root explicit fresh exact-literal/source-route correction + actuallocalstage authorization; independent deltaPASS9','original_root_acceptance':ref(old),'scope':'Source/localbuild only; no provider write/launch'})
t=rd(O/'training/session_result.json');models={};ints={}
for role in ('full-remaining','half-remaining'):
 pp=O/'training'/('run-'+role)/'latest_checkpoint.json';p=rd(pp);models[role]=ref(pp.parent/p['slot']/'model.pt');ints[role]=ref(O/'proposals'/(role+'_complete_integrity_C_PROPOSAL.json'))
save(Q/'quality_decision_PROPOSAL.json',{'schema':'fullrange_actual_N24_screen_decision_a_v1','root_accepted':False,'family':'fullrange_bucket_cooldown_a','training_source_sha256':t['source_sha256'],'training_terminal_sha256':sha(O/'training/session_result.json'),'requested_screen_platform_seconds':900,'canonical_promotion':False,'selected_models_sha256':{k:v['sha256'] for k,v in models.items()},'measured_beam_proxy_fixed1529':{'full':1044,'half':1039,'selected_6M_half':1037},'decision':'Measure both actual CE-best models with unchanged fixed24; modest beam delta is not N24/proof/promotion evidence','quality_evidence':ref(R/'reports/evidence/fullrange_cooldown_pilot_c_quality_independent_20261023.json')})
save(Q/'proof_registry.json',{'schema':'fullrange_proved_inference_registry_a_v1','root_owned':True,'complete':True,'entries':[],'scope':'Candidate-scoped: root explicitly states both newCweights have no measuredN24/proof; not an exhaustive historicalproof catalogue'})
p=rd(W/'configs/fullrange_pair_stage_input_b.template.json');p.update(verified_actual_values=True,paired_source_freeze_sha256=sha(fr),paired_root_source_review=ref(Q/'root_source_wrapper_B.json'),paired_independent_review=ref(ind),root_quality_decision=ref(Q/'quality_decision_PROPOSAL.json'),training_terminal=ref(O/'training/session_result.json'),normalized_recovery=ref(O/'training/normalized_recovery.json'),models=models,integrity=ints,proof_registry=ref(Q/'proof_registry.json'),proof_registry_files={},dataset_id='arsen4ikvar/nlp-hw02-fullrange-n24-tools-20261023c',kernel_id='arsen4ikvar/nlp-hw02-fullrange-n24-paired-20261023c')
# All and only metadata required by accepted pair builder; no state/array paths.
files={}
for x in (O/'training').rglob('*'):
 if x.is_file() and x.name not in ('model.pt','last_training.pt','normalized_recovery.json'):files[str(x.relative_to(O/'training'))]=ref(x)
p['training_files']=files;m=rd(O/'training/fullrange_bucket_cooldown_manifest_a.json');names={'fullrange_bucket_cooldown_source.bin','parent_metadata/data_stats.json'}|{r['contract_path'] for r in m['runs']};p['training_tools_files']={n:ref(O/'tools'/n) for n in names};p['inputs']={n:ref(O/'inputs'/n) for n in p['inputs']}
p['image_origin']=ref(R/'artifacts/kaggle/fullrange-cooldown-20261023c/kernel/kernel-metadata.json')
save(Q/'packet_PROPOSAL.json',p);print(json.dumps({'root_source_wrapper':str(Q/'root_source_wrapper_B.json'),'quality_proposal':str(Q/'quality_decision_PROPOSAL.json'),'packet':str(Q/'packet_PROPOSAL.json'),'packet_sha256':sha(Q/'packet_PROPOSAL.json'),'training_metadata_files':len(files)}))
