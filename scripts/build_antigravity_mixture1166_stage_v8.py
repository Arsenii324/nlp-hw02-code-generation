"""Build a fresh, local-only two-model 1166 proof stage; never contacts Kaggle."""
import hashlib,json,shutil,zipfile,ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'artifacts/kaggle/antigravity-mixture1166-proof-stage-20261024a-v8'
TOOLS=STAGE/'tools';MODELS=TOOLS/'models'
DONOR=ROOT/'artifacts/kaggle/own-fullpop512-n16-proof-20261024a/tools/candidate_bundle.bin'
OUT=ROOT/'artifacts/mixture_fullpop768_n8_512new_n3_test_predictions.jsonl'
VAL=[ROOT/'artifacts/antigravity_gelu768_fullpop531k_eval/validation_n8_candidates.jsonl',ROOT/'artifacts/antigravity_gelu512_long262k_eval/validation_n8_candidates.jsonl']
TEST=[ROOT/'artifacts/antigravity_gelu768_fullpop531k_eval/test_n8_candidates.jsonl',ROOT/'artifacts/antigravity_gelu512_long262k_eval/test_n8_candidates.jsonl']
MODELS_META=[
 {'name':'768fp_n8','file':'models/768fp_n8.pt','source':'artifacts/paid-native-backups-57422123/gelu768/20261024T091350-79019/model.pt','sha256':'ea4f5bc5e72d418987bc8fe1439e0afc736a7ae51447aab5363a70d154bd6972','architecture':'own768_gelu','config':{'vocab_size':2048,'d_model':768,'heads':12,'encoder_layers':4,'decoder_layers':4,'ff':3072,'dropout':0.1,'max_source':256,'max_target':192,'exclusive_self_attention':False,'learnable_xsa':True,'activation':'gelu','loop_repeats':1,'share_loop_weights':True,'residual_scale':1.0,'loop_order':'stack','loop_share_scope':'block','gated_repeats':False,'repeat_residual_scale':0.5},'step':528000,'family':'wide_gelu768_fullpopulation_continuation_a_v1','samples_per_question':8,'metrics':'artifacts/paid-native-backups-57422123/gelu768/20261024T091350-79019/metrics.json','snapshot':'artifacts/paid-native-backups-57422123/gelu768/20261024T091350-79019/snapshot_receipt.json','training_contract':'artifacts/native-gelu768-fullpopulation-bound-20261024a/training_contract_ROOT_BOUND.json','validation_loss':0.21149580623436748},
 {'name':'512_new_n3','file':'models/512_new_n3.pt','source':'artifacts/paid-native-backups-57422123/gelu/20261024T103443-79019/model.pt','sha256':'fd5bbeeadc8f451a2d47c740d73a38e75d3e5cdb65b72194e5065a138c161462','architecture':'own512_gelu','config':{'vocab_size':2048,'d_model':512,'heads':8,'encoder_layers':4,'decoder_layers':4,'ff':2048,'dropout':0.1,'max_source':256,'max_target':192,'exclusive_self_attention':False,'learnable_xsa':True,'activation':'gelu','loop_repeats':1,'share_loop_weights':True,'residual_scale':1.0,'loop_order':'stack','loop_share_scope':'block','gated_repeats':False,'repeat_residual_scale':0.5},'step':662144,'family':'wide_fullpopulation_continuation_a_v1','samples_per_question':3,'metrics':'artifacts/paid-native-backups-57422123/gelu/20261024T103443-79019/metrics.json','snapshot':'artifacts/paid-native-backups-57422123/gelu/20261024T103443-79019/snapshot_receipt.json','training_contract':'artifacts/native-fullpopulation-longer-bound-20261024b/training_contract_ROOT_BOUND.json','validation_loss':0.22055138446033892},
]
MODEL_SOURCE_SHA='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26'
TOKENIZER_SHA='a255d879979714a88f50e64dac1d462705a935acc1bb01f5f055f02bbad66d6f'
CONSENSUS=ROOT/'reports/evidence/primary_reference_consensus.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def notebook():
 md='''# 768FP N8 + GELU512 fullpopulation N3 proof — source only\n\nThis exact package samples two validation-selected checkpoints at fixed counts 8 and 3, then uses the frozen numeric safe-AST plurality selector. The expected test output hash is a post-generation reproduction check; it is not used to choose candidates. The package contains the ordered question-only 1,919-row test projection, no labels, and no extra training data.\n\nThe 143.7-second T4 figure is a scripted estimate, not a measurement. A cold one-visible-T4 run must record generation/selection/serialization, notebook, executor and whole-child clocks; all must be at most 180 seconds. The actual Kaggle terminal proof and cleanup remain separate root-owned gates.\n'''
 setup='''import os,sys,time,json,hashlib,zipfile,shutil,tempfile\nfrom pathlib import Path,PurePosixPath\nNOTEBOOK_START=time.perf_counter(); ROOT=Path.cwd(); PACKAGE=ROOT/"candidate_bundle_v8.zip"\nROOT_BOUND=False; EXPECTED_PACKAGE_SHA="SOURCE_ONLY_UNBOUND"\nif not ROOT_BOUND or EXPECTED_PACKAGE_SHA=="SOURCE_ONLY_UNBOUND": raise RuntimeError("Source-only package: root binding and independent review required")\ndef sha(p):\n h=hashlib.sha256()\n with Path(p).open('rb') as f:\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n return h.hexdigest()\nif sha(PACKAGE)!=EXPECTED_PACKAGE_SHA: raise RuntimeError("Exact bound package SHA required")\nif "torch" in sys.modules: raise RuntimeError("Fresh kernel required")\nos.environ["CUDA_VISIBLE_DEVICES"]="0"\nWORK=Path(tempfile.mkdtemp(prefix="mixture1166-proof-"))\nwith zipfile.ZipFile(PACKAGE) as z:\n names=z.namelist()\n if len(names)!=len(set(names)): raise RuntimeError("Duplicate package members")\n MANIFEST=json.loads(z.read("stage_manifest_v8.json"))\n for info in z.infolist():\n  q=PurePosixPath(info.filename)\n  if q.is_absolute() or ".." in q.parts or info.is_dir() or (info.external_attr>>16)&0o170000==0o120000: raise RuntimeError("Unsafe package path")\n  dst=WORK.joinpath(*q.parts);dst.parent.mkdir(parents=True,exist_ok=True)\n  with z.open(info) as src,dst.open('xb') as out:shutil.copyfileobj(src,out,1048576)\n  expected=MANIFEST.get("payload_sha256",{}).get(info.filename)\n  if expected and sha(dst)!=expected: raise RuntimeError("Payload hash mismatch")\nif MANIFEST.get("status")!="ROOT_BOUND_READY" or MANIFEST.get("verified_actual_values") is not True: raise RuntimeError("Root binding absent")\nif sha(WORK/"test_questions.json")!=MANIFEST["test_questions_sha256"]: raise RuntimeError("Question bytes differ")\nsys.path.insert(0,str(WORK/"src"))\n'''
 load='''import torch\nfrom tokenizers import Tokenizer\nfrom nlp_hw02.model import load_model\nif not torch.cuda.is_available() or torch.cuda.device_count()!=1 or "T4" not in torch.cuda.get_device_name(0): raise RuntimeError("Exactly one visible NVIDIA T4 required")\ntorch.set_num_threads(2); LOAD_START=time.perf_counter(); MODELS=[]\nif sha(WORK/"src/nlp_hw02/model.py")!=MANIFEST["model_source_sha256"]: raise RuntimeError("Model source SHA differs")\nfor spec in MANIFEST["models"]:\n path=WORK/spec["file"]\n if sha(path)!=spec["sha256"]: raise RuntimeError("Weight SHA differs")\n model,checkpoint=load_model(path)\n if checkpoint.get("config")!=spec["config"] or checkpoint.get("step")!=spec["step"] or checkpoint.get("validation_loss")!=spec["validation_loss"] or checkpoint.get("seed")!=42 or checkpoint.get("dataset_revision")!="b2724689ff079e9c22efd548fe134bfb29036dbb": raise RuntimeError("Checkpoint metadata differs")\n MODELS.append(model); del checkpoint\nTOKENIZER=Tokenizer.from_file(str(WORK/"tokenizer.json")); QUESTIONS=json.loads((WORK/"test_questions.json").read_text())\nif len(QUESTIONS)!=1919 or any(not isinstance(q,str) for q in QUESTIONS): raise RuntimeError("Expected 1,919 ordered question-only rows")\ntorch.cuda.synchronize(); LOAD_SECONDS=time.perf_counter()-LOAD_START\n'''
 infer='''from nlp_hw02.mixture1166_runtime import run\nOUT=Path.cwd()/"candidate_output"\nif OUT.exists(): raise RuntimeError("Fresh output dir required; preserve prior results")\nRESULT=run(MODELS,TOKENIZER,QUESTIONS,OUT,MANIFEST)\nNOTEBOOK_SECONDS=time.perf_counter()-NOTEBOOK_START\nRESULT["model_tokenizer_load_seconds"]=LOAD_SECONDS; RESULT["notebook_start_through_hash_seconds"]=NOTEBOOK_SECONDS\nRESULT["expected_output_reproduced"]=RESULT["output_reproduced"] and NOTEBOOK_SECONDS<=180 and RESULT["generation_selection_serialization_seconds"]<=180\nRESULT["four_clock_worker_measurements_required"]=True; RESULT["proof_qualified"]=False\n(OUT/"candidate_timing.json").write_text(json.dumps(RESULT,indent=2,allow_nan=False)+"\\n")\nprint(json.dumps({"rows":RESULT["rows"],"output_reproduced":RESULT["output_reproduced"],"clocks":{k:v for k,v in RESULT.items() if "seconds" in k},"proof_qualified":False},indent=2))\nif not RESULT["expected_output_reproduced"]: raise RuntimeError("Output or notebook/core clock gate failed; preserve outputs")\n'''
 cells=[{'cell_type':'markdown','metadata':{},'source':md.splitlines(True)}]
 for code in (setup,load,infer):
  ast.parse(code);cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(True)})
 return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'proof_status':'SOURCE_ONLY_UNBOUND'},'nbformat':4,'nbformat_minor':5}

def main():
 if STAGE.exists():raise FileExistsError(f'Preserving existing v8 stage {STAGE}')
 correction=ROOT/'reports/evidence/antigravity_mixture1166_stage_20261024a_v6/v5_score_source_invalid.json'
 correction_record={'schema':'antigravity_mixture1169_v5_source_correction_v1','v5_stage':'artifacts/kaggle/antigravity-mixture1169-proof-stage-20261024a','v5_bundle_sha256':'d8c4bf3c107bb0cc4f207470e9dcea024765d31948220c2cbcf0e04ba7219d78','v5_status':'INVALID_UNBOUND_DO_NOT_USE','reason':'Its 512_400k validation/test candidate-pool input came from wide-t4-budget GELU512 selected step228352 (model c33c451d...), not exact selected GELU512 checkpoint step398336/durable endpoint400000 (model 81fd4262...). Correct 768FP8 + exact512_400k4 + 512_new2 replay was1165/1529, not the claimed1169. V5 is preserved byte-for-byte and its copied 512 weight remains present only in the invalid local stage. V6 uses no 512_400k model or pool.','correct_400k_validation_pool_sha256':'f7c83172b0e8bb5b005f00ef03e2ac7ddbb14b1a7ac7c64c489dacbe2cca3511','no_provider_action':True,'no_model_deserialization':True}
 if correction.exists():
  if json.loads(correction.read_text())!=correction_record:raise FileExistsError(f'Preserving changed v5 correction: {correction}')
 else:correction.write_text(json.dumps(correction_record,indent=2)+'\n')
 needed=sum((ROOT/m['source']).stat().st_size for m in MODELS_META)*2+512*1024**2
 if shutil.disk_usage(ROOT).free<needed:raise OSError('Insufficient disk for two exact model copies plus package')
 for m in MODELS_META:
  if sha(ROOT/m['source'])!=m['sha256']:raise ValueError('Source model SHA changed: '+m['name'])
 if not OUT.is_file():raise FileNotFoundError('Correct N8+N3 test output reference absent')
 evidence=ROOT/'reports/evidence/antigravity_mixture1166_stage_20261024a_v6';evidence.mkdir(parents=True,exist_ok=True)
 replay=ROOT/'reports/evidence/antigravity_mixture1166_stage_20261024a_v6/validation_replay.json'
 if not replay.is_file():raise FileNotFoundError('Validation replay receipt absent')
 if STAGE.exists():raise FileExistsError(STAGE)
 TOOLS.mkdir(parents=True);MODELS.mkdir(parents=True)
 model_spec=[]
 for m in MODELS_META:
  dst=TOOLS/m['file'];shutil.copyfile(ROOT/m['source'],dst)
  if sha(dst)!=m['sha256']:raise ValueError('Staged model SHA mismatch')
  model_spec.append({k:m[k] for k in ('name','file','source','sha256','architecture','config','step','family','samples_per_question','validation_loss')})
  model_spec[-1].update({'model_source_sha256':MODEL_SOURCE_SHA,'metrics':{'path':m['metrics'],'sha256':sha(ROOT/m['metrics'])},'snapshot_receipt':{'path':m['snapshot'],'sha256':sha(ROOT/m['snapshot'])},'training_contract':{'path':m['training_contract'],'sha256':sha(ROOT/m['training_contract'])}})
 with zipfile.ZipFile(DONOR) as z:
  for name in ('src/nlp_hw02/__init__.py','src/nlp_hw02/arithmetic.py','src/nlp_hw02/model.py','src/nlp_hw02/sampling_candidate.py','src/nlp_hw02/tokenizer.py','src/nlp_hw02/own_selected_sampling_d.py','tokenizer.json','test_questions.json'):
   p=TOOLS/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
 runtime=TOOLS/'src/nlp_hw02/mixture1166_runtime.py';runtime.write_bytes((ROOT/'scripts/antigravity_mixture1166_runtime_v6.py').read_bytes())
 questions=json.loads((TOOLS/'test_questions.json').read_text())
 if len(questions)!=1919 or any(not isinstance(q,str) for q in questions):raise ValueError('Test projection must contain 1,919 question-only strings')
 payload={p.relative_to(TOOLS).as_posix():sha(p) for p in sorted(TOOLS.rglob('*')) if p.is_file()}
 manifest={'schema':'antigravity_mixture1166_stage_v8_v1','status':'LOCAL_STAGE_AWAITING_INDEPENDENT_REVIEW_AND_ROOT_BINDING','verified_actual_values':False,'models':model_spec,'counts':[8,3],'candidates_per_question':11,'model_source_sha256':MODEL_SOURCE_SHA,'tokenizer_sha256':TOKENIZER_SHA,'selector_sha256':payload['src/nlp_hw02/sampling_candidate.py'],'runtime_sha256':payload['src/nlp_hw02/mixture1166_runtime.py'],'payload_sha256':payload,'expected_validation':{'correct':1166,'denominator':1529,'replay_receipt_sha256':sha(replay),'pool_sha256':[sha(p) for p in VAL],'consensus_sha256':sha(CONSENSUS)},'test_questions_sha256':sha(TOOLS/'test_questions.json'),'ordered_test_questions_sha256':json.loads(replay.read_text())['ordered_test_questions_sha256'],'expected_output_sha256':sha(OUT),'expected_ordered_codes_sha256':json.loads(replay.read_text())['ordered_codes_sha256'],'output_rows':1919,'test_candidate_pool_sha256':[sha(p) for p in TEST],'test_labels_read':False,'extra_training_data':False,'latency_estimate_seconds':143.7,'latency_is_measured':False,'proof_clocks':['generation_selection_serialization','notebook','executor','whole_child'],'clock_limit_seconds':180,'one_visible_t4_required':True,'canonical_promotion':False}
 (TOOLS/'stage_manifest_v8.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with zipfile.ZipFile(TOOLS/'candidate_bundle_v8.zip','x',compression=zipfile.ZIP_STORED) as z:
  for p in sorted(TOOLS.rglob('*')):
   if p.is_file() and p.name!='candidate_bundle_v8.zip':z.write(p,p.relative_to(TOOLS).as_posix())
 nb=notebook();(TOOLS/'inference_v8.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
 receipt={'schema':'antigravity_mixture1166_stage_build_v8_v1','complete':True,'stage':str(STAGE.relative_to(ROOT)),'manifest_sha256':sha(TOOLS/'stage_manifest_v8.json'),'bundle_sha256':sha(TOOLS/'candidate_bundle_v8.zip'),'notebook_sha256':sha(TOOLS/'inference_v8.ipynb'),'runtime_source_sha256':sha(ROOT/'scripts/antigravity_mixture1166_runtime_v6.py'),'model_copy_bytes':sum((TOOLS/m['file']).stat().st_size for m in MODELS_META),'free_before_bytes':shutil.disk_usage(ROOT).free+sum((TOOLS/m['file']).stat().st_size for m in MODELS_META),'free_after_bytes':shutil.disk_usage(ROOT).free,'local_inference':False,'local_model_deserialization':False,'provider_action':False,'proof_qualified':False,'selected_step_768':528000,'completed_endpoint_768':531072,'selected_validation_loss_768':0.21149580623436748}
 (STAGE/'build_receipt_v8.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
