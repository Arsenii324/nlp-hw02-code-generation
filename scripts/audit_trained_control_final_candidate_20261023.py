"""Actual bound candidate static readback; no model/code-cell execution."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 directory=W/'artifacts/portable-trained-control-final-20261023a';freeze=json.loads((W/'configs/trained_selected_portable_source_d_20261023.json').read_text());specpath=W/'configs/trained_control_final_deployment_20261023a.json';spec=json.loads(specpath.read_text());stage=json.loads((directory/'staging.json').read_text());checks={}
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 check('exact_root_selection_receipt',sha(Path(spec['root_quality_decision']['path']))==spec['root_quality_decision']['sha256']=='0b33aa20cef35e2fe484ba1c93f54e9375b72cf13528bb83e439244adfd90ba5')
 check('D_freeze_manifest_and_sources_unchanged',sha(W/'configs/trained_selected_portable_source_d_20261023.json')==spec['portable_source_D_manifest_sha256']=='a11db4db58125336462465283525f79bd26f011bd1693c57d9b755c922e86357' and sha(W/freeze['packager'])==stage['packager_sha256']==freeze['packager_sha256']=='f96ad26981984c90b45437d18fcd5f99d151da8323ec09b52234b042f1e48c73' and all(sha(W/freeze['shared_C_source_paths'][key])==value for key,value in freeze['shared_C_source_pins'].items()))
 check('actual_spec_and_ZIP_staging_hashes',sha(specpath)==stage['deployment_spec_sha256'] and sha(directory/'selected_bundle.zip')==stage['package_sha256']=='7c9c7885912e867cbf2bd8eba9ee583cc6424265a2f8f3032299c2a1411c6083')
 with zipfile.ZipFile(directory/'selected_bundle.zip') as z:
  names=z.namelist();manifest_bytes=z.read('candidate_manifest.json');m=json.loads(manifest_bytes);deployment=m['deployment']
  check('exact11_safe_unique_minimal_members',len(names)==len(set(names))==11 and set(names)==set(stage['member_sha256'])|{'candidate_manifest.json'} and len(stage['member_sha256'])==10 and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names))
  check('manifest_actual_hash_and_staging',hashlib.sha256(manifest_bytes).hexdigest()==stage['manifest_sha256']=='35d1eb02f64397e23fffbba9945efb1e75db441a46208eac4500c6446ea610c7' and m['member_sha256']==stage['member_sha256'])
  for name,digest in m['member_sha256'].items():
   h=hashlib.sha256()
   with z.open(name) as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
   check('streamed_payload:'+name,h.hexdigest()==digest)
   if name.endswith('.py'):ast.parse(z.read(name).decode())
  check('actual_root_control_final_N24_selected',deployment['role_id']==spec['role_id']=='control-final' and deployment['mode']==spec['mode']=='fixed24' and deployment['samples_per_question']==spec['samples_per_question']==24 and deployment['root_quality_decision_sha256']==spec['root_quality_decision']['sha256'] and deployment['model_sha256']==m['member_sha256']['model.pt']==spec['model_sha256']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9')
  check('actual344k_config_source_parameters',deployment['checkpoint_step']==m['checkpoint_step']==344000 and deployment['model_config']==spec['model_config']==m['model_config'] and deployment['model_config']['gated_repeats'] is False and deployment['parameters']==m['parameters']==17524256 and m['member_sha256']['src/nlp_hw02/model.py']==deployment['model_source_sha256']=='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26' and m['member_sha256']['src/nlp_hw02/trained_selected_sampling_c.py']==freeze['shared_C_source_pins']['module'])
  check('actual_E_receipts_parity_output_provenance',deployment['receipts']==spec['receipts'] and deployment['actual_parity']['passed'] and deployment['receipts']['terminal']=='01066a6d40afcb787a8aa5667e956a2f4e32ce22eb3954b8a1f92239e82d4866' and deployment['reproduction']==spec['reproduction'] and deployment['reproduction']['code_only_jsonl_sha256']=='214bf02531a70cccc75a722c3a710d2466899d664c6d2d30c70a1c86f6eb7f29' and deployment['reproduction']['ordered_codes_sha256']=='16ba397472d2ade3f0dc93cbfcb14af62f8e224ff21b1eb39ef55412201f0fa2')
  check('no_history_gold_optimizer_payload',all(name not in names for name in ('bound_contract.json','actual_first4_parity_pairs.json','validation_questions.json','last_training.pt')) and all(not name.endswith(('.jsonl','.npy','.npz')) for name in names) and not any(key in deployment for key in ('actual_E_output_root','actual_E_tools_root','root_quality_decision')))
 nbpath=directory/'inference.ipynb';nb=json.loads(nbpath.read_text());template=json.loads((W/'submit/candidates/trained_selected_c_template_20261023.ipynb').read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code'];old=[c for c in template['cells'] if c['cell_type']=='code']
 check('fresh_notebook3_unexecuted_cells',len(cells)==3 and all(c['execution_count'] is None and c['outputs']==[] for c in cells))
 for c in cells:ast.parse(''.join(c['source']))
 text=[''.join(c['source']) for c in cells];before=[''.join(c['source']) for c in old]
 expected=before[0].replace('ZIP_SHA256 = None',"ZIP_SHA256 = '"+stage['package_sha256']+"'").replace('BOUND_CANDIDATE = False','BOUND_CANDIDATE = True')
 check('exact_reviewed_notebook_code_except_actualsha_bound',text[0]==expected and text[1:]==before[1:])
 check('no_old_selected_module_fallback','from nlp_hw02.trained_selected_sampling_c import' in text[0] and 'from nlp_hw02.trained_selected_sampling_c import run_trained_selected_candidate' in text[1] and not any('nlp_hw02.selected_sampling' in t for t in text))
 check('currentB034_canonical_backup_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/selected_bundle.zip')=='aba6112e3d3d3c9d80c742a71d146d88bccf65946ba7ef6edded6dce2bd581fb' and sha(ROOT/'submit/inference.ipynb')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/inference.ipynb')=='c940570f7b24903ceade666e4f56b0441a8cdfd0e538480a56bf251daa322270')
 report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS actual local bound344kN24 candidate static readback; independent bound review/fresh proof pending','checks':checks,'all_checks_pass':all(checks.values()),'candidate_directory':str(directory),'pins':{'package':sha(directory/'selected_bundle.zip'),'notebook':sha(nbpath),'staging':sha(directory/'staging.json'),'bound_spec':sha(specpath),'package_manifest':stage['manifest_sha256'],'root_selection_receipt':spec['root_quality_decision']['sha256']},'package_bytes':(directory/'selected_bundle.zip').stat().st_size,'new_runtime_proof':None,'canonical_promoted':False,'no_model_tests_cloud_actions':True};out=ROOT/'reports/evidence/trained_control_final_candidate_static_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_checks_pass':True,'pins':report['pins'],'package_bytes':report['package_bytes'],'evidence':str(out)}))

if __name__=='__main__':main()
