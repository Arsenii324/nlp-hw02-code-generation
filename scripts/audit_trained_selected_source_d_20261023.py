"""Read-only normalized AST/actual-identity D audit; no binding/build/model/tests."""
import ast
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];W=ROOT/'.worktrees/strong-inference-fallback-20261022'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fn(t,name):return next(n for n in t.body if getattr(n,'name',None)==name)
def dump(n):return ast.dump(n,include_attributes=False)

def main():
 checks={}
 def check(key,ok):
  checks[key]=bool(ok)
  if not ok:raise ValueError(key)
 cp=W/'scripts/package_trained_selected_c.py';dp=W/'scripts/package_trained_selected_d.py';cs=cp.read_text();ds=dp.read_text();ct=ast.parse(cs);dt=ast.parse(ds)
 check('C_original_packager_preserved',sha(cp)=='bc3454f7805410bfb13d76926a3576f91b0cf1f79258c666911e205d161d9407')
 cf=fn(ct,'bind_reproduction');df=fn(dt,'bind_reproduction')
 ci=next(i for i,n in enumerate(cf.body) if (ast.get_source_segment(cs,n) or '').startswith("require(terminal['artifact_hashes_complete']"));di=next(i for i,n in enumerate(df.body) if (ast.get_source_segment(ds,n) or '').startswith("require(terminal['artifact_hashes_complete']"))
 check('guard_location_unchanged',ci==di)
 check('actual_4statement_replacement',all(text in '\n'.join(ast.get_source_segment(ds,n) for n in df.body[di:di+4]) for text in ("terminal['artifact_hashes_complete'] is True","actual_manifest=outputs/'trained_gated_sampling_manifest.json'","terminal['artifact_sha256']['trained_gated_sampling_manifest.json']==E_MANIFEST_SHA","sha(actual_manifest)==E_MANIFEST_SHA","actual_manifest.read_bytes()==manifest_path.read_bytes()","['inputs_sha256']==m['inputs_sha256']")))
 df.body[di:di+4]=[copy.deepcopy(cf.body[ci])];dt.body[0]=copy.deepcopy(ct.body[0]);check('whole_packager_AST_only_narrow_guard_and_doc_delta',dump(ct)==dump(dt))
 t=W/'artifacts/kaggle/trained-gated-sampling-tools-20261023b';out=ROOT/'artifacts/kaggle/trained-gated-sampling-v1-b';terminal=json.loads((out/'session_result.json').read_text());m=json.loads((t/'trained_gated_sampling_manifest.json').read_text())
 check('actual_terminal_missing_duplicated_inputs_recorded','inputs_sha256' not in terminal)
 check('actual_mapped_recovered_manifest_guard_passes',terminal['artifact_hashes_complete'] and terminal['artifact_sha256']['trained_gated_sampling_manifest.json']==sha(out/'trained_gated_sampling_manifest.json')==sha(t/'trained_gated_sampling_manifest.json')=='75432d90bb4969a7e4ddc823f845f90112522240804801d3a1dbe081dcfac2e2' and (out/'trained_gated_sampling_manifest.json').read_bytes()==(t/'trained_gated_sampling_manifest.json').read_bytes() and json.loads((out/'trained_gated_sampling_manifest.json').read_text())['inputs_sha256']==m['inputs_sha256'])
 check('all22_original_tools_payload_hashes_unchanged',all(sha(t/name)==digest for name,digest in m['inputs_sha256'].items()))
 old=json.loads((W/'configs/trained_selected_sampling_c_20261023.template.json').read_text());new=json.loads((W/'configs/trained_selected_sampling_d_20261023.template.json').read_text());new.pop('source_revision');check('D_contract_unbound_same_all_values_C',new==old and not new['verified_actual_values'] and new['mode'] is None and new['role_id'] is None and all(v is None for v in new['receipts'].values()))
 static=json.loads((ROOT/'reports/evidence/trained_selected_portable_c_static_20261023.json').read_text());check('all5_frozen_C_source_pins_unchanged',all(sha(W/static['source_paths'][key])==value for key,value in static['pins'].items()))
 decision=ROOT/'reports/evidence/trained_control_final_portable_selection_20261023.json';choice=json.loads(decision.read_text());check('actual_root_quality_choice_receipt',sha(decision)=='0b33aa20cef35e2fe484ba1c93f54e9375b72cf13528bb83e439244adfd90ba5' and choice['selected_after_actual_quality_audit'] and choice['role_id']=='control-final' and choice['samples_per_question']==24 and choice['model_sha256']=='eed1fdbb4f7e056536b322c08fad8660f5df24460f716a4d8e5382eb58ce8dc9' and choice['E_terminal_sha256']==sha(out/'session_result.json'))
 check('all63_actual_E_outputs_unchanged',all(sha(out/name)==value for name,value in terminal['artifact_sha256'].items()))
 check('currentcanonical_proof_backup_unchanged',sha(ROOT/'artifacts/selected_bundle.zip')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/selected_bundle.zip')=='aba6112e3d3d3c9d80c742a71d146d88bccf65946ba7ef6edded6dce2bd581fb' and sha(ROOT/'submit/inference.ipynb')==sha(ROOT/'artifacts/selected-b034-fixed24-preserved-20261023/inference.ipynb')=='c940570f7b24903ceade666e4f56b0441a8cdfd0e538480a56bf251daa322270')
 report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS fresh D source-only actual-schema correction; independent review before actual binding/build','all_checks_pass':all(checks.values()),'checks':checks,'C_preserved_pins':static['pins'],'D_packager_sha256':sha(dp),'D_contract_sha256':sha(W/'configs/trained_selected_sampling_d_20261023.template.json'),'root_actual_selection_receipt_sha256':sha(decision),'actual_E_terminal_sha256':sha(out/'session_result.json'),'script_sha256':sha(__file__),'actual_package_built':False,'canonical_changed':False}
 p=ROOT/'reports/evidence/trained_selected_portable_d_static_20261023.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'all_checks_pass':True,'D_packager_sha256':sha(dp),'D_contract_sha256':report['D_contract_sha256'],'evidence':str(p)}))

if __name__=='__main__':main()
