"""Actual common portable-D files only; no executable binding/package/notebook or tensor load."""
from pathlib import Path
import contextlib,hashlib,json,os,subprocess
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022'
T=R/'artifacts/kaggle/graph-g-budget-20261024d/tools';OUT=R/'artifacts/own512-portable-payload-20261024a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not OUT.exists();freeze=W/'configs/own_portable_candidate_source_d_20261024.json';assert sha(freeze)=='f67d6b7d31ed77c96468cb8d212c558414e8966a57170d99a61954f1d372c73c';f=json.loads(freeze.read_text());runtime=W/'src/nlp_hw02/own_selected_sampling_d.py';assert sha(runtime)==f['source_sha256']['src/nlp_hw02/own_selected_sampling_d.py']=='391c9d8f0bdde49b7a3711ec603592bd27bc57a58108c670674aed4b2cc7e1e5'
 receipt=json.loads((T.parent/'build_receipt.json').read_text());assert receipt['model_sha256']=='81fd42627751284c7ed61d23fa3aec76f8a2c34f857ae9f78d5bd12155068306' and receipt['model_step']==398336 and receipt['training_durable_step']==400000
 src={'model.pt':T/'projection/model.pt','tokenizer.json':T/'projection/tokenizer.json','source.bin':T/'evidence/source.bin','sampling_candidate.py':T/'sampling_candidate.py','runtime/own_selected_sampling_d.py':runtime,'evidence/cpu_ROOT_ACCEPTED.json':T/'cpu_ROOT_ACCEPTED.json','evidence/lineage_ROOT_ACCEPTED.json':T/'lineage_ROOT_ACCEPTED.json','evidence/portable_D_source_freeze.json':freeze}
 expected={'model.pt':'81fd42627751284c7ed61d23fa3aec76f8a2c34f857ae9f78d5bd12155068306','tokenizer.json':'0fa06854a98e82cd520c13f4c670040c772fec38f2cd2cacb5de11130bc42d74','source.bin':'97494fb88adab12d244716a782b079b74f3072ae11f5f17166c5f22159c857bd','sampling_candidate.py':'250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5','runtime/own_selected_sampling_d.py':f['source_sha256']['src/nlp_hw02/own_selected_sampling_d.py'],'evidence/cpu_ROOT_ACCEPTED.json':receipt['tools_sha256']['cpu_ROOT_ACCEPTED.json'],'evidence/lineage_ROOT_ACCEPTED.json':receipt['tools_sha256']['lineage_ROOT_ACCEPTED.json'],'evidence/portable_D_source_freeze.json':sha(freeze)}
 before={str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [*src.values(),T/'test.parquet']}
 assert all(sha(p)==expected[n] for n,p in src.items());assert sha(T/'test.parquet')=='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
 cpu=json.loads(src['evidence/cpu_ROOT_ACCEPTED.json'].read_text());lineage=json.loads(src['evidence/lineage_ROOT_ACCEPTED.json'].read_text())
 assert cpu['root_accepted'] is cpu['complete'] is lineage['root_accepted'] is lineage['complete'] is True and cpu['model_sha256']==lineage['model_sha256']==expected['model.pt'] and cpu['step']==lineage['step']==398336 and lineage['absolute_durable_step']==400000 and cpu['parameters']==30705728 and cpu['unique_xsa_scalars']==64
 # Optional local pandas/numexpr import notices are suppressed; any actual read exception still fails.
 with open(os.devnull,'w') as sink,contextlib.redirect_stderr(sink):
  import pyarrow.parquet as pq
  questions=pq.read_table(T/'test.parquet',columns=['question'])['question'].to_pylist()
 assert len(questions)==1919 and all(isinstance(q,str) for q in questions);compact=json.dumps(questions,ensure_ascii=False,separators=(',',':')).encode();ordered=hashlib.sha256(compact).hexdigest()
 OUT.mkdir();(OUT/'runtime').mkdir();(OUT/'evidence').mkdir()
 for n,p in src.items():
  target=OUT/n;subprocess.run(['cp','-c',str(p),str(target)],check=True);assert sha(target)==expected[n]
 # This file is the exact canonical compact ordered list, no label column or trailing newline.
 (OUT/'test_questions.json').write_bytes(compact)
 assert json.loads((OUT/'test_questions.json').read_text())==questions and sha(OUT/'test_questions.json')==ordered
 after={str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [*src.values(),T/'test.parquet']};assert after==before
 files={str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file()}
 inventory={'purpose':'COMMON_PORTABLE_D_PAYLOAD_ONLY_NOT_A_BINDING_PACKAGE_OR_PROOF','accepted':False,'common_model_identity':{'model_sha256':expected['model.pt'],'selected_CEbest_step':398336,'training_endpoint_step':400000,'parameters':30705728,'unique_xsa_scalars':64,'model_source_sha256':'07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26'},'selection':None,'samples_per_question':None,'engine':None,'quality':None,'test_reference':None,'actual_G_outputs_bound':False,'ordered_test_questions_sha256':ordered,'test_questions_count':1919,'test_questions_encoding':'JSON ensure_ascii=False,separators=(comma,colon), no trailing newline','test_source_parquet_sha256':'8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a','parquet_columns_read':['question'],'portable_D_source_freeze_sha256':sha(freeze),'files':files,'source_parent_files_used_before_after':before,'copied_source_files_and_test_parquet_unchanged':True,'whole_graph_stage_rehashed':False,'accepted_CPU_lineage_receipts_copied_byte_exact_as_evidence':True,'lineage_internal_refs_are_preserved_original_provenance_not_a_new_complete_adapter_projection':True,'no_fullstate_arrays_or_gold_validation_payload':True,'model_tensor_or_forward_calls':False,'package_or_notebook_created':False,'provider_actions':False}
 write(OUT/'payload_inventory.json',inventory)
 out=R/'reports/evidence/own512_portable_payload_20261024a';out.mkdir();record={'complete':True,'common_payload_only_unaccepted':True,'payload':str(OUT.relative_to(R)),'payload_inventory_sha256':sha(OUT/'payload_inventory.json'),'all_output_files':{str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file()},'model_APFS_clone':True,'model_or_fullstate_tensor_load':False,'parquet_columns_read':['question'],'1919_canonical_compact_ordered_question_sha256':ordered,'selection_count_engine_quality_test_reference_all_NULL':True,'preparation_script_sha256':sha(Path(__file__))};write(out/'preparation_receipt.json',record)
 print(json.dumps({'payload':str(OUT.relative_to(R)),'receipt_sha256':sha(out/'preparation_receipt.json'),'inventory_sha256':sha(OUT/'payload_inventory.json'),'test_question_order_sha256':ordered,'files':len(record['all_output_files'])}))
if __name__=='__main__':main()
