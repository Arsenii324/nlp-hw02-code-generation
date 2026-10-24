import ast,hashlib,json,math,operator,datetime,zipfile
from pathlib import Path
R=Path.cwd();T=R/'artifacts/kaggle/own-fullpop512-n16-proof-20261024a/tools';D=R/'artifacts/kaggle/own-fullpop512-n16-proof-v1-20261024a-recovered';E=R/'reports/evidence/own_fullpop512_n16_proof_launch_20261024a'
checks={}
def ck(n,b):
 if not b:raise ValueError(n)
 checks[n]=True
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rd(p):return json.loads(p.read_text())
def ordered(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
# Independent numeric interpreter: AST only. Never compile/eval/execute generated code.
class Unsupported(Exception):pass
def bounded(x):
 if type(x) not in (int,float) or (type(x)is int and x.bit_length()>256) or abs(x)>1e100 or (type(x)is float and not math.isfinite(x)):raise Unsupported()
 return x
BINARY={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}
CALLS={'int':int,'float':float,'round':round,'abs':abs,'min':min,'max':max}
def evaluate(code):
 try:
  if not isinstance(code,str) or len(code.encode('utf-8',errors='replace'))>65536:return False,None
  tr=ast.parse(code);nodes=list(ast.walk(tr))
  if len(nodes)>4096 or len(tr.body)!=1 or not isinstance(tr.body[0],ast.FunctionDef):return False,None
  f=tr.body[0];a=f.args
  if f.name!='solve' or a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or f.decorator_list:return False,None
  env={}
  def expr(n):
   if isinstance(n,ast.Constant):return bounded(n.value)
   if isinstance(n,ast.Name):
    if n.id not in env:raise Unsupported()
    return env[n.id]
   if isinstance(n,ast.UnaryOp) and type(n.op)in(ast.UAdd,ast.USub):return bounded((operator.pos if isinstance(n.op,ast.UAdd) else operator.neg)(expr(n.operand)))
   if isinstance(n,ast.BinOp):
    l,v=expr(n.left),expr(n.right)
    if isinstance(n.op,ast.Pow):
     if int(v)!=v or abs(v)>12:raise Unsupported()
     fn=operator.pow
    else:fn=BINARY.get(type(n.op))
    if fn is None:raise Unsupported()
    return bounded(fn(l,v))
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
    fn=CALLS.get(n.func.id)
    if fn is None or n.keywords or not n.args:raise Unsupported()
    vals=[expr(x) for x in n.args]
    if fn is round:
     if len(vals)not in(1,2)or(len(vals)==2 and(type(vals[1])is not int or abs(vals[1])>12)):raise Unsupported()
    elif fn in(int,float,abs)and len(vals)!=1:raise Unsupported()
    return bounded(fn(*vals))
   raise Unsupported()
  for k,s in enumerate(f.body,1):
   if k>512:raise Unsupported()
   if isinstance(s,ast.Expr) and isinstance(s.value,ast.Constant) and isinstance(s.value.value,str):continue
   if isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=='print':continue
   if isinstance(s,ast.Assign):
    if not s.targets or any(not isinstance(x,ast.Name)or x.id in {*CALLS,'print'} for x in s.targets):raise Unsupported()
    val=expr(s.value)
    for x in s.targets:env[x.id]=val
   elif isinstance(s,ast.AnnAssign):
    if not isinstance(s.target,ast.Name)or s.value is None or s.target.id in {*CALLS,'print'}:raise Unsupported()
    env[s.target.id]=expr(s.value)
   elif isinstance(s,ast.Return):return True,expr(s.value)
   else:raise Unsupported()
   if len(env)>256:raise Unsupported()
 except (Unsupported,ArithmeticError,ValueError,TypeError,SyntaxError,RecursionError,MemoryError):return False,None
 return False,None
def equal(a,b):return a==b if type(a)is type(b)is int else math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
def select(codes,scores):
 groups=[];support=0
 for i,code in enumerate(codes):
  ok,value=evaluate(code)
  if not ok:continue
  support+=1;g=next((x for x in groups if equal(x['value'],value)),None)
  if g is None:g={'value':value,'indices':[]};groups.append(g)
  g['indices'].append(i)
 if not groups:return 0,{'supported_candidates':0,'winning_votes':0,'numeric_groups':0}
 g=max(groups,key=lambda x:(len(x['indices']),max(scores[i]for i in x['indices'])));idx=max(g['indices'],key=lambda i:scores[i]);return idx,{'supported_candidates':support,'winning_votes':len(g['indices']),'numeric_groups':len(groups)}
rec=rd(E/'recovery.json');audit=rd(E/'recovery_proof_readback.json');term=rd(D/'session_result.json');m=rd(T/'own_portable_notebook_proof_manifest_c.json');dep=m['candidate_manifest']['deployment'];phase=term['lanes']['gpu0']['phases']['notebook'];side=rd(D/'gpu0/run/candidate_output/candidate_timing.json');execution=rd(D/'gpu0/executed.json')
ck('root_recovery_complete_readback_qualified',rec['complete']is rec['proof_qualified']is audit['qualified']is True and rec['proof_readback_sha256']==sha(E/'recovery_proof_readback.json'))
ck('current_exact_identity_before_after',rec['identity_before']==rec['identity_after'] and rec['identity_after']['numeric_id']==140522384 and rec['identity_after']['version']==1 and rec['identity_after']['driver_sha256']=='6855e945794976f95d57e729623221aa670cd8f9b7a75f93193aa74e1bfbbe03')
i=rec['identity_after'];ck('private_offline_T4_exact_mount',i['private']is i['gpu']is True and i['internet']is i['tpu']is False and i['machine_shape']=='NvidiaTeslaT4' and i['dataset_sources']==['arsen4ikvar/nlp-hw02-fullpop512-n16-proof-tools-20261024a'] and not any(i[k] for k in ('kernel_sources','competition_sources','model_sources')))
ck('terminal_C_bound_complete',term['schema']=='own_portable_notebook_proof_result_c_v1'and term['manifest_sha256']==sha(T/'own_portable_notebook_proof_manifest_c.json')=='874901fcf255c74f5f485116e46d9ba0e7f56d0d8011260cc87ea1b63aa96594'and term['driver_sha256']==i['driver_sha256'] and all(term[k]is True for k in ('complete','work_complete','cleanup_verified','artifact_hashes_complete','materialized_inputs_verified'))and term['phase']=='finished')
ck('exact14mapped_plus_terminal',len(term['artifact_sha256'])==14 and set(rec['files'])==set(term['artifact_sha256'])|{'session_result.json'} and set(str(p.relative_to(D))for p in D.rglob('*')if p.is_file())==set(rec['files']))
for n,v in rec['files'].items():ck('retained:'+n,(D/n).stat().st_size==v['size_bytes']and sha(D/n)==v['sha256']and(n=='session_result.json'or {k:v[k]for k in ('sha256','size_bytes')}==term['artifact_sha256'][n]))
ck('aggregate_bytes',sum(v['size_bytes']for v in rec['files'].values())==rec['total_bytes']==8952757 and sum(v['size_bytes']for v in term['artifact_sha256'].values())==term['artifact_total_bytes'])
ck('all21stage_payloads_unchanged',len(m['payload_sha256'])==21 and all(sha(T/n)==h for n,h in m['payload_sha256'].items()))
ck('exact_zip_notebook',sha(T/'candidate_bundle.bin')==term['candidate_bundle_sha256']==audit['candidate_bundle_sha256']=='22a91d853126a017b67a1a2cd290ff5c05794ee7a3764a9dd8250c4796e27918'and sha(T/'notebook.bin')==sha(D/'gpu0/run/inference.ipynb')==term['candidate_notebook_sha256']=='9b13a8a9f2e971a2df1de8ef30fa803a1c43db662a3b0b928e8aa6b7ca0e9730')
with zipfile.ZipFile(T/'candidate_bundle.bin')as z:
 ck('zip11member_no_duplicates',len(z.namelist())==len(set(z.namelist()))==11)
 ck('zip_manifest_deployment',json.loads(z.read('candidate_manifest.json'))==m['candidate_manifest']);questions=json.loads(z.read('test_questions.json'))
ck('actual_question_only1919_order',len(questions)==1919 and all(isinstance(q,str)for q in questions)and ordered(questions)==dep['test_questions_sha256']=='6dd86866b2ed86a09781b7f471fc2fce5c00975503c4a85f1ec959dae0b93279')
ck('same_completed_fullpopulation512_N16_eager',term['deployment']==side['deployment']==audit['deployment']==dep and dep['model_sha256']=='8f72f22a383bfe97aa251f5c5dabf895c118b375530bc73fb99893d66272131e'and dep['checkpoint_step']==525952 and dep['training_family']=='wide_fullpopulation_continuation_a_v1'and dep['training_complete']is True and dep['engine']=='eager'and dep['samples_per_question']==16 and dep['parameters']==30705728 and dep['unique_xsa_scalars']==64)
ck('six_root_receipts_exact',all(sha(T/('receipt-'+k+'.json'))==h for k,h in dep['root_receipts'].items()))
lin=rd(T/'receipt-lineage.json');par=rd(T/'paired_worker_result.json');ck('paired_ref_exact_and_parity',sha(T/'paired_worker_result.json')==m['reference']['worker_result_sha256'] and par['schema']=='fullpopulation512_count_worker_result_a_v1'and par['parity']==audit['actual_reference_four_pair_parity']and par['parity']['passed']is True and par['parity']['actual_rows']==4 and par['parity']['active_target_tokens']==399)
for k,limit in [('fp32',.001),('fp16',.125)]:
 v=par['parity']['results'][k];ck('fourpair:'+k,v['finite']is v['passed']is v['elementwise_close']is True and math.isfinite(v['max_abs_logit_error'])and v['max_abs_logit_error']<=limit and v['cache_depth']==4)
ck('backend_exact_oneT4',side['backend']['torch']==term['reference_backend']['torch']=='2.10.0+cu128'and side['backend']['cuda']==term['reference_backend']['cuda']=='12.8'and side['backend']['gpu']==term['reference_backend']['gpu']=='Tesla T4'and side['backend']['visible_gpu_count']==1 and term['lanes']['gpu1']['unused']is True)
ck('cleanup_exit0_no_signals',len(term['cleanup'])==1 and all(v['verified']is v['reaped']is v['group_exited']is True and not v['errors']and not v['signals']for v in term['cleanup'])and phase['actual_exit']==0 and phase['complete']is True and phase['cleanup']['verified']is True and not phase['cleanup']['signals'])
original=rd(T/'notebook.bin');executed=rd(D/'gpu0/executed.ipynb');oc=[c for c in original['cells']if c['cell_type']=='code'];ec=[c for c in executed['cells']if c['cell_type']=='code']
ck('exact3cells_source_and_execution',len(oc)==len(ec)==3 and [c['execution_count']for c in ec]==[1,2,3]and all(a['source']==b['source']and not any(o.get('output_type')=='error'for o in b['outputs'])for a,b in zip(oc,ec)))
for c in oc:ast.parse(''.join(c['source']))
ck('executor_exact_input_output',execution['complete']is True and execution['input_sha256']==sha(T/'notebook.bin')and execution['output_sha256']==sha(D/'gpu0/executed.ipynb')and execution['executed_code_cells']==3 and phase['executor']==execution)
ck('actual_sidecar_exact',phase['candidate_timing']==side and side['schema']=='own_portable_candidate_execution_a_v1'and side['rows']==1919 and side['execution_complete']is True and side['reproduction']['matches']is side['runtime_gate_passed']is side['notebook_within180']is True)
clocks={'generation_selection_serialization_seconds':side['generation_selection_serialization']['complete_path_seconds'],'strongest_notebook_seconds':side['timing_seconds']['notebook_start_through_output_readback_hashes'],'executor_seconds':execution['elapsed_seconds'],'whole_child_seconds':phase['whole_child_seconds']};ck('all4inclusive_clocks_le180',clocks==phase['clocks']==audit['clocks']and all(type(v)in(int,float)and math.isfinite(v)and 0<v<=180 for v in clocks.values()))
selector=R/'.worktrees/strong-inference-fallback-20261022/src/nlp_hw02/sampling_candidate.py';ck('frozen_selector_SHA',sha(selector)=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5');ck('restricted_arithmetic_source_SHA',sha(R/'src/nlp_hw02/arithmetic.py')=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6')
summary={};candidate_count=0
for label,qs,predname,key in [('warmup_n16',questions[:4],'warmup_n16_predictions.jsonl','warmup_first4'),('test_n16',questions,'predictions.jsonl','generation_selection_serialization')]:
 out=D/'gpu0/run/candidate_output';timing=rd(out/(label+'_timing.json'));rows=[json.loads(x)for x in (out/(label+'_candidates.jsonl')).read_text().splitlines()];preds=[json.loads(x)for x in (out/predname).read_text().splitlines()];prog=rd(out/(label+'_progress.json'));ck(label+':rows_timing_progress',len(rows)==len(preds)==len(qs)==timing['rows']==prog['completed_rows']==prog['requested_rows'] and timing==side[key]and timing['samples_per_question']==16 and timing['total_candidates']==16*len(qs)and timing['candidate_sha256']==sha(out/(label+'_candidates.jsonl'))and timing['prediction_sha256']==sha(out/predname)and timing['ordered_questions_sha256']==ordered(qs))
 ck(label+':exact_policy',all(timing[k]==v for k,v in dep['generation'].items()))
 caps=selected_caps=unsupported=0;codes=[]
 for j,(q,row,pred)in enumerate(zip(qs,rows,preds)):
  assert row['row_index']==j and row['question_sha256']==hashlib.sha256(q.encode()).hexdigest()and row['samples']==16
  c,s,l,cap=row['candidates'],row['scores'],row['generated_lengths'],row['capped'];assert len(c)==len(s)==len(l)==len(cap)==16 and all(isinstance(x,str)for x in c)and all(type(x)in(int,float)and math.isfinite(x)for x in s)and all(type(x)is int and 0<=x<=192 for x in l)and all(type(x)is bool for x in cap)and all((n==192)if b else(n<192)for n,b in zip(l,cap))
  idx,info=select(c,s);assert idx==row['selected_index']and all(row[k]==v for k,v in info.items())and set(pred)=={'code'}and pred['code']==c[idx]
  codes.append(pred['code']);candidate_count+=16;caps+=sum(cap);selected_caps+=cap[idx];unsupported+=info['supported_candidates']==0
 ck(label+':all_rows_selector_projection_lengths_caps',timing['all_candidate_capped_rows']==caps and timing['selected_candidate_capped_rows']==selected_caps and timing['without_supported_candidate']==unsupported)
 actual={'rows':len(qs),'total_candidates':16*len(qs),'code_only_jsonl_sha256':sha(out/predname),'ordered_codes_sha256':ordered(codes),'ordered_questions_sha256':ordered(qs),'candidate_sha256':sha(out/(label+'_candidates.jsonl')),'all_candidate_caps':caps,'selected_caps':selected_caps};ck(label+':parent_readback_population_match',actual==audit['populations'][label]==phase['actual_test'if label.startswith('test')else'actual_warmup']);summary[label]={**actual,'unsupported_selected_rows':unsupported}
ck('all30768_candidates_replayed',candidate_count==30768==audit['all_candidates_selector_replayed'])
a=summary['test_n16'];ck('exact1919_reproduction',a['code_only_jsonl_sha256']==dep['reproduction']['test_code_only_jsonl_sha256']==side['output_sha256']=='67990faa733f8adfb52d6891d733586599ce041d9c54f6b61080c2ce54501504'and a['ordered_codes_sha256']==dep['reproduction']['test_ordered_codes_sha256']==side['ordered_codes_sha256']=='20dfd534b2332ab27b0e9d9e3332235ea6d85caf60c7cf87db573b8b9f4a5f12'and a['ordered_questions_sha256']==dep['reproduction']['test_ordered_questions_sha256'])
report={'schema':'own_fullpopulation512_n16_exact_notebook_actual_independent_review_v1','verdict':'PASS_ACTUAL_EXACT_ONE_T4_PROOF','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'check_count':len(checks),'recovery_sha256':sha(E/'recovery.json'),'root_readback_sha256':sha(E/'recovery_proof_readback.json'),'terminal_sha256':sha(D/'session_result.json'),'stage_manifest_sha256':sha(T/'own_portable_notebook_proof_manifest_c.json'),'model_sha256':dep['model_sha256'],'selected_step':525952,'training_endpoint_step':531072,'architecture':'own512_gelu','training_family':dep['training_family'],'engine':'eager','samples_per_question':16,'question_batch_size':32,'clocks':clocks,'generation_graph_setup_close_rename_seconds':side['timing_seconds']['generation_selection_graph_setup_serialization_and_rename'],'backend':side['backend'],'populations':summary,'candidates_independently_replayed':candidate_count,'actual_retained_files':len(rec['files']),'actual_terminal_mapped_files':len(term['artifact_sha256']),'actual_retained_bytes':rec['total_bytes'],'artifact_sha256':rec['files'],'findings':[],'limits':['Read-only local JSON/JSONL/hash/AST audit; no recovery/helper/source imports, candidate execution, model/PT member open/deserialization, forward, provider requests or canonical mutation.','Numeric selector independently reconstructed with a bounded non-executing AST arithmetic interpreter; generated programs were never compiled/evaluated/executed.','Actual fresh one-visibleT4 notebook matched already-accepted same-model/N16/eager reference/parity. It did not rerun validation1792; quality remains separately measured1135legacy/1136aug on fixed1529.','Same model and q32 policy only; no qualification of other counts/question batches/engines/backends, no official judge score or promotion decision.'],'canonical_promotion':False}
output=E/'actual_proof_independent_review.json';assert not output.exists();report['audit_script_sha256']=sha(Path(__file__));output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'verdict':report['verdict'],'check_count':len(checks),'receipt':str(output.relative_to(R)),'sha256':sha(output),'clocks':clocks}))
