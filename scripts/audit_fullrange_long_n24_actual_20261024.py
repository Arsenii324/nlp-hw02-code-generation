"""Existing actual paired-N24 outputs: bounded safe-AST replay, never generated Python execution."""
from pathlib import Path
from collections import Counter
from functools import lru_cache
from datetime import datetime,timezone
import ast,hashlib,json,math,sys,time
R=Path(__file__).resolve().parents[1];W=R/'.worktrees/strong-inference-fallback-20261022';B=R/'artifacts/kaggle/fullrange-long-n24-v1-a-diagnostics';T=R/'artifacts/kaggle/fullrange-long-n24-20261024a/tools';START=time.monotonic();checks={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def ordered(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def check(k,v):checks[k]=bool(v);require(v,k)
sys.path.insert(0,str(R/'src'))
from nlp_hw02.arithmetic import evaluate_code as rawlegacy
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign as rawaug
@lru_cache(maxsize=None)
def legacy(s):return rawlegacy(s)
@lru_cache(maxsize=None)
def aug(s):
 a=legacy(s);b=rawaug(s);require(not a.supported or b.supported and a.value==b.value,'Aug altered old supported value');return b
sources={'legacy':R/'src/nlp_hw02/arithmetic.py','augassign':R/'src/nlp_hw02/arithmetic_augassign_audit.py','selector':W/'src/nlp_hw02/sampling_candidate.py'}
check('three_scoring_sourcepins',sha(sources['legacy'])=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(sources['augassign'])=='fa26100d91331f2ffe684f7729ad37870c6ac53a0f8834cf6922f430908ecc16' and sha(sources['selector'])=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5')
# Only two pinned selector function definitions compiled; no dataset program eval/exec.
tree=ast.parse(sources['selector'].read_text());fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('equal','select_plurality')];require(len(fn)==2,'Two selector functions');ns={'math':math,'evaluate_code':legacy};exec(compile(ast.Module(body=fn,type_ignores=[]),str(sources['selector']),'exec'),ns);select=ns['select_plurality'];equal=ns['equal']
rr=R/'reports/evidence/long_n24_screens_20261024a/fullrange/recovery.json';receipt=load(rr);terminal=load(B/'session_result.json');manifest=load(T/'fullrange_pair_sampling_manifest_a.json');amap=terminal['artifact_sha256']
check('recovery_receipt_exact',sha(rr)=='a9f3fb878662f7883a536ab640638073a05f8a4c6439ab6c57423c5dbdbf6769' and receipt['complete'])
check('terminal_exact',sha(B/'session_result.json')=='392696ad5ef00cacc206330adc7a0a511ba7fb807ae2436f91987835f91ce5de')
check('all31mapped_files_SHA',len(amap)==31 and all(sha(B/n)==h for n,h in amap.items()))
files=receipt['files']
check('32local_files_exact_allowlist',len(files)==32 and set(files)=={str(p.relative_to(B)) for p in B.rglob('*') if p.is_file()}==set(amap)|{'session_result.json'})
check('all32receipt_SHA_sizes_bytes',all(sha(B/n)==x['sha256'] and (B/n).stat().st_size==x['bytes'] for n,x in files.items()) and sum(x['bytes'] for x in files.values())==receipt['total_bytes']==51239517)
check('actualV1_identity_stable',receipt['before']==receipt['after'] and receipt['before']['version']==1 and receipt['before']['numeric_id']==140495822 and receipt['before']['driver_sha256']==terminal['parent_sha256']=='9786d9dddb8123d5e893a9bb59ef50b227237e3473534c3aaf98ffa478f3336f' and receipt['before']['private'] is True and receipt['before']['internet'] is False)
check('complete_both_groups_clean_no_signal',terminal['complete'] is terminal['artifact_hashes_complete'] is True and terminal['phase']=='finished' and len(terminal['children'])==len(terminal['cleanup'])==2 and all(x['complete'] and x['actual_exit']==0 for x in terminal['children']) and all(x['reaped'] and x['group_exited'] and not x['errors'] and not x['signals'] for x in terminal['cleanup']))
check('bound_manifest_and_source',sha(T/'fullrange_pair_sampling_manifest_a.json')==terminal['tools_manifest_sha256']=='3d8aae3155199e5aff22d103a001ccdf16eab9212439f9cde5b4807ff28943d9' and terminal['source_freeze_sha256']=='49768fc7374b5896d2c3caea7edf9c2a42865cc5072454acbb2cc4452a0d6eab' and terminal['training_terminal_sha256']=='9c5ae9db4160739d374d8b5b421519842a7959759f09bac2944b56d3d40eb328')
check('all60nonPTstage_payload_source_input_hashes',len(manifest['payload_sha256'])==62 and sum(Path(n).suffix!='.pt' for n in manifest['payload_sha256'])==60 and all(sha(T/n)==h for n,h in manifest['payload_sha256'].items() if Path(n).suffix!='.pt'))
qs=load(T/'inputs/validation_questions.json');testtools=W/'artifacts/colab-adaptive24-tools-20261023c';tm=load(testtools/'colab_adaptive24_manifest.json');test=load(testtools/'test_questions.json');check('same1792and1919goldfree_order',len(qs)==1792 and len(test)==1919 and sha(testtools/'test_questions.json')==tm['input_sha256']['test_questions.json'] and ordered(test)=='6dd86866b2ed86a09781b7f471fc2fce5c00975503c4a85f1ec959dae0b93279')
qsets={'validation_n24':qs,'warmup_n24':test[:4],'test_n24':test};qh=[hashlib.sha256(q.encode()).hexdigest() for q in qs];refsfile=R/'reports/evidence/primary_reference_consensus.json';refs=load(refsfile)['groups_by_question_sha256'];check('fixed_original1529refs',sha(refsfile)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119' and sum(refs[q]['status']=='supported_consensus' for q in qh)==1529)
allpred={};poolsummary={};runtime={};replay={'rows':0,'candidates':0};events=rows(B/'session_events.jsonl');policy=load(T/'contracts/full-remaining.json')['generation']
for role in manifest['role_order']:
 d=B/('run-'+role);result=load(d/'fullrange_sampling_result.json');child=next(x for x in terminal['children'] if x['run_id']==role);parity=load(d/'actual_parity.json');identity=result['identity'];rrole=manifest['roles'][role]
 check(role+'_result_identity_sources',sha(d/'fullrange_sampling_result.json')==child['worker_result_sha256'] and result['complete'] and result['phase']=='finished' and result['parity_passed_before_sampling'] and not result['training'] and result['gold_used_for_selection'] is False and result['producer_sha256']==terminal['worker_sha256']==manifest['payload_sha256']['scripts/probe_fullrange_selected_sampling_c.py'] and result['adapter_sha256']==terminal['adapter_sha256']==manifest['payload_sha256']['scripts/fullrange_sampling_contract_c.py'] and result['contract_sha256']==child['contract_sha256'] and identity==child['worker_identity'] and all(identity[k]==rrole[k] for k in ['run_id','pointer_sha256','model_sha256','step','validation_loss','root_cpu_integrity_sha256']) and identity['training_terminal_sha256']==terminal['training_terminal_sha256'])
 check(role+'_actualparity399_fourdistinctcache',result['parity']==child['actual_parity']==parity and parity['passed'] and parity['actual_rows']==4 and parity['prefix_width']==192 and parity['active_target_tokens']==399 and parity['topology']['unique_parameters']==17524256 and parity['topology']['unique_xsa_scalars']==32 and parity['topology']['decoder_cache_depth']==4 and parity['topology']['gates']=={} and all(z=={'stored_blocks':4,'effective_blocks':4,'unique_ffns':4} for z in parity['topology']['topology'].values()) and all(z['passed'] and z['finite'] and z['elementwise_close'] and z['cache_depth']==z['distinct_self_K_buffers']==z['distinct_self_V_buffers']==4 and z['active_argmax_disagreements']==0 and z['max_abs_logit_error']<=parity['tolerances'][dt]['max_abs'] for dt,z in parity['results'].items()))
 check(role+'_hardware_software',result['hardware']=={'gpu':'Tesla T4','visible_gpu_count':1} and result['software']['torch']=='2.10.0+cu128' and result['software']['cuda']=='12.8' and result['observed_backend']['matches_training_origin'] and result['observed_backend']['driver']=='580.178.04')
 runtime[role]={'identity':identity,'software':result['software'],'observed_backend':result['observed_backend'],'parity':parity,'model_tokenizer_load_seconds':result['model_tokenizer_load_seconds'],'worker_main_seconds':result['whole_worker_main_seconds']}
 for label,questions in qsets.items():
  cp=d/(label+'_candidates.jsonl');pp=d/(label+'_predictions.jsonl');pool=rows(cp);pred=rows(pp);timing=load(d/(label+'_timing.json'));progress=load(d/(label+'_progress.json'));require(timing==result['paths'][label]==child['worker_paths'][label],'Timing result equality');require(len(pool)==len(pred)==timing['rows']==len(questions),'Pool counts');caps=selectedcaps=unsupported=0
  for i,(row,pr,q) in enumerate(zip(pool,pred,questions)):
   require(row['row_index']==i and row['question_sha256']==hashlib.sha256(q.encode()).hexdigest() and row['samples']==24,'Exact question/order/count')
   require(all(len(row[k])==24 for k in ['candidates','scores','capped','generated_lengths']),'24vectors')
   require(all(isinstance(x,str) for x in row['candidates']) and all(type(x) in (int,float) and math.isfinite(x) for x in row['scores']) and all(type(x) is bool for x in row['capped']) and all(type(x) is int and 0<=x<=192 for x in row['generated_lengths']) and all(cap==(length==192) for cap,length in zip(row['capped'],row['generated_lengths'])),'Types finite cap-EOS length coherence')
   index,info=select(row['candidates'],row['scores']);require(index==row['selected_index'] and all(row[k]==v for k,v in info.items()),'Exact frozen selector replay');require(set(pr)=={'code'} and isinstance(pr['code'],str) and pr['code']==row['candidates'][index],'Codeonly selected projection')
   caps+=sum(row['capped']);selectedcaps+=row['capped'][index];unsupported+=info['supported_candidates']==0;replay['rows']+=1;replay['candidates']+=24
  require(all(timing[k]==v for k,v in policy.items()) and timing['samples_per_question']==24 and timing['total_candidates']==len(questions)*24 and timing['all_candidate_capped_rows']==caps and timing['selected_candidate_capped_rows']==selectedcaps and timing['without_supported_candidate']==unsupported and timing['candidate_sha256']==sha(cp) and timing['prediction_sha256']==sha(pp) and timing['ordered_questions_sha256']==ordered(questions),'Timing/count/order/hash consistency')
  require(math.isfinite(timing['complete_path_seconds']) and 0<timing['complete_path_seconds']<=180 and timing['complete_path_within180'] is True and progress['completed_rows']==progress['requested_rows']==len(questions),'Actual complete timing/progress')
  poolsummary[role+'/'+label]={'rows':len(pool),'candidates':len(pool)*24,'all_capped':caps,'selected_capped':selectedcaps,'without_supported_candidate':unsupported,'candidate_sha256':sha(cp),'prediction_sha256':sha(pp),'ordered_codes_sha256':ordered([x['code'] for x in pred]),'timing':timing}
  if label=='validation_n24':allpred[role]=pred
 check(role+'_all3fullpaths_replayed',True)
old=R/'artifacts/kaggle/cooldown-paired-n24-v1-a-diagnostics';ot=load(old/'session_result.json');op=old/'run-cooldown-half/validation_n24_predictions.jsonl';oldpred=rows(op);check('selected_half_exactbaseline1792',sha(op)==ot['artifact_sha256']['run-cooldown-half/validation_n24_predictions.jsonl'] and len(oldpred)==1792 and all(set(x)=={'code'} for x in oldpred));allpred['selected-half']=oldpred
maskfile=R/'reports/evidence/gemini_reference_screen_20261024/auxiliary_reference_screen_mask.json';mask=load(maskfile);excluded=set(mask['excluded_global_indices']);auxrows=[i for i,q in enumerate(qh) if i not in excluded and refs[q]['status']=='supported_consensus'];check('fixedauxiliary1141',sha(maskfile)=='0e16c82bc9711fa3c71e56a74dac09c7d4bbde1132397dc139d697dbf9b434e2' and len(auxrows)==1141)
quality={};outcomes={}
for name,pred in allpred.items():
 cats={x:[] for x in ['legacy','augassign_fixed_output','augassign_additive_1e6']};support=Counter();reasons={x:Counter() for x in cats}
 for pr,q in zip(pred,qh):
  a=legacy(pr['code']);b=aug(pr['code']);support['legacy_supported']+=a.supported;support['augassign_supported']+=b.supported
  for label,v in [('legacy',a),('augassign_fixed_output',b),('augassign_additive_1e6',b)]:
   if refs[q]['status']!='supported_consensus':cats[label].append(None);continue
   target=refs[q]['consensus_value'];ok=(abs(v.value-target)<=1e-6+1e-6*abs(target) if label=='augassign_additive_1e6' else equal(v.value,target)) if v.supported else False;cats[label].append('correct' if ok else 'wrong_numeric' if v.supported else 'unsupported')
   if not v.supported:reasons[label][v.reason]+=1
 quality[name]={'fixed1529_categories':{k:dict(Counter(v for v in a if v is not None)) for k,a in cats.items()},'auxiliary1141_categories':{k:dict(Counter(a[i] for i in auxrows)) for k,a in cats.items()},'all1792support':dict(support),'unsupported_reasons_fixed1529':{k:dict(v) for k,v in reasons.items()}};outcomes[name]=cats
check('selectedhalf_reproduced1117legacy',quality['selected-half']['fixed1529_categories']['legacy']['correct']==1117)
paired=[]
for left,right in [('selected-half','full-remaining'),('selected-half','half-remaining'),('full-remaining','half-remaining')]:
 for diag in outcomes[left]:
  for population,indices in [('fixed1529',range(len(qh))),('auxiliary1141',auxrows)]:
   pairs=Counter((outcomes[left][diag][i],outcomes[right][diag][i]) for i in indices if outcomes[left][diag][i] is not None);wins=sum(v for (a,b),v in pairs.items() if a!='correct' and b=='correct');loss=sum(v for (a,b),v in pairs.items() if a=='correct' and b!='correct');n=wins+loss
   paired.append({'left':left,'right':right,'diagnostic':diag,'population':population,'rows':sum(pairs.values()),'right_only_correct':wins,'left_only_correct':loss,'net_right':wins-loss,'both_correct':pairs['correct','correct'],'descriptive_exact_discordance_p':min(1.,2*sum(math.comb(n,k) for k in range(min(wins,loss)+1))/2**n) if n else 1.,'category_transitions':{a+'->'+b:v for (a,b),v in pairs.items()}})
check('all6newpools_7430rows178320candidates',replay=={'rows':7430,'candidates':178320})
report={'schema':'fullrange_pair_n24_actual_quality_independent_review_v1','utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_ACTUAL_RECOVERY_PARITY_ORDER_POOLS_SELECTOR_AND_FIXED_DIAGNOSTICS','checks':checks,'check_count':len(checks),'all_checks_pass':all(checks.values()),'terminal_sha256':sha(B/'session_result.json'),'recovery_receipt_sha256':sha(rr),'actual_file_count':32,'actual_mapped_file_count':31,'actual_bytes':51239517,'all_recovered_sha256':{str(p.relative_to(B)):sha(p) for p in B.rglob('*') if p.is_file()},'manifest_sha256':sha(T/'fullrange_pair_sampling_manifest_a.json'),'source_audit_path':str(Path(__file__).relative_to(R)),'source_audit_sha256':sha(Path(__file__)),'scoring_source_sha256':{n:sha(p) for n,p in sources.items()},'reference_sha256':sha(refsfile),'auxiliary_mask_sha256':sha(maskfile),'replay':replay,'runtime':runtime,'pools':poolsummary,'quality':quality,'paired':paired,'lifecycle':{'session_script_seconds':terminal['session_script_seconds'],'bounds_seconds':terminal['bounds_seconds'],'cleanup':terminal['cleanup'],'worker_complete_parent_events':[e for e in events if e['event']=='worker_complete']},
'metadata_caveats':['Root recovered32 files in one terminal-first operation; zero-length logs accepted. Reviewer made no provider calls.','Progress complete=false/parent done=false are stale telemetry; exactcounts/resultcomplete/exit0/cleanPGIDs establishactualcompletion.','Inherited timingprose all8-depth isstale; actual4encoder+4decoder/cache4 and32XSA verified.','31terminalmap excludes2runtimecontractJSONs; staged originals andmatchingchild/worker contractSHA arebound. Those2runtimebytes notrecovered.'],
'limits':['Original1529 noisy supplied-reference consensus/restrictedAST diagnostics; notofficialpass1. No filtering/targetrewrite/generatedPythonexecution.','Production legacyselector unchanged; AugAssign extends onlygenerated-outputinterpretation, additive1e-6 diagnostic separate. Repeatedvalidation/modelselection/onesample-seed makespairedpdescriptive.','Current pairH184768/equalfullpopulationdraws/CEbest529320/endpoint529768; baseline6MhalfH72384/CEbest415656 differsbudget/population/schedule, notisolatedcausaleffect.','Actualtestcomplete-pathtimes exclude load/parity/warmup/laterhashes; not chosen-bytefourclockexactnotebookproof. Newwinnerneedsnewportablepackage+actualoneT4proof. No model/PT/arrays/provider/test/canonical/HANDOFF mutation byreviewer.','No test outcomes used; question order/generation/timingonly. Existingcanonical root-owned.'],
'audit_seconds':time.monotonic()-START}
out=R/'reports/evidence/long_n24_screens_20261024a/fullrange/quality_independent_review.json';require(not out.exists(),'Do not overwrite frozen evidence');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'path':str(out.relative_to(R)),'sha256':sha(out),'checks':len(checks),'quality':{k:{d:v['correct'] for d,v in x['fixed1529_categories'].items()} for k,x in quality.items()},'paired':[x for x in paired if x['diagnostic']=='legacy'],'audit_seconds':report['audit_seconds']},indent=2))
