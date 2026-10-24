"""Audit preserved actual Colab outputs only: SHA/order/selector replay and fixed diagnostics.

No model imports, weights reads, generation, fixtures, cloud or dataset-code execution.
"""
from pathlib import Path
import ast,hashlib,json,zipfile,math,sys,time
from collections import Counter
from datetime import datetime,timezone
from functools import lru_cache
from nlp_hw02.arithmetic import evaluate_code as raw_legacy
from nlp_hw02.arithmetic_augassign_audit import evaluate_code_augassign as raw_aug
R=Path.cwd();B=R/'artifacts/colab/adaptive24-20261023c';V=B/'verified-result';W=R/'.worktrees/strong-inference-fallback-20261022';T=W/'artifacts/colab-adaptive24-tools-20261023c';started=time.monotonic();checks={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(v) for v in Path(p).read_text().splitlines()]
def ordered(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def require(x,msg):
 if not x:raise ValueError(msg)
@lru_cache(maxsize=None)
def legacy(s):return raw_legacy(s)
@lru_cache(maxsize=None)
def aug(s):
 a=legacy(s);b=raw_aug(s);require(not a.supported or (b.supported and a.value==b.value),'AugAssign altered accepted value');return b
# Load only the two pinned selector function AST nodes, no module/model import.
require(sha(R/'src/nlp_hw02/arithmetic.py')=='e01078307ec395b41afe03b588e014b7498ee968ed6ed2f3d13f8105c45663d6' and sha(R/'src/nlp_hw02/arithmetic_augassign_audit.py')=='fa26100d91331f2ffe684f7729ad37870c6ac53a0f8834cf6922f430908ecc16','Exact diagnostic evaluators');checks['exact_evaluator_source_pins']=True
selector=W/'src/nlp_hw02/sampling_candidate.py';require(sha(selector)=='250d34bee2dc60847a65a7d5a784fac261a77db153a4d49f0687a82fe5a98fe5','Selector source pin')
st=ast.parse(selector.read_text());functions=[n for n in st.body if isinstance(n,ast.FunctionDef) and n.name in ('equal','select_plurality')];require(len(functions)==2,'Selector functions');ns={'math':math,'evaluate_code':legacy};exec(compile(ast.Module(body=functions,type_ignores=[]),str(selector),'exec'),ns);equal=ns['equal'];select=ns['select_plurality']
owner=load(B/'owner-result.json');receipt=load(B/'archive-receipt.json');outer=load(B/'session-result.json');inner=load(V/'session_result.json');manifest=load(V/'source/colab_adaptive24_manifest.json');worker=load(V/'results/worker_result.json');admit=load(V/'results/adaptive_test1919_admission.json')
checks['31_actual_terminal_map_hashes']=len(outer['artifact_sha256'])==31 and all(sha(V/n)==h for n,h in outer['artifact_sha256'].items())
checks['32_receipt_actual_file_hashes']=len(receipt['members'])==receipt['member_count']==32 and all(sha(V/n)==h for n,h in receipt['members'].items())
checks['receipt_terminal_map_equal_plus_inner_terminal']=receipt['members']=={**outer['artifact_sha256'],'session_result.json':sha(V/'session_result.json')}
checks['actual_archive_SHA_size']=sha(B/'adaptive24-results.zip')==receipt['archive_sha256']==outer['archive_sha256'] and (B/'adaptive24-results.zip').stat().st_size==receipt['archive_bytes']==outer['archive_bytes']
with zipfile.ZipFile(B/'adaptive24-results.zip') as z:
 checks['all32_safe_unique_archived_members_actualSHA']=len(z.namelist())==len(set(z.namelist()))==32 and set(z.namelist())==set(receipt['members']) and all(not Path(n).is_absolute() and '..' not in Path(n).parts and hashlib.sha256(z.read(n)).hexdigest()==receipt['members'][n] for n in z.namelist())
 checks['no_weight_array_or_question_input_in_archive']=not any(Path(n).suffix in ('.pt','.npy','.npz','.parquet','.zip') or Path(n).name in manifest['input_sha256'] for n in z.namelist())
checks['legitimate_inner_terminal_phase_difference']=inner['phase']=='archive_ready' and outer['phase']=='finished' and all(inner[k]==outer[k] for k in inner if k!='phase')
checks['owner_recovery_report_consistent']=owner['archive_recovery_verified'] is True and owner['verified_member_count']==32 and owner['stop']['verified_absent'] is True and owner['remote_experiment_complete'] is False
checks['incomplete_worker_cleanly_reaped']=worker['complete'] is outer['complete'] is receipt['complete'] is False and outer['worker_returncode']==1 and outer['process_group_cleanup']['verified'] and outer['process_group_cleanup']['reaped'] and outer['process_group_cleanup']['group_exited'] and not outer['process_group_cleanup']['signals']
checks['manifest_actual_pin']=sha(V/'source/colab_adaptive24_manifest.json')==outer['input_manifest_sha256']==worker['manifest_sha256']==owner['manifest_sha256']=='ab8052d5dd72f35ab92c58ac622469329c31b012a05640434731429b87a19961'
checks['three_actual_source_pins']=all(sha(V/'source'/n)==manifest['payload_sha256'][n] for n in manifest['source_allowlist'])
with zipfile.ZipFile(T/'selected_bundle.bin') as runtime_zip:
 checks['actual_selected_runtime_source_only_ZIP_pins']=all(hashlib.sha256(runtime_zip.read(n)).hexdigest()==h for n,h in manifest['runtime_member_sha256'].items() if n.startswith('src/'))
checks['actual_question_inputs_pinned']=all(sha(T/n)==h for n,h in manifest['input_sha256'].items())
qsets={'validation1792':load(T/'validation_questions.json'),'test1919':load(T/'test_questions.json')};qsets['test_warmup4']=qsets['test1919'][:4]
checks['ordered1792_1919_question_inputs']=len(qsets['validation1792'])==1792 and len(qsets['test1919'])==1919 and all(isinstance(q,str) for a in qsets.values() for q in a) and ordered(qsets['validation1792'])==manifest['ordered_question_sha256']['validation'] and ordered(qsets['test1919'])==manifest['ordered_question_sha256']['test']
parity=load(V/'results/parity.json');checks['actual_parity_before_sampling']=worker['parity_passed_before_sampling'] is outer['parity_passed_before_sampling'] is True and worker['parity']==parity and parity['complete'] and parity['passed'] and parity['actual_rows']==4 and parity['active_target_tokens']==399 and all(v['passed'] and v['finite'] and v['cache_depth']==4 and v['distinct_self_K_buffers']==v['distinct_self_V_buffers']==4 and v['active_argmax_disagreements']==0 for v in parity['results'].values())
refs_path=R/'reports/evidence/primary_reference_consensus.json';require(sha(refs_path)=='deb96602c88b983ecfbf23738d2a64192057273798548ab61a7cdf7acfc7c119','Refs pin');refs=load(refs_path)['groups_by_question_sha256'];qh=[hashlib.sha256(q.encode()).hexdigest() for q in qsets['validation1792']];checks['same_original1529_scored_population']=sum(refs[q]['status']=='supported_consensus' for q in qh)==1529
policies={};pools={};preds={};quality={};outcomes={};replays={'rows':0,'candidates':0,'prefix8_replays':0};alltiming={}
for mode,phases in [('fixed24',['validation1792','test_warmup4','test1919']),('adaptive8-24',['validation1792','test_warmup4'])]:
 for phase in phases:
  d=V/'results'/mode/phase;poolpath=d/(phase+'_candidates.jsonl');predpath=d/(phase+'_predictions.jsonl');pool=rows(poolpath);pred=rows(predpath);timing=load(d/(phase+'_timing.json'));qs=qsets[phase];cohorts=Counter();counts=Counter();caps=selcaps=unsupported=0
  require(len(pool)==len(pred)==timing['rows']==len(qs),'Rowcounts')
  for i,(row,p,q) in enumerate(zip(pool,pred,qs)):
   require(row['row_index']==i and row['question_sha256']==hashlib.sha256(q.encode()).hexdigest(),'Row/question order')
   n=row['samples'] if mode=='fixed24' else row['samples_generated'];require(n in ([24] if mode=='fixed24' else [8,24]),'Counts')
   require(all(len(row[k])==n for k in ['candidates','scores','capped','generated_lengths']),'Vectors')
   require(all(isinstance(s,str) for s in row['candidates']) and all(math.isfinite(s) for s in row['scores']) and all(type(x) is bool for x in row['capped']) and all(type(x) is int and 0<=x<=192 for x in row['generated_lengths']),'Pool type/length cap')
   index,info=select(row['candidates'],row['scores']);require(index==row['selected_index'] and set(p)=={'code'} and p['code']==row['candidates'][index],'Selector/code projection')
   if mode=='fixed24':require(all(row[k]==v for k,v in info.items()),'Fixed diagnostics')
   else:
    pi,di=select(row['candidates'][:8],row['scores'][:8]);stop=di['supported_candidates']==8 and di['numeric_groups']==1;cat='unanimous8' if stop else 'no_supported' if di['supported_candidates']==0 else 'majority6_7' if di['winning_votes']>=6 else 'disputed'
    require(pi==row['prefix8_selected_index'] and di==row['prefix8_diagnostics'] and stop==row['stopped_at8'] and cat==row['prefix8_category'] and n==(8 if stop else 24) and info==row['final_diagnostics'],'Actual adaptive stop/expansion')
    require(all(row[k]==v for k,v in manifest['policy'].items() if k in ['temperature','top_k','question_batch_size','length_penalty','seed','max_new_tokens']),'Generation params');cohorts[cat]+=1;replays['prefix8_replays']+=1
   counts[n]+=1;caps+=sum(row['capped']);selcaps+=row['capped'][index];unsupported+=info['supported_candidates']==0;replays['rows']+=1;replays['candidates']+=n
  require(timing['candidate_sha256']==sha(poolpath) and timing['prediction_sha256']==sha(predpath) and timing['ordered_questions_sha256']==ordered(qs),'Timing source/order SHA')
  require(sum(k*v for k,v in counts.items())==timing['total_candidates'],'Timing candidates')
  if mode=='adaptive8-24':
   require(timing['stopped_at8_rows']==counts[8] and timing['expanded_to24_rows']==counts[24] and timing['prefix8_categories']==dict(cohorts) and timing['mean_candidates']==timing['total_candidates']/len(pool),'Adaptive totals')
   chunks=timing['chunks'];require([v['offset'] for v in chunks]==list(range(0,len(qs),32)) and sum(v['rows'] for v in chunks)==len(qs) and sum(v['expanded_rows'] for v in chunks)==counts[24],'Chunk compaction counts')
   require(all(c['expanded_rows']==sum(not r['stopped_at8'] for r in pool[c['offset']:c['offset']+c['rows']]) for c in chunks),'Perchunk expansions')
  else:require(timing['all_candidate_capped_rows']==caps and timing['selected_candidate_capped_rows']==selcaps and timing['without_supported_candidate']==unsupported,'Fixed totals')
  policies[mode+'/'+phase]={'rows':len(pool),'candidates':sum(k*v for k,v in counts.items()),'candidate_count_distribution':dict(counts),'mean_candidates':sum(k*v for k,v in counts.items())/len(pool),'prefix8_categories':dict(cohorts),'all_capped_candidates':caps,'selected_capped_rows':selcaps,'without_supported_candidate':unsupported,'seconds':timing['complete_path_seconds'],'peak_allocated_bytes':timing['peak_allocated_bytes'],'candidate_sha256':sha(poolpath),'prediction_sha256':sha(predpath)};alltiming[mode+'/'+phase]=timing
  if phase=='validation1792':pools[mode]=pool;preds[mode]=pred
checks['all_actual_pool_selector_order_counts_replayed']=True
checks['fixed_test1919_exact_canonical_SHA']=sha(V/'results/fixed24/test1919/test1919_predictions.jsonl')=='214bf02531a70cccc75a722c3a710d2466899d664c6d2d30c70a1c86f6eb7f29'
checks['fixed_validation_pool_predictions_exact_originalE']=sha(V/'results/fixed24/validation1792/validation1792_candidates.jsonl')=='e4983c5b3b32813d9b02373d45557df173f6f2debf9fb046a546ea60f31911e9' and sha(V/'results/fixed24/validation1792/validation1792_predictions.jsonl')=='1de89c6c0437b3ff3a653f8605386848aadf5a55ecf503dfac99aadebd65a427'
for mode in pools:
 cats={label:[] for label in ['legacy','augassign_fixed_output','augassign_additive_1e-6_diagnostic']};why={k:Counter() for k in cats};cohortcats={};supportall=Counter()
 for i,(row,p,q) in enumerate(zip(pools[mode],preds[mode],qh)):
  old=legacy(p['code']);new=aug(p['code']);supportall['legacy_supported']+=old.supported;supportall['augassign_supported']+=new.supported
  if refs[q]['status']!='supported_consensus':
   for v in cats.values():v.append(None)
   continue
  target=refs[q]['consensus_value']
  for label,value in [('legacy',old),('augassign_fixed_output',new),('augassign_additive_1e-6_diagnostic',new)]:
   ok=(abs(value.value-target)<=1e-6+1e-6*abs(target)) if value.supported and label=='augassign_additive_1e-6_diagnostic' else equal(value.value,target) if value.supported else False
   category='correct' if ok else 'wrong_numeric' if value.supported else 'unsupported';cats[label].append(category)
   if not value.supported:why[label][value.reason]+=1
  if mode=='adaptive8-24':cohortcats.setdefault(row['prefix8_category'],Counter())[cats['legacy'][-1]]+=1
 quality[mode]={'fixed1529_categories':{k:dict(Counter(x for x in a if x is not None)) for k,a in cats.items()},'all1792_support':dict(supportall),'unsupported_reasons':{k:dict(v) for k,v in why.items()},'adaptive_legacy_cohort_categories_fixed1529':{k:dict(v) for k,v in cohortcats.items()}};outcomes[mode]=cats
paired={}
for label in outcomes['fixed24']:
 a=outcomes['fixed24'][label];b=outcomes['adaptive8-24'][label];c=Counter((x,y) for x,y in zip(a,b) if x is not None);wins=sum(v for (x,y),v in c.items() if x!='correct' and y=='correct');losses=sum(v for (x,y),v in c.items() if x=='correct' and y!='correct');n=wins+losses
 paired[label]={'rows':sum(c.values()),'adaptive_only_correct':wins,'fixed_only_correct':losses,'net_adaptive':wins-losses,'both_correct':c['correct','correct'],'both_not_correct':1529-wins-losses-c['correct','correct'],'descriptive_exact_binomial_p':min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.,'category_transition_fixed_to_adaptive':{x+'->'+y:v for (x,y),v in c.items()}}
forecast=max(admit['same_t4_baselines_seconds'].values())*1.25;checks['controlled_admission_rejection_recomputed']=admit['status']=='rejected_insufficient_measured_budget' and admit['forecast_seconds']==forecast and admit['forecast_seconds']+20>admit['remaining_worker_deadline_seconds'] and worker['failure_type']=='ValueError' and worker['failure']=='Insufficient same-T4 measured budget to admit adaptive test1919 path' and not (V/'results/adaptive8-24/test1919').exists()
checks['all_pass']=all(checks.values());require(checks['all_pass'],str([k for k,v in checks.items() if not v]))
report={'schema':'colab_adaptive24_independent_recovery_quality_20261023_v1','utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'all_checks_pass':True,'artifact_checks':{'terminal_map_count':31,'receipt_count':32,'archive_sha256':sha(B/'adaptive24-results.zip'),'inner_terminal_sha256':sha(V/'session_result.json'),'standalone_terminal_sha256':sha(B/'session-result.json')},'runtime':{'hardware':worker['hardware'],'software':worker['software'],'model_sha256_reported_by_verified_worker':worker['model_sha256'],'parity':parity},'policies':policies,'replay':replays,'fixed_reference_sha256':sha(refs_path),'quality':quality,'paired':paired,'selected_code_changed_1792':sum(a['code']!=b['code'] for a,b in zip(preds['fixed24'],preds['adaptive8-24'])),'admission':{**admit,'required_minus_available_seconds':forecast+20-admit['remaining_worker_deadline_seconds']},'lifecycle':{'worker_seconds':worker['worker_seconds'],'parent_worker_seconds':outer['worker_seconds'],'worker_returncode':outer['worker_returncode'],'worker_failure_type':worker['failure_type'],'worker_failure':worker['failure'],'owner_elapsed_seconds':owner['owner_elapsed_seconds'],'cli_exec':[{k:v for k,v in event.items() if k in ('command','returncode','elapsed_seconds','timed_out')} for event in owner['events'] if event['command']=='exec_bounded_worker'],'cli_error':'Connection was lost (observed CLI traceback); cause unconfirmed','cleanup':outer['process_group_cleanup'],'stop':owner['stop'],'server_absence_evidence':[{k:event[k] for k in ('command','returncode','stdout_sanitized')} for event in owner['events'] if event['command']=='sessions_after_stop']},'timing_comparison':{'adaptive_over_fixed_validation_wall':policies['adaptive8-24/validation1792']['seconds']/policies['fixed24/validation1792']['seconds'],'candidate_reduction_fraction':1-policies['adaptive8-24/validation1792']['candidates']/policies['fixed24/validation1792']['candidates'],'memory_reduction_fraction':1-policies['adaptive8-24/validation1792']['peak_allocated_bytes']/policies['fixed24/validation1792']['peak_allocated_bytes']},'source_audit_path':str(Path(__file__).resolve().relative_to(R)),'source_audit_sha256':sha(Path(__file__)),'local_attempt1':{'preserved_source':'reports/evidence/colab_adaptive24_audit_attempt1_20261023.py','evidence':'reports/evidence/colab_adaptive24_audit_attempt1_20261023.json','cause':'Physical runtime source absent in inference worktree; corrected to read only source .py members of existing pinned ZIP. No tensor member or artifact changed.'},'scoring_source_sha256':{str(p.relative_to(R)):sha(p) for p in [selector,R/'src/nlp_hw02/arithmetic.py',R/'src/nlp_hw02/arithmetic_augassign_audit.py']},'limits':['Fixed original1529 noisy supplied-reference consensus, restricted AST numeric diagnostic, not official pass1. No label changes/filtering/generated code execution.','AugAssign changes generated-output support only; additive1e-6 diagnostic formula not confirmed official. Production selector and outputs unchanged.','Fixed24 and adaptive whole-call RNG streams/tensor shapes differ; no candidate-prefix equivalence or isolated causal early-stop effect claimed.','Within-Colab same-T4 fixed/adaptive measured comparison is available; ColabPython3.13/Torch2.11CUDA13 differs KaggleTorch2.10CUDA12.8. Exact fixed outputs/pool hashes reproduce, but this does not establish universal backend equivalence.','1792validation includes past model/decoder selection; one seed and repeated use limit descriptive p interpretation.','Adaptive full1919 path absent: no final test timing/output/<=180coldnotebook proof. No promotion or new run implied.','Clock excludes model/inputload, parity,warmup,laterhash/timingwrite; includes selection,compaction and candidate/progress serialization. Not standalone notebook proof.','No local weights/model loads, model generation, tests/fixtures, downloads/cloud, canonical orHEAD edits. Remote reported modelidentity only; source artifacts and outputs independently SHA checked.','CLI connection-loss cause unconfirmed. Worker admission rejection distinct from later CLI failure. Stop GET404 observed, not successful stopPOST; subsequent server inventory absent.'],'audit_seconds':time.monotonic()-started}
report['metadata_caveats']=[{'artifact': 'results/fixed24/validation1792/validation1792_timing.json', 'field': 'includes', 'issue': 'Copied text says all8-depth encoder and separate-depth KVcache. Actual selected model is standard4encoder+4decoder and observed cachedepth4. Stale timing-description metadata does not establish8layers per side; frozen bytes preserved.'}, {'artifact': 'results/worker_result.json', 'field': 'paths', 'issue': 'Partial current mode only enters paths after all3phases. Adaptive1792 and warmup4 completed files/policy_result are retained and SHAverified even though adaptive mode absent from worker_result.paths. Final mode incomplete flag is correct.'}]
report['recommendation']={'unchanged_adaptive_finish_run': 'NOT_JUSTIFIED_BY_CURRENT_EVIDENCE', 'reason': 'Fixed24 has1099 vs1095 supported-consensus diagnostic agreements, adaptive18.74% slower on sameT4; lower38.24% peakallocatedmemory is not currently binding. Retain proved canonicalfixed24; no new allocation or promotion authorized by audit.'}
out=R/'reports/evidence/colab_adaptive24_independent_quality_20261023.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'replay':replays,'quality':quality,'paired':paired,'evidence':str(out.relative_to(R)),'sha256':sha(out),'seconds':report['audit_seconds']},indent=2))
