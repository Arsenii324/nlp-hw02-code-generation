"""Portable source for one actual ordinary own-07e checkpoint and one fixed count."""
import hashlib,json,math,os,time
from pathlib import Path
MODEL_SOURCE_SHA='07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26'
TEST_PARQUET_SHA='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
GENERATION={'temperature':.8,'top_k':50,'question_batch_size':32,'length_penalty':.6,'seed':20261002,'max_new_tokens':192}
BACKEND_POLICY={'schema':'own_portable_observed_backend_a_v1','training_software_is_provenance_only':True,'live_version_equality_gate':False,'qualification_requires_actual_backend_parity_and_exact_reference_reproduction_and_four_clocks_le180':True}
BASE={'vocab_size':2048,'encoder_layers':4,'decoder_layers':4,'dropout':.1,'max_source':256,'max_target':192,'exclusive_self_attention':False,'learnable_xsa':True,'loop_repeats':1,'share_loop_weights':True,'residual_scale':1.,'loop_order':'stack','loop_share_scope':'block','gated_repeats':False,'repeat_residual_scale':.5}
ARCHITECTURES={
 'own384_gelu':{**BASE,'d_model':384,'heads':4,'ff':1536,'activation':'gelu'},
 'own512_gelu':{**BASE,'d_model':512,'heads':8,'ff':2048,'activation':'gelu'},
 'own768_gelu':{**BASE,'d_model':768,'heads':12,'ff':3072,'activation':'gelu'},
}
PARAMETERS={'own384_gelu':17524256,'own512_gelu':30705728,'own768_gelu':68078688}
XSA={'own384_gelu':32,'own512_gelu':64,'own768_gelu':96}
COUNTS=(8,12,16,20,24)

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def valid_sha(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def atomic_json(path,obj):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def verify_topology(model,architecture):
    import torch
    cfg=model.cfg;require(model.training is False and vars(cfg)==ARCHITECTURES[architecture],'Exact eval config required')
    require(sum(p.numel() for p in model.parameters())==PARAMETERS[architecture],'Unique parameter count differs')
    require(sum(p.numel() for n,p in model.named_parameters() if n.endswith('.xsa_alpha'))==XSA[architecture],'Learned XSA count/alias differs')
    for name,stack in [('encoder',model.encoder),('decoder',model.decoder)]:
        require(len(stack)==4 and len({id(x.ff) for x in stack})==len({id(x.norm1) for x in stack})==len({id(x.norm_ff) for x in stack})==4,'Ordinary 4+4 block topology required:'+name)
        if name=='decoder':require(len({id(x.norm_cross) for x in stack})==4,'Distinct cross norms required')
    require(not cfg.gated_repeats and cfg.loop_repeats==1,'Only exact ordinary 4+4 architecture is supported')
    return {'architecture':architecture,'parameters':PARAMETERS[architecture],'unique_xsa_scalars':XSA[architecture],'encoder_blocks':4,'decoder_blocks':4,'gated_repeats':False}

def validate_deployment(d):
    require(d.get('schema')=='own_portable_candidate_deployment_b_v1' and d.get('verified_actual_values') is True,'Root-bound selected candidate required before Torch')
    a=d.get('architecture');require(a in ARCHITECTURES and d.get('model_config')==ARCHITECTURES[a] and d.get('parameters')==PARAMETERS[a] and d.get('unique_xsa_scalars')==XSA[a],'Exact architecture whitelist/count required')
    require(type(d.get('samples_per_question')) is int and d['samples_per_question'] in COUNTS and d.get('mode')==f"fixed{d['samples_per_question']}",'One fixed8/12/16/20/24 count required')
    require(d.get('generation')==GENERATION and d.get('test_source_parquet_sha256')==TEST_PARQUET_SHA and valid_sha(d.get('test_questions_sha256')) and d.get('model_source_sha256')==MODEL_SOURCE_SHA,'Frozen model/input/sampler identity required')
    require(valid_sha(d.get('model_sha256')) and type(d.get('checkpoint_step')) is int and d['checkpoint_step']>0 and math.isfinite(d.get('checkpoint_validation_loss')) and d.get('checkpoint_seed')==42,'Actual selected checkpoint fields required')
    require(d.get('training_complete') in (True,False) and isinstance(d.get('training_family'),str) and d['training_family'],'Truthful selected lineage required')
    require(d.get('backend_policy')==BACKEND_POLICY and isinstance(d.get('training_software'),dict),'Observed backend policy and training software provenance required')
    refs=d.get('root_receipts',{});required={'selection','cpu_integrity','lineage','quality','actual_reference_outputs','actual_parity'}
    require(set(refs)==required and all(valid_sha(x) for x in refs.values()),'Root-accepted CPU/lineage/quality/reference/parity receipt hashes required')
    repro=d.get('reproduction',{});fields={'validation_ordered_questions_sha256','validation_ordered_codes_sha256','test_ordered_questions_sha256','test_code_only_jsonl_sha256','test_ordered_codes_sha256'}
    require(set(repro)==fields and all(valid_sha(x) for x in repro.values()),'Actual ordered1792/1919 reproduction hashes required')
    engine=d.get('engine');require(engine in ('eager','cuda_graph'),'A fixed reviewed inference engine is required')
    if engine=='cuda_graph':
        graph=d.get('graph_engine',{})
        require(graph.get('root_accepted') is True and graph.get('independent_review_pass') is True and valid_sha(graph.get('source_sha256')),'Graph path requires root-accepted reviewed helper source')
        require(graph.get('architecture')==a and isinstance(graph.get('supported_counts'),list)
                and d['samples_per_question'] in graph['supported_counts'] and d['samples_per_question'] in (16,20,24),
                'This reviewed graph helper does not qualify the selected architecture/count')
        require(graph.get('actual_parity_sha256')==refs['actual_parity'],'Graph actual-parity receipt must match this selected deployment')
    return d['mode']

def generate_candidates(model,tok,questions,count):
    # The complete GPU With block below is copied from the frozen sampler.
    import torch
    from types import SimpleNamespace
    from nlp_hw02.inference import padded_tokens
    from nlp_hw02.tokenizer import BOS,EOS,decode
    require(count in (8,12,16,20,24) and 1<=len(questions)<=32,'Only prescribed8/12/16/20/24 and qbatch<=32')
    args=SimpleNamespace(temperature=.8);group=[{'question':q} for q in questions]
    src=padded_tokens(tok,[r['question'] for r in group],model.cfg.max_source).cuda().repeat_interleave(count,dim=0)
    with torch.autocast('cuda',dtype=torch.float16):
        memory,valid=model.encode(src);cache=model.new_cache(memory)
        token=torch.full((len(group)*count,),BOS,device='cuda',dtype=torch.long);done=torch.zeros_like(token,dtype=torch.bool)
        score=torch.zeros_like(token,dtype=torch.float32);length=torch.zeros_like(token);history=[]
        for t in range(model.cfg.max_target):
            logits=model.decode_step(token,t,cache,valid).float()/args.temperature
            values,indices=logits.topk(min(50,logits.shape[-1]),dim=-1);logprob=values.log_softmax(-1)
            draw=torch.multinomial(logprob.exp(),1);token=indices.gather(1,draw).squeeze(1)
            score+=torch.where(done,0.,logprob.gather(1,draw).squeeze(1));length+=(~done).long()
            token=torch.where(done,torch.full_like(token,EOS),token);history.append(token);done|=token.eq(EOS)
            if done.all():break
        ids=torch.stack(history,1).cpu().tolist();scores=(score/((5+length.float())/6).pow(.6)).cpu().tolist()
    records=[]
    for j in range(len(group)):
        codes=[decode(tok,ids[j*count+k]) for k in range(count)];weights=scores[j*count:(j+1)*count]
        caps=[EOS not in ids[j*count+k] for k in range(count)]
        lengths=[row.index(EOS) if EOS in row else len(row) for row in ids[j*count:(j+1)*count]]
        records.append({'candidates':codes,'scores':weights,'capped':caps,'generated_lengths':lengths})
    return records

def sample_fixed(model,tok,questions,count,out,label,deadline,stopping,*,engine="eager",graph_module=None,graphs=None):
    import torch
    from nlp_hw02.sampling_candidate import select_plurality
    require(not model.training and questions and all(isinstance(q,str) for q in questions),'Eval and ordered gold-free questions required')
    pool_path=out/(label+'_candidates.jsonl');prediction_path=out/(label+'_predictions.jsonl');completed=all_caps=selected_caps=unsupported=0
    torch.cuda.synchronize();began=time.perf_counter();torch.manual_seed(20261002);torch.cuda.manual_seed_all(20261002);torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),pool_path.open('x') as pool,prediction_path.open('x') as predictions:
        try:
            for offset in range(0,len(questions),32):
                if stopping() or time.monotonic()>=deadline:raise InterruptedError('Whole-worker stop/deadline between sampling chunks')
                group=questions[offset:offset+32]
                if engine=='eager':records=generate_candidates(model,tok,group,count)
                else:
                    require(engine=='cuda_graph' and graph_module is not None and graphs is not None,'Reviewed graph engine required')
                    records=graph_module.generate_candidates_graph(model,tok,group,count,graphs)
                for j,(question,record) in enumerate(zip(group,records)):
                    index,info=select_plurality(record['candidates'],record['scores'])
                    row={'row_index':offset+j,'question_sha256':hashlib.sha256(question.encode()).hexdigest(),'samples':count,**record,'selected_index':index,**info}
                    pool.write(json.dumps(row,ensure_ascii=False)+'\n');predictions.write(json.dumps({'code':record['candidates'][index]},ensure_ascii=False)+'\n')
                    all_caps+=sum(record['capped']);selected_caps+=record['capped'][index];unsupported+=info['supported_candidates']==0;completed+=1
                atomic_json(out/(label+'_progress.json'),{'complete':False,'completed_rows':completed,'requested_rows':len(questions),'samples_per_question':count})
        finally:
            for stream in (pool,predictions):stream.flush();os.fsync(stream.fileno())
    torch.cuda.synchronize();seconds=time.perf_counter()-began
    require(completed==len(questions),'Incomplete ordered sampling')
    timing={'rows':completed,'samples_per_question':count,'total_candidates':completed*count,'temperature':.8,'top_k':50,'question_batch_size':32,'seed':20261002,'length_penalty':.6,'max_new_tokens':192,
        'all_candidate_capped_rows':all_caps,'selected_candidate_capped_rows':selected_caps,'without_supported_candidate':unsupported,
        'complete_path_seconds':seconds,'complete_path_within180':seconds<=180,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'candidate_sha256':sha(pool_path),'prediction_sha256':sha(prediction_path),
        'ordered_questions_sha256':hashlib.sha256(json.dumps(questions,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        'includes':'Seedreset/tokenization/transfers/all8-depth encoder and separate-depth KVcache/decoder/sampling/detokenization/unchangednumericselection and candidate+prediction+progress serialization/flush/fsync/close/CUDAsync; excludes model/inputload/parity/warmup and subsequent hashes/timingwrite.'}
    atomic_json(out/(label+'_timing.json'),timing);return timing

def ordered_sha(values):return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def run_portable_candidate(model,tok,questions,output,manifest,*,test_questions_path,deadline=float('inf'),stopping=lambda:False,graph_module=None):
    deployment=manifest['deployment'];mode=validate_deployment(deployment);started=time.perf_counter()
    require(len(questions)==1919 and all(isinstance(q,str) for q in questions),'Exact ordered question-only test projection required')
    require(ordered_sha(questions)==deployment['reproduction']['test_ordered_questions_sha256'],'Actual test question order differs from root-bound reference')
    require(sha(test_questions_path)==deployment['test_questions_sha256'],'Question-only test projection differs from root-bound bytes')
    import platform
    import torch
    from . import model as model_module
    require(sha(model_module.__file__)==MODEL_SOURCE_SHA and vars(model.cfg)==deployment['model_config'],'Exact unchanged07e model source/config required')
    require(not model.training and torch.cuda.is_available() and torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0),'One visible T4 eval inference required')
    topology=verify_topology(model,deployment['architecture'])
    engine=deployment['engine'];count=deployment['samples_per_question'];out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    def graph_session():
        if engine!='cuda_graph':return None
        require(graph_module is not None,'Root-accepted graph helper is absent')
        return graph_module.DecodeGraphs(model,deadline,stopping)
    graphs=graph_session()
    try:warmup=sample_fixed(model,tok,questions[:4],count,out,f'warmup_n{count}',deadline,stopping,engine=engine,graph_module=graph_module,graphs=graphs)
    finally:
        if graphs is not None:graphs.close();torch.cuda.empty_cache()
    torch.cuda.synchronize();gen=time.perf_counter();graphs=graph_session()
    try:timing=sample_fixed(model,tok,questions,count,out,f'test_n{count}',deadline,stopping,engine=engine,graph_module=graph_module,graphs=graphs)
    finally:
        if graphs is not None:graphs.close();torch.cuda.empty_cache()
    path=out/'predictions.jsonl';os.replace(out/f'test_n{count}_predictions.jsonl',path);rename_seconds=time.perf_counter()-gen
    codes=[]
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);require(set(row)=={'code'} and isinstance(row['code'],str),'Output must be code-only JSONL');codes.append(row['code'])
    require(len(codes)==1919,'Exact ordered1919 output required')
    question_sha=ordered_sha(questions)
    pool_path=out/f'test_n{count}_candidates.jsonl'
    require(pool_path.is_file(),'Candidate pool was not retained')
    with pool_path.open(encoding='utf-8') as stream:
        pool_rows=[json.loads(line) for line in stream]
    require(len(pool_rows)==1919 and all(row.get('row_index')==i and row.get('samples')==count
            and row.get('question_sha256')==hashlib.sha256(questions[i].encode()).hexdigest()
            and isinstance(row.get('candidates'),list) and len(row['candidates'])==count
            and type(row.get('selected_index')) is int and 0<=row['selected_index']<count
            and row['candidates'][row['selected_index']]==codes[i]
            for i,row in enumerate(pool_rows)),
            'Selected output does not match its ordered candidate pool')
    code_hash=ordered_sha(codes);output_hash=sha(path);expected=deployment['reproduction']
    reproduced=output_hash==expected['test_code_only_jsonl_sha256'] and code_hash==expected['test_ordered_codes_sha256']
    sidecar={'schema':'own_portable_candidate_execution_a_v1','execution_complete':True,'status':'candidate-executed-awaiting-fresh-one-T4-proof','deployment':deployment,'rows':1919,'output_sha256':output_hash,'ordered_codes_sha256':code_hash,
      'reproduction':{'matches':reproduced,'comparison':'Actual post-generation exact reference hashes; references are not used by selection.'},'warmup_first4':warmup,'generation_selection_serialization':timing,'topology':topology,
      'timing_seconds':{'generation_selection_graph_setup_serialization_and_rename':rename_seconds,'input_generation_output_readback_hashes':time.perf_counter()-started},
      'backend':{'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'visible_gpu_count':torch.cuda.device_count(),'training_software_provenance':deployment['training_software'],'version_equality_required':False},
      'selection_uses_reference':False,'gold_used':False,'validation_generated':False,'training':False,'fresh_one_T4_proof_required':True,'official_result':None}
    atomic_json(out/'candidate_timing.json',sidecar);return sidecar
