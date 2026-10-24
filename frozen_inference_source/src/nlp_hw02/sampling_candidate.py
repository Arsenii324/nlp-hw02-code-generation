"""Draft portable B034 sampling; frozen legacy selector, no labels or submission.

The candidate default is temperature.8/K24/qbatch32/top50/seed20261002. Each
sampling call resets CPU/CUDA seeds, including the first4-question warmup and
full ordered pass. This retains the original sample-scaling generation policy.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from .arithmetic import evaluate_code
from .inference import padded_tokens
from .tokenizer import BOS,EOS,decode

EXPECTED_TEST_SHA256='8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
PARENT_BUNDLE_SHA256='8c20ea91f891843b25d2ee60194e6ccad0e6b076b61fd2613a60bd8712c12a0e'
MODEL_SHA256='d09a8611bd6ef8ab2d0711d6999aa419efe50454d1bb12252069c59bc20c8a85'


@dataclass(frozen=True)
class SamplingPolicy:
    temperature:float=.8
    seed:int=20261002
    samples_per_question:int=24
    question_batch_size:int=32
    top_k:int=50
    length_penalty:float=.6

    def __post_init__(self):
        if self.temperature not in (.6,.8):raise ValueError('Reviewed temperature choices are.6 and.8')
        if type(self.seed) is not int or not 0<=self.seed<2**63:raise ValueError('Explicit nonnegative Torch seed required')
        if (self.samples_per_question,self.question_batch_size,self.top_k,self.length_penalty)!=(24,32,50,.6):
            raise ValueError('This candidate fixes K24/qbatch32/top50/lengthpenalty.6')

    @property
    def tag(self):return f'temp{int(round(self.temperature*10)):02d}-seed{self.seed}'


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def ordered_strings_sha256(values):
    return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def equal(a,b):
    return a==b if isinstance(a,int) and isinstance(b,int) else math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)

def select_plurality(codes,scores):
    if not codes or len(codes)!=len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError('Nonempty candidates and finite scores required')
    groups=[];supported=0
    for i,code in enumerate(codes):
        result=evaluate_code(code)
        if not result.supported:continue
        supported+=1
        group=next((g for g in groups if equal(g['value'],result.value)),None)
        if group is None:group={'value':result.value,'indices':[]};groups.append(group)
        group['indices'].append(i)
    if not groups:return 0,{'supported_candidates':0,'winning_votes':0,'numeric_groups':0}
    best=max(groups,key=lambda g:(len(g['indices']),max(scores[i] for i in g['indices'])))
    index=max(best['indices'],key=lambda i:scores[i])
    return index,{'supported_candidates':supported,'winning_votes':len(best['indices']),'numeric_groups':len(groups)}


@torch.inference_mode()
def sample_questions(model,tok,questions,policy):
    if model.training:raise RuntimeError("Call eval() before cached sampling")
    if not questions or not all(isinstance(q,str) for q in questions):raise ValueError("Nonempty ordered question strings required")
    args=policy;n_samples=policy.samples_per_question
    selected=[{"question":q} for q in questions]
    count=n_samples;question_batch=32
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.cuda.synchronize();start=time.perf_counter();first=[];plurality=[];diagnostics=[];all_candidates=[]
    all_lengths=[];all_caps=first_caps=selected_caps=0
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0,len(selected),question_batch):
        group=selected[offset:offset+question_batch];src=padded_tokens(tok,[r['question'] for r in group],model.cfg.max_source).cuda().repeat_interleave(count,dim=0)
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
        for j in range(len(group)):
            codes=[decode(tok,ids[j*count+k]) for k in range(count)];weights=scores[j*count:(j+1)*count]
            index,info=select_plurality(codes,weights);first.append(codes[0]);plurality.append(codes[index]);diagnostics.append(info)
            caps=[EOS not in ids[j*count+k] for k in range(count)]
            lengths=[row.index(EOS) if EOS in row else len(row) for row in ids[j*count:(j+1)*count]]
            all_lengths.extend(lengths);all_caps+=sum(caps);first_caps+=caps[0];selected_caps+=caps[index]
            all_candidates.append({'tag':args.tag,'temperature':args.temperature,'seed':args.seed,'samples_per_question':count,'question_batch_size':question_batch,'top_k':50,'length_penalty':.6,'max_new_tokens':model.cfg.max_target,'question_sha256':hashlib.sha256(group[j]['question'].encode()).hexdigest(),'candidates':codes,'scores':weights,'selected_index':index,'capped':caps,'generated_lengths':lengths,**info})
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    timing={'seconds':seconds,'rows':len(selected),'samples_per_question':count,'temperature':args.temperature,'top_k':50,'seed':args.seed,'tag':args.tag,
        'question_batch_size':question_batch,'length_penalty':.6,'max_new_tokens':model.cfg.max_target,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'without_supported_candidate':sum(d['supported_candidates']==0 for d in diagnostics),
        'all_candidate_capped_rows':all_caps,'first_candidate_capped_rows':first_caps,'selected_candidate_capped_rows':selected_caps,
        'all_candidate_mean_generated_tokens':sum(all_lengths)/len(all_lengths),
        'tie_break_score':f'length-normalized log probability under temperature {args.temperature:g}/top50-renormalized sampler, not raw-model likelihood',
        'includes':'tokenization/transfers/encoder/cached sampling/detokenization and reference-free numeric selection; excludes loading/warmup/output serialization'}
    return first,plurality,timing


def run_candidate(model,tok,test_parquet,output_dir,policy,manifest):
    """Generate ordered code-only JSONL and timing sidecar; never submit anything."""
    import pyarrow.parquet as pq
    import platform
    started=time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This proof candidate requires exactly one visible NVIDIA T4')
    if (model.cfg.max_source,model.cfg.max_target)!=(256,192):raise RuntimeError('Frozen checkpoint caps changed')
    if sum(p.numel() for p in model.parameters())!=17524256:raise RuntimeError('Frozen B034 topology changed')
    if model.training or next(model.parameters()).device.type!='cuda':raise RuntimeError('CUDA eval model required')
    test_parquet=Path(test_parquet);out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    if sha_file(test_parquet)!=EXPECTED_TEST_SHA256:raise RuntimeError('Pinned test Parquet hash mismatch')
    questions=pq.read_table(test_parquet,columns=['question'])['question'].to_pylist()
    if len(questions)!=1919 or not all(isinstance(q,str) for q in questions):raise RuntimeError('Expected1919 ordered questions')
    torch.set_num_threads(2)
    _,_,warmup=sample_questions(model,tok,questions[:4],policy)
    torch.cuda.synchronize();generation_started=time.perf_counter()
    _,predictions,timing=sample_questions(model,tok,questions,policy)
    if len(predictions)!=1919 or not all(isinstance(p,str) for p in predictions):raise RuntimeError('Incomplete ordered predictions')
    path=out/'predictions.jsonl';temporary=out/'predictions.jsonl.partial'
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        for prediction in predictions:stream.write(json.dumps({'code':prediction},ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    serialized_seconds=time.perf_counter()-generation_started
    # Readback is timed separately and never changes selection.
    readback=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(readback)!=1919 or any(set(row)!={'code'} or not isinstance(row['code'],str) for row in readback):
        raise RuntimeError('Code-only1919-line JSONL check failed')
    if [row['code'] for row in readback]!=predictions:raise RuntimeError('Ordered serialization changed predictions')
    output_sha=sha_file(path);ordered_code_sha=ordered_strings_sha256(predictions)
    expected=manifest['original_n24_reproduction']
    applicable=asdict(policy)==manifest['default_policy']
    reproduced=(output_sha==expected['code_only_jsonl_sha256'] and ordered_code_sha==expected['ordered_codes_sha256']) if applicable else None
    inclusive_seconds=time.perf_counter()-started
    sidecar={'status':'draft-unpromoted','policy':asdict(policy),'tag':policy.tag,'rows':1919,
        'input_sha256':EXPECTED_TEST_SHA256,'ordered_questions_sha256':ordered_strings_sha256(questions),
        'output_sha256':output_sha,'ordered_codes_sha256':ordered_code_sha,
        'parent_bundle_sha256':PARENT_BUNDLE_SHA256,'model_sha256':MODEL_SHA256,
        'package_member_sha256':manifest['member_sha256'],'selection_uses_reference':False,
        'hardware':{'gpu':torch.cuda.get_device_name(0),'visible_gpu_count':torch.cuda.device_count()},
        'software':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version()},
        'caps':{'source':256,'target':192,'target64_recommendation_not_enforced':True},
        'generation_and_selection':timing,'warmup_first4':warmup,
        'timing_seconds':{'generation_selection_serialization':serialized_seconds,'input_warmup_generation_serialization_readback_hashes':inclusive_seconds},
        'timing_boundaries':{'generation_and_selection':timing['includes'],
            'generation_selection_serialization':'Starts before full sample call (including seed reset); includes ordered code-only UTF8 JSONL serialization/flush/fsync/close/atomic rename. Excludes input loading, warmup, model loading and sidecar.',
            'input_warmup_generation_serialization_readback_hashes':'Starts before pinned input hashing/read; includes first4 warmup, full inference, serialization and ordered readback/hash; excludes model/package loading and sidecar.'},
        'within180':{'generation_and_selection':timing['seconds']<=180,'generation_selection_serialization':serialized_seconds<=180,'input_warmup_generation_serialization_readback_hashes':inclusive_seconds<=180},
        'original_n24_reproduction':{'applicable':applicable,'matches':reproduced,'comparison':'Only postgeneration hashes; no historical candidate text or outcomes used in inference. Different software/hardware may change sampling; exact reproduction must be measured.'},
        'open_contract':['Pinned Parquet correspondence to bot pass1_questions.jsonl order unconfirmed','Official timer boundary/entrypoint/library list/exact tolerance formula unconfirmed'],
        'official_result':None}
    (out/'candidate_timing.json').write_text(json.dumps(sidecar,indent=2)+'\n',encoding='utf-8')
    if applicable and not reproduced:raise RuntimeError('Original N24 output reproduction failed; retain sidecar/output for diagnosis, do not promote')
    return sidecar
