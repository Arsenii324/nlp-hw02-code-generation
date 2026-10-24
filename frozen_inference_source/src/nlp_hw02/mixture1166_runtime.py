"""Two-model N8+N3 sampler adapted from reviewed portable D generation code.

Exact model, count, selector, output identities must be root-bound before use.
This module is inert at import; it performs no model load/inference at import.
"""
import hashlib,json,os,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def ordered_sha(values):
    return hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def run(models,tokenizer,questions,out,manifest):
    import torch
    from nlp_hw02.own_selected_sampling_d import require,atomic_json
    from nlp_hw02.inference import padded_tokens
    from nlp_hw02.tokenizer import BOS,EOS,decode
    from nlp_hw02.sampling_candidate import select_plurality
    require(len(models)==2 and [m['samples_per_question'] for m in manifest['models']]==[8,3],'Frozen 8+3 two-model allocation required')
    require(len(questions)==1919 and all(isinstance(q,str) for q in questions),'Exact ordered question-only projection required')
    require(all(not m.training for m in models),'Eval mode required')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0),'Exactly one visible NVIDIA T4 required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def sample(model,spec,qs,dest,label):
        count=spec['samples_per_question'];poolpath=dest/(label+'_candidates.jsonl');predpath=dest/(label+'_predictions.jsonl')
        done=capped=0;torch.cuda.synchronize();start=time.perf_counter()
        torch.manual_seed(20261002);torch.cuda.manual_seed_all(20261002);torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),poolpath.open('x',encoding='utf-8') as pool,predpath.open('x',encoding='utf-8') as pred:
            for offset in range(0,len(qs),32):
                group=qs[offset:offset+32]
                src=padded_tokens(tokenizer,group,model.cfg.max_source).cuda().repeat_interleave(count,dim=0)
                with torch.autocast('cuda',dtype=torch.float16):
                    memory,valid=model.encode(src);cache=model.new_cache(memory)
                    token=torch.full((len(group)*count,),BOS,device='cuda',dtype=torch.long);finished=torch.zeros_like(token,dtype=torch.bool)
                    score=torch.zeros_like(token,dtype=torch.float32);length=torch.zeros_like(token);history=[]
                    for t in range(model.cfg.max_target):
                        logits=model.decode_step(token,t,cache,valid).float()/.8
                        values,indices=logits.topk(min(50,logits.shape[-1]),dim=-1);lp=values.log_softmax(-1)
                        draw=torch.multinomial(lp.exp(),1);token=indices.gather(1,draw).squeeze(1)
                        score+=torch.where(finished,0.,lp.gather(1,draw).squeeze(1));length+=(~finished).long()
                        token=torch.where(finished,torch.full_like(token,EOS),token);history.append(token);finished|=token.eq(EOS)
                        if finished.all():break
                    ids=torch.stack(history,1).cpu().tolist();scores=(score/((5+length.float())/6).pow(.6)).cpu().tolist()
                for j,q in enumerate(group):
                    codes=[decode(tokenizer,ids[j*count+k]) for k in range(count)];weights=scores[j*count:(j+1)*count]
                    flags=[EOS not in ids[j*count+k] for k in range(count)];lens=[row.index(EOS) if EOS in row else len(row) for row in ids[j*count:(j+1)*count]]
                    rec={'row_index':offset+j,'question_sha256':hashlib.sha256(q.encode()).hexdigest(),'samples':count,'candidates':codes,'scores':weights,'capped':flags,'generated_lengths':lens}
                    pool.write(json.dumps(rec,ensure_ascii=False)+'\n');pred.write(json.dumps({'code':codes[0]},ensure_ascii=False)+'\n');done+=1;capped+=sum(flags)
                atomic_json(dest/(label+'_progress.json'),{'complete':False,'rows':done,'requested':len(qs),'samples':count})
            for f in (pool,pred):f.flush();os.fsync(f.fileno())
        torch.cuda.synchronize()
        return {'rows':done,'samples':count,'seconds':time.perf_counter()-start,'capped':capped,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'candidate_sha256':sha(poolpath),'prediction_sha256':sha(predpath)}
    warmdir=out/'warmup';warmdir.mkdir();warm=[]
    for model,spec in zip(models,manifest['models']):
        d=warmdir/spec['name'];d.mkdir();warm.append(sample(model,spec,questions[:4],d,'warmup'))
    torch.cuda.synchronize();start=time.perf_counter();pools=[];model_timing=[]
    for model,spec in zip(models,manifest['models']):
        d=out/spec['name'];d.mkdir();model_timing.append(sample(model,spec,questions,d,'test'))
        rows=[json.loads(line) for line in (d/'test_candidates.jsonl').read_text().splitlines()]
        if len(rows)!=1919 or any(r['row_index']!=i or r['samples']!=spec['samples_per_question'] or r['question_sha256']!=hashlib.sha256(questions[i].encode()).hexdigest() for i,r in enumerate(rows)):
            raise ValueError('Model candidate rows/order differ')
        pools.append(rows)
    cpath=out/'mixture_candidates.jsonl';opath=out/'predictions.jsonl';total=0
    with cpath.open('x',encoding='utf-8') as cf,opath.open('x',encoding='utf-8') as of:
        for i,q in enumerate(questions):
            codes=pools[0][i]['candidates'][:8]+pools[1][i]['candidates'][:3]
            scores=pools[0][i]['scores'][:8]+pools[1][i]['scores'][:3]
            flags=pools[0][i]['capped'][:8]+pools[1][i]['capped'][:3]
            if len(codes)!=11:raise ValueError('Expected 11 candidates per question')
            index,info=select_plurality(codes,scores)
            rec={'row_index':i,'question_sha256':hashlib.sha256(q.encode()).hexdigest(),'samples':11,'candidates':codes,'scores':scores,'capped':flags,'selected_index':index,**info}
            cf.write(json.dumps(rec,ensure_ascii=False)+'\n');of.write(json.dumps({'code':codes[index]},ensure_ascii=False)+'\n');total+=len(codes)
        for f in (cf,of):f.flush();os.fsync(f.fileno())
    torch.cuda.synchronize();core=time.perf_counter()-start
    out_sha=sha(opath);expected=manifest['expected_output_sha256']
    qsha=ordered_sha(questions)
    if qsha!=manifest['ordered_test_questions_sha256']:raise ValueError('Ordered test question sequence differs')
    return {'schema':'antigravity_mixture1166_two_model_execution_v6_v1','rows':1919,'candidate_count':total,'candidates_per_question':11,'output_sha256':out_sha,'expected_output_sha256':expected,'output_reproduced':out_sha==expected,'ordered_test_questions_sha256':qsha,'ordered_codes_sha256':ordered_sha([json.loads(x)['code'] for x in opath.read_text().splitlines()]),'candidate_pool_sha256':sha(cpath),'warmup':warm,'model_test_timings':model_timing,'generation_selection_serialization_seconds':core,'backend':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'visible_gpu_count':torch.cuda.device_count()},'test_labels_read':False,'validation_generated':False,'promotion':False}
