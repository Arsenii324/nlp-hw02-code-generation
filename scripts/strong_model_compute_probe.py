"""Bounded B034 validation and full-test inference experiments; references never select candidates."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from nlp_hw02.arithmetic import evaluate_code


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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from tokenizers import Tokenizer
    from nlp_hw02.model import load_model
    from nlp_hw02.beam import generate_beam
    from nlp_hw02.inference import padded_tokens,syntax_valid
    from nlp_hw02.tokenizer import BOS,EOS,decode
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--test-parquet');parser.add_argument('--consensus',required=True);parser.add_argument('--lane',choices=['speed','quality'],required=True)
    args=parser.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    validation=json.loads((run/'validation.json').read_text());pilot=set(json.loads((run/'metrics.json').read_text())['validation_generation_indices'])
    rows=[r for i,r in enumerate(validation) if i not in pilot];assert len(rows)==1792
    consensus=json.loads(Path(args.consensus).read_text())['groups_by_question_sha256']
    model,ckpt=load_model(run/'model.pt');tok=Tokenizer.from_file(str(run/'tokenizer.json'))
    report={'lane':args.lane,'checkpoint_sha256':sha(run/'model.pt'),'validation_sha256':sha(run/'validation.json'),
        'consensus_sha256':sha(args.consensus),'probe_sha256':sha(__file__),'selected_step':ckpt['step'],
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_count':1,
        'scope':'remaining1792; CE-used validation, not official pass@1','paths':{}}
    def save(): (out/'probe_result.json').write_text(json.dumps(report,indent=2)+'\n')
    def score_and_save(label,selected,predictions,timing):
        assert len(predictions)==len(selected)
        categories=Counter();reasons=Counter()
        for row,code in zip(selected,predictions):
            entry=consensus[hashlib.sha256(row['question'].encode()).hexdigest()]
            if entry['status']!='supported_consensus':continue
            value=evaluate_code(code)
            if value.supported:categories['correct' if equal(value.value,entry['consensus_value']) else 'wrong_numeric']+=1
            else:categories['unsupported']+=1;reasons[value.reason]+=1
        timing.update(strict_rows=sum(categories.values()),categories=dict(categories),unsupported_reasons=dict(reasons),
            syntax_valid=sum(map(syntax_valid,predictions)),exact_reference_text=sum(p==r['code'] for r,p in zip(selected,predictions)))
        path=out/f'{label}_predictions.jsonl'
        with path.open('w') as stream:
            for row,code in zip(selected,predictions):stream.write(json.dumps({**row,'prediction':code},ensure_ascii=False)+'\n')
        timing['prediction_sha256']=sha(path);report['paths'][label]=timing;save()
        print(json.dumps({'event':'probe_path_complete','label':label,**timing}),flush=True)

    @torch.inference_mode()
    def beam(selected,batch,width):
        torch.cuda.synchronize();start=time.perf_counter();predictions=[];lengths=[];capped=0
        torch.cuda.reset_peak_memory_stats()
        for offset in range(0,len(selected),batch):
            src=padded_tokens(tok,[r['question'] for r in selected[offset:offset+batch]],model.cfg.max_source).cuda()
            with torch.autocast('cuda',dtype=torch.float16):ids=generate_beam(model,src,beam_size=width,length_penalty=.6)
            for row in ids.cpu().tolist():
                ended=EOS in row;lengths.append(row.index(EOS) if ended else len(row));capped+=not ended
                predictions.append(decode(tok,row))
        torch.cuda.synchronize()
        return predictions,{'seconds':time.perf_counter()-start,'rows':len(selected),'batch_size':batch,'beam_width':width,
            'length_penalty':.6,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'capped_rows':capped,
            'mean_generated_tokens':sum(lengths)/len(lengths),'includes':'tokenization, transfers, encoder, cached decoder, detokenization; excludes loading/warmup/scoring/writing'}

    @torch.inference_mode()
    def samples(selected,tag="validation"):
        torch.manual_seed(20261002);torch.cuda.manual_seed_all(20261002)
        torch.cuda.synchronize();start=time.perf_counter();first=[];plurality=[];diagnostics=[];all_candidates=[]
        all_lengths=[];all_caps=first_caps=selected_caps=0
        torch.cuda.reset_peak_memory_stats()
        for offset in range(0,len(selected),32):
            group=selected[offset:offset+32];src=padded_tokens(tok,[r['question'] for r in group],model.cfg.max_source).cuda().repeat_interleave(8,dim=0)
            with torch.autocast('cuda',dtype=torch.float16):
                memory,valid=model.encode(src);cache=model.new_cache(memory)
                token=torch.full((len(group)*8,),BOS,device='cuda',dtype=torch.long);done=torch.zeros_like(token,dtype=torch.bool)
                score=torch.zeros_like(token,dtype=torch.float32);length=torch.zeros_like(token);history=[]
                for t in range(model.cfg.max_target):
                    logits=model.decode_step(token,t,cache,valid).float()/.8
                    values,indices=logits.topk(min(50,logits.shape[-1]),dim=-1);logprob=values.log_softmax(-1)
                    draw=torch.multinomial(logprob.exp(),1);token=indices.gather(1,draw).squeeze(1)
                    score+=torch.where(done,0.,logprob.gather(1,draw).squeeze(1));length+=(~done).long()
                    token=torch.where(done,torch.full_like(token,EOS),token);history.append(token);done|=token.eq(EOS)
                    if done.all():break
                ids=torch.stack(history,1).cpu().tolist();scores=(score/((5+length.float())/6).pow(.6)).cpu().tolist()
            for j in range(len(group)):
                codes=[decode(tok,ids[j*8+k]) for k in range(8)];weights=scores[j*8:(j+1)*8]
                index,info=select_plurality(codes,weights);first.append(codes[0]);plurality.append(codes[index]);diagnostics.append(info)
                caps=[EOS not in ids[j*8+k] for k in range(8)]
                lengths=[row.index(EOS) if EOS in row else len(row) for row in ids[j*8:(j+1)*8]]
                all_lengths.extend(lengths);all_caps+=sum(caps);first_caps+=caps[0];selected_caps+=caps[index]
                all_candidates.append({'question_sha256':hashlib.sha256(group[j]['question'].encode()).hexdigest(),'candidates':codes,'scores':weights,'selected_index':index,'capped':caps,'generated_lengths':lengths,**info})
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        with (out/f'{tag}_sample_candidates.jsonl').open('w') as stream:
            for row in all_candidates:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        timing={'seconds':seconds,'rows':len(selected),'samples_per_question':8,'temperature':.8,'top_k':50,'seed':20261002,
            'question_batch_size':32,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'without_supported_candidate':sum(d['supported_candidates']==0 for d in diagnostics),
            'all_candidate_capped_rows':all_caps,'first_candidate_capped_rows':first_caps,'selected_candidate_capped_rows':selected_caps,
            'all_candidate_mean_generated_tokens':sum(all_lengths)/len(all_lengths),
            'tie_break_score':'length-normalized log probability under temperature .8/top50-renormalized sampler, not raw-model likelihood',
            'includes':'eight-sample tokenization/transfers/encoder/decoding/detokenization and reference-free bounded numeric selection; excludes loading/warmup/reference scoring/writing'}
        return first,plurality,timing

    beam(rows[:8],8,4)
    if args.lane=='speed':
        subset=rows[:512];baseline=None
        for batch in (64,128,192,256):
            try:
                beam(subset[:batch],batch,4);durations=[]
                for repeat in range(2):
                    predictions,timing=beam(subset,batch,4);durations.append(timing['seconds'])
                    if baseline is None:baseline=predictions
                    timing['changed_text_rows_vs_first64']=sum(a!=b for a,b in zip(predictions,baseline))
                    score_and_save(f'beam4-batch{batch}-repeat{repeat+1}',subset,predictions,timing)
                report['paths'][f'beam4-batch{batch}-summary']={'median_seconds':statistics.median(durations),'repeats':2};save()
            except torch.cuda.OutOfMemoryError as exc:
                report['paths'][f'beam4-batch{batch}-failure']={'status':'OOM','message':str(exc)};torch.cuda.empty_cache();save()
    else:
        for width,batch in ((4,256),):
            beam(rows[:8],8,width);predictions,timing=beam(rows,batch,width);score_and_save(f'beam{width}',rows,predictions,timing)
        first,plurality,timing=samples(rows)
        score_and_save('sample-first',rows,first,{**timing,'note':'First of same eight samples; timing is total eight-sample cost, not isolated one-sample timing.'})
        score_and_save('sample-plurality8',rows,plurality,timing)
    if args.test_parquet and args.lane=='quality':
        import pyarrow.parquet as pq
        manifest=json.loads((Path(__file__).resolve().parents[1]/'assignment/dataset_manifest.json').read_text())
        expected=next(r['sha256'] for r in manifest['files'] if r['path']=='data/test/test.parquet')
        assert sha(args.test_parquet)==expected
        test=[{'question':q} for q in pq.read_table(args.test_parquet,columns=['question'])['question'].to_pylist()];assert len(test)==1919
        samples(test[:4],tag='warmup')
        first,preds,timing=samples(test,tag='test')
        timing.update(test_parquet_sha256=expected,gpu_count=1,gpu=torch.cuda.get_device_name(0),selection_uses_reference=False,meets_180_seconds_on_one_T4=timing['seconds']<=180)
        with (out/'test_plurality8_predictions.jsonl').open('w') as f:
            for i,pred in enumerate(preds):f.write(json.dumps({'row_index':i,'code':pred},ensure_ascii=False)+'\n')
        (out/'test_plurality8_timing.json').write_text(json.dumps(timing,indent=2))
        report['paths']['full_test_plurality8']=timing;save()
    report['complete']=True;save();print('TEST_COMPUTE_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
