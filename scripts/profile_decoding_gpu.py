"""One-T4 validation comparison and fixed beam timing; never tunes on test rows."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from tokenizers import Tokenizer
from nlp_hw02.beam import generate_beam
from nlp_hw02.inference import generate_questions,padded_tokens,syntax_valid
from nlp_hw02.model import load_model,Seq2Seq,ModelConfig
from nlp_hw02.tokenizer import decode,EOS


@torch.inference_mode()
def check_cuda_beam():
    torch.manual_seed(17)
    model=Seq2Seq(ModelConfig(vocab_size=32,d_model=16,heads=2,encoder_layers=1,
        decoder_layers=2,ff=32,dropout=0,max_source=8,max_target=8)).cuda().eval()
    src=torch.tensor([[4,EOS,0],[6,7,EOS]],device='cuda')
    class FullPrefix:
        training=False
        cfg=model.cfg
        def encode(self,source): return source[:,:,None].float(),source.ne(0)
        def new_cache(self,memory):
            prefix=torch.empty(memory.shape[0],1,self.cfg.max_target,1,device=memory.device)
            return [(prefix,torch.empty_like(prefix),memory[:,None],memory[:,None])]
        def decode_step(self,token,t,cache,valid):
            sk,sv,ck,cv=cache[0]; sk[:,:,t,0]=token[:,None].float()
            return model(ck[:,0,:,0].long(),sk[:,0,:t+1,0].long())[:,-1]
    for use_fp16 in (False,True):
        with torch.autocast('cuda',dtype=torch.float16,enabled=use_fp16):
            assert torch.equal(generate_beam(model,src,beam_size=1),model.generate(src))
            for width in (2,3):
                for alpha in (0.,.6):
                    actual=generate_beam(model,src,beam_size=width,length_penalty=alpha,max_new_tokens=4)
                    expected=generate_beam(FullPrefix(),src,beam_size=width,length_penalty=alpha,max_new_tokens=4)
                    assert torch.equal(actual,expected),'CUDA beam cache/full-prefix discrepancy'
    print('CUDA_BEAM_CACHE_CHECKS_OK',flush=True)


@torch.inference_mode()
def beam_questions(model,tok,questions,batch_size=256):
    model.eval(); device=next(model.parameters()).device
    if device.type=='cuda': torch.cuda.synchronize()
    start=time.perf_counter(); predictions=[]; lengths=[]; capped=0
    for offset in range(0,len(questions),batch_size):
        src=padded_tokens(tok,questions[offset:offset+batch_size],model.cfg.max_source).to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            tokens=generate_beam(model,src,beam_size=4,length_penalty=.6)
        for ids in tokens.cpu().tolist():
            ended=EOS in ids; lengths.append(ids.index(EOS) if ended else len(ids)); capped+=not ended
            predictions.append(decode(tok,ids))
    if device.type=='cuda': torch.cuda.synchronize()
    assert len(predictions)==len(questions)
    return predictions,{'seconds':time.perf_counter()-start,'rows':len(predictions),
        'batch_size':batch_size,'beam_width':4,'length_penalty':.6,'max_new_tokens':model.cfg.max_target,
        'mean_generated_tokens':sum(lengths)/len(lengths),'capped_rows':capped,
        'includes':'tokenization, transfers, encoder, cached beam decoder, detokenization; excludes loading, warm-up and writing'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--test-parquet',help='Optional fixed full-test timing; never scored or used to select decoder')
    parser.add_argument('--skip-synthetic-check',action='store_true',help='Omit inherited synthetic CUDA fixture; actual profile and real-artifact audits still run')
    args=parser.parse_args(); run=Path(args.run); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert 'T4' in torch.cuda.get_device_name(0),'One T4 required'
    torch.set_num_threads(2)
    if not args.skip_synthetic_check:check_cuda_beam()
    validation=json.loads((run/'validation.json').read_text())
    pilot=set(json.loads((run/'metrics.json').read_text())['validation_generation_indices'])
    selected=[r for i,r in enumerate(validation) if i not in pilot]; assert len(selected)==1792
    model,checkpoint=load_model(run/'model.pt'); tok=Tokenizer.from_file(str(run/'tokenizer.json'))
    report={'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'gpu_count':torch.cuda.device_count(),'dtype':'FP16 autocast; FP32 beam logprob/score',
        'selected_step':checkpoint['step'],'scope':'remaining1792, pilot256 excluded; previously used for CE/model selection',
        'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
        'beam_source_sha256':hashlib.sha256(Path(__import__('nlp_hw02.beam',fromlist=['x']).__file__).read_bytes()).hexdigest(),
        'profiler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'synthetic_fixture_executed':not args.skip_synthetic_check,'paths':{}}
    for label,generate in [('greedy',generate_questions),('beam4-alpha06',beam_questions)]:
        generate(model,tok,[validation[0]['question']],batch_size=256)
        torch.cuda.reset_peak_memory_stats()
        predictions,timing=generate(model,tok,[r['question'] for r in selected],batch_size=256)
        directory=output/label; directory.mkdir(exist_ok=True)
        shutil.copyfile(run/'validation.json',directory/'validation.json')
        with (directory/'validation_predictions.jsonl').open('w') as stream:
            for row,pred in zip(selected,predictions): stream.write(json.dumps({**row,'prediction':pred},ensure_ascii=False)+'\n')
        timing.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            syntax_valid=sum(syntax_valid(p) for p in predictions),
            exact_reference_text=sum(p==r['code'] for p,r in zip(predictions,selected)))
        report['paths'][label]=timing; print(json.dumps({'event':'validation_finished','path':label,**timing}),flush=True)
        (output/'decoding_profile.json').write_text(json.dumps(report,indent=2))
    if args.test_parquet:
        import pyarrow.parquet as pq
        manifest=json.loads((Path(__file__).resolve().parents[1]/'assignment/dataset_manifest.json').read_text())
        expected=next(r['sha256'] for r in manifest['files'] if r['path']=='data/test/test.parquet')
        actual=hashlib.sha256(Path(args.test_parquet).read_bytes()).hexdigest()
        assert actual==expected,'Test Parquet differs from pinned assignment file'
        questions=pq.read_table(args.test_parquet,columns=['question'])['question'].to_pylist(); assert len(questions)==1919
        predictions,timing=beam_questions(model,tok,questions,batch_size=256)
        timing.update(gpu=report['gpu'],gpu_count=1,selected_step=checkpoint['step'],
            test_parquet_sha256=actual,meets_180_seconds_on_one_T4=timing['seconds']<=180)
        (output/'beam_test_timing.json').write_text(json.dumps(timing,indent=2))
        with (output/'beam_test_predictions.jsonl').open('w') as stream:
            for index,pred in enumerate(predictions): stream.write(json.dumps({'row_index':index,'code':pred},ensure_ascii=False)+'\n')
        print(json.dumps({'event':'fixed_beam_test_timing',**timing}),flush=True)
    print('GPU_DECODING_PROFILE_FINISHED_OK',flush=True)


if __name__=='__main__': main()
