"""Bounded optimizer-only CUDA gate; no training data/targets or retained model.

Run inside the next useful GPU session, not as a standalone allocation.
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import torch
from nlp_hw02.model import ModelConfig,Seq2Seq
from nlp_hw02.optim_batched import BatchedMuon,orthogonalize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--width',type=int,default=384)
    args=parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(20261001)
    assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    start=time.monotonic()
    cfg=ModelConfig(d_model=args.width,ff=4*args.width)
    model=Seq2Seq(cfg).cuda()
    matrices=[p for n,p in model.named_parameters() if p.ndim==2 and n.startswith(('encoder.','decoder.'))]
    gradients=[torch.randn_like(p)*.01 for p in matrices]
    parameters=[p.detach().clone() for p in matrices]
    record={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'capability':list(torch.cuda.get_device_capability()),'width':args.width,
        'matrix_shapes':[list(p.shape) for p in matrices],
        'note':'Optimizer-only fixed random gradient fixture; no data, targets, learned weights or speed-quality claim.',
        'paths':{}}
    fixtures={}
    for g in gradients: fixtures.setdefault(tuple(g.shape),[]).append(g)
    errors=[]
    for shape,values in fixtures.items():
        fixture=torch.stack(values)
        fp32=orthogonalize(fixture,dtype=torch.float32)
        fp16=orthogonalize(fixture,dtype=torch.float16).float()
        assert torch.isfinite(fp16).all()
        errors.append({'shape':list(shape),'count':len(values),
            'relative_l2':float((fp16-fp32).norm()/fp32.norm()),
            'max_absolute':float((fp16-fp32).abs().max())})
    record['fp16_precision_by_shape']=errors
    del fixture,fp32,fp16
    for name in ['stock_bf16','grouped_fp16','grouped_fp32']:
        try:
            for p,value,g in zip(matrices,parameters,gradients):
                p.data.copy_(value); p.grad=g.clone()
            if name=='stock_bf16':
                opt=torch.optim.Muon(matrices,lr=.003,weight_decay=.01,momentum=.95,nesterov=True,
                                     ns_steps=5,adjust_lr_fn='match_rms_adamw')
            else:
                opt=BatchedMuon(matrices,ns_dtype=torch.float16 if name=='grouped_fp16' else torch.float32)
            for _ in range(5): opt.step()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            samples=[]
            for _ in range(20):
                # Wall duration includes Python dispatch, the optimization target.
                tick=time.perf_counter(); opt.step(); torch.cuda.synchronize()
                samples.append(time.perf_counter()-tick)
            assert all(torch.isfinite(p).all() for p in matrices)
            delta=torch.cat([(p-value).flatten() for p,value in zip(matrices,parameters)])
            record['paths'][name]={'status':'OK','median_seconds':statistics.median(samples),
                'min_seconds':min(samples),'max_seconds':max(samples),'iterations':20,'warmup':5,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cumulative_delta_rms':float(delta.square().mean().sqrt())}
            del opt,delta; gc.collect(); torch.cuda.empty_cache()
        except (RuntimeError,AttributeError) as exc:
            record['paths'][name]={'status':'FAILED','type':type(exc).__name__,'reason':str(exc)}
    record['elapsed_seconds']=time.monotonic()-start
    Path(args.output).write_text(json.dumps(record,indent=2)); print(json.dumps(record),flush=True)
    assert record['paths']['grouped_fp16']['status']=='OK'
    assert all(r['relative_l2']<.05 for r in errors),'Unexpected fixture precision drift'


if __name__=='__main__': main()
