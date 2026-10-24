"""Focused gates for adjacent full-block versus attention-only sharing."""
import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import torch
from nlp_hw02.model import Seq2Seq, ModelConfig, load_model
from nlp_hw02.optim import make_optimizers
from nlp_hw02.beam import generate_beam


def pair(cfg, device='cpu'):
    torch.manual_seed(42); full=Seq2Seq(cfg).to(device).eval(); rng=torch.get_rng_state()
    torch.manual_seed(42); partial=Seq2Seq(replace(cfg,loop_share_scope='attention')).to(device).eval()
    assert torch.equal(rng,torch.get_rng_state()), 'Clones consumed RNG'
    return full,partial


def check_structure(full, partial):
    for stack_name, n in [('encoder',full.cfg.encoder_layers),('decoder',full.cfg.decoder_layers)]:
        f=getattr(full,stack_name+'_steps')(); p=getattr(partial,stack_name+'_steps')()
        assert len(f)==len(p)==2*n
        for i in range(n):
            assert f[2*i] is f[2*i+1]
            assert p[2*i] is not p[2*i+1]
            assert p[2*i].self_attn is p[2*i+1].self_attn
            assert p[2*i].ff is not p[2*i+1].ff
            assert p[2*i].norm1 is not p[2*i+1].norm1
            assert p[2*i].norm_ff is not p[2*i+1].norm_ff
            if stack_name=='decoder':
                assert p[2*i].cross is p[2*i+1].cross
                assert p[2*i].norm_cross is not p[2*i+1].norm_cross
    optimizers,_=make_optimizers(partial,'muon_grouped')
    ids=[id(v) for o in optimizers for g in o.param_groups for v in g['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(v) for v in partial.parameters()}


def check_cached(model, src, tgt, atol, rtol, autocast=False):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=autocast):
        logits=model(src,tgt);memory,valid=model.encode(src);cache=model.new_cache(memory)
        assert len(cache)==len(model.decoder_steps())
        for k in [0,1]:assert len({entry[k].data_ptr() for entry in cache})==len(cache)
        maximum=0.
        for t in range(tgt.shape[1]):
            step=model.decode_step(tgt[:,t],t,cache,valid)
            torch.testing.assert_close(step,logits[:,t],atol=atol,rtol=rtol)
            maximum=max(maximum,float((step-logits[:,t]).abs().max()))
        changed=tgt.clone();changed[:,-1]=17
        torch.testing.assert_close(model(src,changed)[:,:-1],logits[:,:-1],atol=atol,rtol=rtol)
        padded=torch.cat([src,torch.zeros((len(src),2),device=src.device,dtype=torch.long)],1)
        torch.testing.assert_close(model(padded,tgt),logits,atol=atol,rtol=rtol)
    with torch.autocast('cuda',dtype=torch.float16,enabled=autocast):loss=model(src,tgt).float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return maximum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    cfg=ModelConfig(vocab_size=32,d_model=16,heads=2,encoder_layers=2,decoder_layers=2,
        ff=32,dropout=0.,max_source=10,max_target=8,loop_repeats=2,residual_scale=.5,
        loop_order='adjacent')
    full,partial=pair(cfg);check_structure(full,partial)
    src=torch.tensor([[4,5,2,0],[6,2,0,0]]);tgt=torch.tensor([[1,8,9,2],[1,10,11,2]])
    with torch.no_grad():assert torch.equal(full(src,tgt),partial(src,tgt))
    stochastic=replace(cfg,dropout=.1);a,b=pair(stochastic);a.train();b.train()
    torch.manual_seed(19);left=a(src,tgt)
    torch.manual_seed(19);right=b(src,tgt)
    assert torch.equal(left,right), 'Initial dropout path differs'
    for model in (full,partial):
        check_cached(model,src,tgt,2e-6,1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save({'config':asdict(model.cfg),'model':model.state_dict(),'step':1},path)
            restored,_=load_model(path,'cpu')
            with torch.no_grad():assert torch.equal(restored(src,tgt),model(src,tgt))
            if model is partial:check_structure(full,restored)
        class FullPrefix:
            training=False
            cfg=model.cfg
            def encode(self,source):return source[:,:,None].float(),source.ne(0)
            def new_cache(self,memory):
                prefix=torch.empty(memory.shape[0],1,self.cfg.max_target,1)
                return [(prefix,torch.empty_like(prefix),memory[:,None],memory[:,None])]
            def decode_step(self,token,t,cache,valid):
                sk,_,ck,_=cache[0];sk[:,:,t,0]=token[:,None].float()
                return model(ck[:,0,:,0].long(),sk[:,0,:t+1,0].long())[:,-1]
        for width in [2,4]:
            assert torch.equal(generate_beam(model,src,beam_size=width,length_penalty=.6,max_new_tokens=4),
                generate_beam(FullPrefix(),src,beam_size=width,length_penalty=.6,max_new_tokens=4))
    bad={k:v.clone() for k,v in partial.state_dict().items()}
    bad['encoder.2.self_attn.q.weight']+=1
    try:partial.load_state_dict(bad)
    except ValueError:pass
    else:raise AssertionError('Inconsistent shared attention checkpoint accepted')
    try:partial.load_state_dict(partial.state_dict(),assign=True)
    except ValueError:pass
    else:raise AssertionError('Alias-breaking assign=True accepted')
    for change in [dict(loop_order='unknown'),dict(loop_share_scope='unknown'),
                   dict(loop_share_scope='attention',share_loop_weights=False),
                   dict(loop_share_scope='attention',loop_repeats=1)]:
        try:replace(cfg,**change)
        except ValueError:pass
        else:raise AssertionError('Invalid config accepted')
    independent=Seq2Seq(replace(cfg,share_loop_weights=False)).eval()
    assert independent.encoder_steps()[0] is independent.encoder[0]
    assert independent.encoder_steps()[1] is independent.encoder[2]
    before=Path(__file__).resolve().parents[1]/'before-model.py'
    if before.exists():
        spec=importlib.util.spec_from_file_location('nlp_hw02._before_adjacent',before)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        plain={k:v for k,v in asdict(replace(cfg,loop_repeats=1,residual_scale=1.)).items()
               if k not in ('loop_order','loop_share_scope')}
        torch.manual_seed(31);old=module.Seq2Seq(module.ModelConfig(**plain)).eval()
        torch.manual_seed(31);new=Seq2Seq(ModelConfig(**plain)).eval()
        assert old.state_dict().keys()==new.state_dict().keys()
        assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
        with torch.no_grad():assert torch.equal(old(src,tgt),new(src,tgt))
    real=ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,
        loop_repeats=2,residual_scale=.5,loop_order='adjacent')
    counts={scope:sum(p.numel() for p in Seq2Seq(replace(real,loop_share_scope=scope)).parameters())
            for scope in ['block','attention']}
    assert counts=={'block':17524224,'attention':26992128}
    print(json.dumps({'event':'adjacent_sharing_cpu_checks_passed','parameter_counts':counts,
        'exact_initial_eval_dropout_rng':True,'optimizer_deduplicated':True,
        'cache_padding_causal_beam_reload_alias_validation':True,'default_unchanged_checked':before.exists()}),flush=True)
    if args.cuda:
        assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
        full,partial=pair(replace(real,dropout=0.,max_source=32,max_target=32),'cuda')
        source=torch.randint(4,32,(2,16),device='cuda');target=torch.randint(4,32,(2,12),device='cuda')
        source[:,-2:]=0;source[:,-3]=2;target[:,0]=1
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            assert torch.equal(full(source,target),partial(source,target))
        errors={name:check_cached(model,source,target,.015,.03,True)
                for name,model in [('block',full),('attention',partial)]}
        print(json.dumps({'event':'adjacent_sharing_cuda_checks_passed','gpu':torch.cuda.get_device_name(0),
            'initial_FP16_exact':True,'cached_max_abs_errors':errors,'finite_gradients':True}),flush=True)


if __name__=='__main__':main()
