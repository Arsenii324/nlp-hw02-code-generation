"""Check sharing versus cloned depth, including per-application decoder caches."""
from dataclasses import asdict,replace
import argparse
import json
from pathlib import Path
import tempfile
import torch
from nlp_hw02.model import Seq2Seq,ModelConfig,load_model
from nlp_hw02.beam import generate_beam


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    cfg=ModelConfig(vocab_size=32,d_model=16,heads=2,encoder_layers=2,
        decoder_layers=2,ff=32,dropout=0,max_source=8,max_target=8,
        loop_repeats=2,residual_scale=.5)
    src=torch.tensor([[4,5,2,0],[6,2,0,0]]);tgt=torch.tensor([[1,8,9,2],[1,10,11,2]])
    torch.manual_seed(42);shared=Seq2Seq(cfg).eval()
    torch.manual_seed(42);independent=Seq2Seq(replace(cfg,share_loop_weights=False)).eval()
    assert len(shared.encoder)==2 and len(shared.encoder_steps())==4
    assert len(independent.encoder)==4 and len(independent.encoder_steps())==4
    assert shared.encoder_steps()[0] is shared.encoder_steps()[2]
    assert independent.encoder_steps()[0] is not independent.encoder_steps()[2]
    for name,value in shared.state_dict().items():assert torch.equal(value,independent.state_dict()[name])
    for stack in (independent.encoder,independent.decoder):
        for i in range(2):
            assert all(torch.equal(value,stack[i+2].state_dict()[name]) for name,value in stack[i].state_dict().items())
            assert next(stack[i].parameters()) is not next(stack[i+2].parameters())
    with torch.no_grad():assert torch.equal(shared(src,tgt),independent(src,tgt))
    # Cloning preserves the global RNG and initial stochastic dropout path too.
    shared.drop.p=independent.drop.p=.1
    for model in (shared,independent):
        for layer in model.encoder_steps()+list(model.decoder_steps()):
            layer.drop.p=.1;layer.self_attn.dropout=.1;layer.ff[2].p=.1
            if layer.decoder:layer.cross.dropout=.1
        model.train()
    torch.manual_seed(17);a=shared(src,tgt)
    torch.manual_seed(17);b=independent(src,tgt)
    assert torch.equal(a,b)
    # Reset dropout to test causal/cached equivalence; use cfg-preserving models.
    torch.manual_seed(42);shared=Seq2Seq(cfg).eval()
    torch.manual_seed(42);independent=Seq2Seq(replace(cfg,share_loop_weights=False)).eval()
    for model in (shared,independent):
        with torch.no_grad():
            logits=model(src,tgt);memory,valid=model.encode(src);cache=model.new_cache(memory)
            assert len(cache)==4
            assert len({entry[0].data_ptr() for entry in cache})==4
            assert len({entry[1].data_ptr() for entry in cache})==4
            for t in range(tgt.shape[1]):
                torch.testing.assert_close(model.decode_step(tgt[:,t],t,cache,valid),logits[:,t],atol=2e-6,rtol=1e-5)
            changed=tgt.clone();changed[:,-1]=17
            torch.testing.assert_close(model(src,changed)[:,:-1],logits[:,:-1],atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(model(torch.cat([src,torch.zeros(2,2,dtype=torch.long)],1),tgt),logits,atol=2e-6,rtol=1e-5)
        model(src,tgt).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';torch.save({'config':asdict(model.cfg),'model':model.state_dict(),'step':1},path)
            restored,record=load_model(path,'cpu')
            with torch.no_grad():assert torch.equal(restored(src,tgt),model(src,tgt))
        class FullPrefix:
            training=False
            cfg=model.cfg
            def encode(self,source):return source[:,:,None].float(),source.ne(0)
            def new_cache(self,memory):
                prefix=torch.empty(memory.shape[0],1,self.cfg.max_target,1,device=memory.device)
                return [(prefix,torch.empty_like(prefix),memory[:,None],memory[:,None])]
            def decode_step(self,token,t,cache,valid):
                sk,sv,ck,cv=cache[0];sk[:,:,t,0]=token[:,None].float()
                return model(ck[:,0,:,0].long(),sk[:,0,:t+1,0].long())[:,-1]
        for width in (2,4):
            assert torch.equal(generate_beam(model,src,beam_size=width,length_penalty=.6,max_new_tokens=4),
                generate_beam(FullPrefix(),src,beam_size=width,length_penalty=.6,max_new_tokens=4))
    real=ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,loop_repeats=2,residual_scale=.5)
    counts={}
    for name,share in [('shared',True),('independent',False)]:
        model=Seq2Seq(replace(real,share_loop_weights=share));counts[name]=sum(p.numel() for p in model.parameters());del model
    assert counts=={'shared':17524224,'independent':34088448}
    for repeat in [0,-1,True,1.5]:
        try:replace(cfg,loop_repeats=repeat)
        except ValueError:pass
        else:raise AssertionError('invalid repeat count accepted')
    print(json.dumps({'event':'looped_model_cpu_checks_passed','parameter_counts':counts,
        'exact_initial_eval_and_dropout_logits':True,'separate_application_caches':True,
        'cached_full_causal_padding_gradients_reload_beam':True,'CUDA_loop_check':False}),flush=True)
    if args.cuda:
        assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
        real=replace(real,dropout=0.,max_source=32,max_target=32)
        torch.manual_seed(42);shared=Seq2Seq(real).cuda().eval()
        torch.manual_seed(42);independent=Seq2Seq(replace(real,share_loop_weights=False)).cuda().eval()
        source=torch.randint(4,32,(2,16),device='cuda');target=torch.randint(4,32,(2,12),device='cuda')
        source[:,-2:]=0;source[:,-3]=2;target[:,0]=1
        errors={}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            assert torch.equal(shared(source,target),independent(source,target))
            for name,model in [('shared',shared),('independent',independent)]:
                logits=model(source,target);memory,valid=model.encode(source);cache=model.new_cache(memory)
                assert len(cache)==8 and len({entry[0].data_ptr() for entry in cache})==8
                error=0.
                for t in range(target.shape[1]):
                    cached=model.decode_step(target[:,t],t,cache,valid)
                    torch.testing.assert_close(cached,logits[:,t],atol=.015,rtol=.03)
                    error=max(error,float((cached-logits[:,t]).abs().max()))
                errors[name]=error
        for model in (shared,independent):
            with torch.autocast('cuda',dtype=torch.float16):loss=model(source,target).float().square().mean()
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print(json.dumps({'event':'looped_model_cuda_checks_passed','gpu':torch.cuda.get_device_name(0),
            'FP16_initial_shared_clone_equality':True,'FP16_cached_max_abs_errors':errors,'finite_gradients':True}),flush=True)


if __name__=='__main__':main()
