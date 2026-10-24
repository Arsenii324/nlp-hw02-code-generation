"""Focused checks for the authorized learnable-XSA/SwiGLU experiment."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import torch
from nlp_hw02.model import ModelConfig, Seq2Seq, exclusive_projection, load_model
from nlp_hw02.optim import make_optimizers


def main():
    torch.set_num_threads(2)
    base=ModelConfig(vocab_size=40,d_model=32,heads=4,encoder_layers=2,
        decoder_layers=2,ff=96,dropout=0,max_source=16,max_target=12)
    src=torch.tensor([[4,5,2,0],[6,2,0,0]])
    target=torch.tensor([[1,8,9,2],[1,10,11,2]])
    torch.manual_seed(42); control=Seq2Seq(base).eval()
    torch.manual_seed(42); gated=Seq2Seq(replace(base,learnable_xsa=True)).eval()
    old=control.state_dict(); new=gated.state_dict()
    assert all(torch.equal(value,new[name]) for name,value in old.items())
    assert set(new)-set(old)=={f'{stack}.{i}.self_attn.xsa_alpha'
                              for stack in ('encoder','decoder') for i in range(2)}
    assert all(torch.count_nonzero(p)==0 for n,p in gated.named_parameters() if n.endswith('.xsa_alpha'))
    with torch.no_grad():
        assert torch.equal(control(src,target),gated(src,target)), 'zero gate changed initial logits'
    # Zero initialization must still expose a gradient that can open each gate.
    loss=torch.nn.functional.cross_entropy(gated(src,target).flatten(0,1),target.flatten())
    loss.backward()
    gates=[p for n,p in gated.named_parameters() if n.endswith('.xsa_alpha')]
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in gates)
    for optimizer_kind in ('adamw','muon_grouped'):
        opts,info=make_optimizers(gated,optimizer_kind)
        params=[p for opt in opts for group in opt.param_groups for p in group['params']]
        assert len(params)==len(list(gated.parameters()))==len({id(p) for p in params})
        for gate in gates:
            owners=[(opt,group) for opt in opts for group in opt.param_groups
                    if any(p is gate for p in group['params'])]
            assert len(owners)==1 and isinstance(owners[0][0],torch.optim.AdamW)
            assert owners[0][1]['weight_decay']==0 and owners[0][1]['lr_multiplier']==1.
        for opt in opts: opt.step()
        assert all(torch.isfinite(p).all() for p in gated.parameters())
    # Generalized subtraction, tanh sign/range and exactly-zero own value safety.
    y=torch.randn(2,4,3,8); own=torch.randn_like(y); alpha=torch.randn(4,requires_grad=True)
    expected=y-torch.tanh(alpha)[None,:,None,None]*(y*torch.nn.functional.normalize(own,dim=-1,eps=1e-6)).sum(-1,keepdim=True)*torch.nn.functional.normalize(own,dim=-1,eps=1e-6)
    torch.testing.assert_close(exclusive_projection(y,own,alpha),expected)
    zero=torch.zeros_like(own,requires_grad=True)
    safe=exclusive_projection(y,zero,alpha); assert torch.equal(safe,y)
    safe.square().sum().backward()
    assert torch.isfinite(zero.grad).all() and torch.isfinite(alpha.grad).all()
    # Nonzero gates, all signs: causal cached/teacher-forced equivalence.
    with torch.no_grad():
        for gate in gates: gate.copy_(torch.tensor([-1.,0.,.5,2.]))
    swiglu=Seq2Seq(replace(base,activation='swiglu',ff=64)).eval()
    for model in (gated,swiglu):
        model.eval()
        with torch.no_grad():
            full=model(src,target); memory,valid=model.encode(src); cache=model.new_cache(memory)
            for t in range(target.shape[1]):
                torch.testing.assert_close(model.decode_step(target[:,t],t,cache,valid),full[:,t],atol=2e-6,rtol=1e-5)
            altered=target.clone(); altered[:,-1]=17
            torch.testing.assert_close(model(src,altered)[:,:-1],full[:,:-1],atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(model(torch.cat([src,torch.zeros(2,2,dtype=torch.long)],1),target),full,atol=2e-6,rtol=1e-5)
        model.zero_grad(); model(src,target).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save({'config':asdict(model.cfg),'model':model.state_dict(),'step':1},path)
            restored,record=load_model(path,'cpu')
            with torch.no_grad(): torch.testing.assert_close(restored(src,target),model(src,target),rtol=0,atol=0)
    # Actual matched-budget parameter counts.
    actual=ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4)
    counts={}
    for label,cfg in [('ordinary',actual),('learnable_xsa',replace(actual,learnable_xsa=True)),
                      ('swiglu',replace(actual,activation='swiglu',ff=1024))]:
        model=Seq2Seq(cfg); counts[label]=sum(p.numel() for p in model.parameters()); del model
    assert counts=={'ordinary':17524224,'learnable_xsa':17524256,'swiglu':17528320}
    # Frozen baseline constructor comparison when the recovered archive exists.
    frozen=Path('artifacts/kaggle/b011-b012-v1/training_source.bin')
    if frozen.exists():
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frozen_model.py'
            with zipfile.ZipFile(frozen) as archive: path.write_bytes(archive.read('src/nlp_hw02/model.py'))
            name='nlp_hw02._frozen_model'; spec=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
            torch.manual_seed(42); original=module.Seq2Seq(module.ModelConfig(**{k:v for k,v in asdict(base).items() if k in module.ModelConfig.__dataclass_fields__}))
            assert all(torch.equal(p,control.state_dict()[n]) for n,p in original.state_dict().items())
    print(json.dumps({'event':'model_followup_checks_passed','parameter_counts':counts,
        'checks':['zero-gate initial tensor/logit parity','finite nonzero gate gradients',
        'tanh projection and zero-value derivatives','exact no-decay Adam gate routing',
        'finite hybrid updates','nonzero-gate and SwiGLU cached/causal/padding equivalence',
        'strict checkpoint roundtrip','frozen baseline initialization when available']}),flush=True)


if __name__=='__main__': main()
