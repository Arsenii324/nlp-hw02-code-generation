"""Focused deterministic numerical/state checks for the grouped Muon candidate."""
import copy
import json
import math
import torch
from nlp_hw02.optim_batched import BatchedMuon, orthogonalize


def reference(x, steps=5):
    x=x.clone()
    tall=x.shape[0]>x.shape[1]
    if tall: x=x.T
    x=x/x.norm().clamp(min=1e-7)
    for _ in range(steps):
        a=x@x.T
        b=torch.addmm(a,a,a,beta=-4.775,alpha=2.0315)
        x=torch.addmm(x,b,x,beta=3.4445)
    return x.T if tall else x


def main():
    torch.set_num_threads(2); torch.manual_seed(20261001)
    errors=[]
    for rows,cols in [(8,8),(4,16),(16,4),(192,192),(768,192),(192,768)]:
        bank=torch.randn(3,rows,cols)
        bank[1].zero_(); bank[2]=bank[0]*1e-4
        bank[0,1:]=bank[0,0]  # rank deficient, independent of other slices
        expected=torch.stack([reference(x) for x in bank])
        actual=orthogonalize(bank,dtype=torch.float32)
        torch.testing.assert_close(actual,expected,atol=5e-6,rtol=5e-5)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            torch.testing.assert_close(orthogonalize(bank,dtype=torch.float32),actual,atol=0,rtol=0)
        perm=torch.tensor([2,0,1])
        torch.testing.assert_close(orthogonalize(bank[perm],dtype=torch.float32),actual[perm])
        half=orthogonalize(bank,dtype=torch.float16).float()
        assert torch.isfinite(half).all() and half[1].eq(0).all()
        errors.append({'shape':[rows,cols],'fp16_max_absolute':float((half-expected).abs().max()),
                       'fp16_relative_l2':float((half-expected).norm()/expected.norm())})
    for nesterov in [False,True]:
        for mode in ['original','match_rms_adamw']:
            params=[torch.nn.Parameter(torch.randn(*shape)) for shape in [(8,8),(16,4),(8,8),(4,16)]]
            expected=[p.detach().clone() for p in params]; buffers=[torch.zeros_like(p) for p in params]
            opt=BatchedMuon(params,lr=.003,weight_decay=.01,nesterov=nesterov,adjust_lr_fn=mode,ns_dtype=torch.float32)
            for step in range(4):
                grads=[torch.randn_like(p) for p in params]
                for index,(p,g) in enumerate(zip(params,grads)):
                    p.grad=None if step==1 and index==0 else g
                    if p.grad is None: continue
                    buffers[index].lerp_(g,.05)
                    u=g.lerp(buffers[index],.95) if nesterov else buffers[index]
                    scale=math.sqrt(max(1,p.shape[0]/p.shape[1])) if mode=='original' else .2*math.sqrt(max(p.shape))
                    expected[index].mul_(1-.003*.01).add_(reference(u),alpha=-.003*scale)
                opt.step()
                for p,e in zip(params,expected): torch.testing.assert_close(p,e,atol=1e-6,rtol=1e-5)
            state=copy.deepcopy(opt.state_dict())
            restored_params=[torch.nn.Parameter(p.detach().clone()) for p in params]
            restored=BatchedMuon(restored_params,ns_dtype=torch.float32)
            restored.load_state_dict(state)
            for p,q in zip(params,restored_params):
                p.grad=torch.randn_like(p); q.grad=p.grad.clone()
            opt.step(); restored.step()
            for p,q in zip(params,restored_params): torch.testing.assert_close(p,q,atol=0,rtol=0)
            assert all(v['momentum_buffer'].dtype==torch.float32 for v in restored.state.values())
    for bad in [torch.nn.Parameter(torch.zeros(3)),torch.nn.Parameter(torch.zeros(2,2,dtype=torch.float16))]:
        try: BatchedMuon([bad])
        except ValueError: pass
        else: raise AssertionError('unsupported parameter accepted')
    p=torch.nn.Parameter(torch.ones(2,2)); q=torch.nn.Parameter(torch.ones(2,2))
    opt=BatchedMuon([p]); opt.add_param_group({'params':[q],'momentum':2.})
    p.grad=torch.ones_like(p); q.grad=torch.ones_like(q)
    try: opt.step()
    except ValueError: pass
    else: raise AssertionError('invalid later group accepted')
    assert p.eq(1).all() and q.eq(1).all() and not opt.state
    try: BatchedMuon([{'params':[p],'lr':-1.}])
    except ValueError: pass
    else: raise AssertionError('invalid constructor group accepted')
    print(json.dumps({'status':'PASS','operator_precision_diagnostics':errors,
        'checks':'independent recurrence; independent slices; zero; rank deficient; transpose; both LR rules; Nesterov; absent grad; serialization continuation'},indent=2))


if __name__=='__main__': main()
