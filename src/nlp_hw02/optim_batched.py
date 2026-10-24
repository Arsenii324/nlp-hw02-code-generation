"""Experimental grouped Muon: FP32 state/norms, selectable NS precision.

Momentum, polynomial, transpose, decay and LR rules follow PyTorch v2.10.
FP16 intermediates and FP32 normalization deliberately differ from stock BF16.
No CUDA graphs, compilation, new coefficients or global PyTorch patches.
"""
import math
import torch


@torch.no_grad()
def orthogonalize(bank, *, dtype=torch.float16, steps=5, eps=1e-7):
    """Apply independent quintic NS maps to a [count, rows, cols] FP32 bank."""
    if bank.ndim!=3 or bank.dtype!=torch.float32:
        raise ValueError('Expected a three-dimensional FP32 matrix bank')
    if dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError('Unsupported NS precision')
    if not isinstance(steps,int) or not 0<=steps<100 or not math.isfinite(eps) or eps<=0:
        raise ValueError('Invalid NS steps/epsilon')
    tall=bank.shape[-2]>bank.shape[-1]
    # Normalize before the low precision cast to avoid FP16 input overflow.
    # Unlike stock2.10, norm accumulation and division are explicitly FP32.
    with torch.autocast(device_type=bank.device.type,enabled=False):
        x=bank/bank.norm(dim=(-2,-1),keepdim=True).clamp(min=eps)
        if tall: x=x.transpose(-2,-1)
        x=x.to(dtype)
        for _ in range(steps):
            a=torch.bmm(x,x.transpose(-2,-1))
            b=torch.baddbmm(a,a,a,beta=-4.775,alpha=2.0315)
            x=torch.baddbmm(x,b,x,beta=3.4445)
        return x.transpose(-2,-1) if tall else x


class BatchedMuon(torch.optim.Optimizer):
    """Muon on dense real FP32 2-D parameters, grouping exact shapes each step.

    Ordinary per-parameter momentum state makes state_dict portable across the
    same ordered parameter groups. Missing gradients cause neither decay nor
    momentum updates. Embeddings/norms/biases must be routed to another optimizer.
    """
    orthogonalizer_name='newton_schulz'

    def orthogonalize_bank(self,bank,group):
        return orthogonalize(bank,dtype=group['ns_dtype'],steps=group['ns_steps'],eps=group['eps'])

    def __init__(self,params,lr=.003,weight_decay=.01,momentum=.95,nesterov=True,
                 ns_steps=5,eps=1e-7,adjust_lr_fn='match_rms_adamw',ns_dtype=torch.float16):
        if not all(math.isfinite(v) and v>=0 for v in (lr,weight_decay,momentum)) or momentum>1:
            raise ValueError('Invalid learning rate, decay or momentum')
        if adjust_lr_fn not in ('original','match_rms_adamw'):
            raise ValueError('Unsupported shape adjustment')
        if ns_dtype not in (torch.float16,torch.bfloat16,torch.float32):
            raise ValueError('Unsupported NS precision')
        if not isinstance(ns_steps,int) or not 0<=ns_steps<100 or not math.isfinite(eps) or eps<=0:
            raise ValueError('Invalid NS steps/epsilon')
        super().__init__(params,dict(lr=lr,weight_decay=weight_decay,momentum=momentum,
            nesterov=nesterov,ns_steps=ns_steps,eps=eps,adjust_lr_fn=adjust_lr_fn,ns_dtype=ns_dtype))
        for group in self.param_groups: self._validate_group(group)

    @staticmethod
    def _validate_group(group):
        if not all(isinstance(group[k],(int,float)) and math.isfinite(group[k]) and group[k]>=0
                   for k in ('lr','weight_decay','momentum')) or group['momentum']>1:
            raise ValueError('Invalid group learning rate, decay or momentum')
        if group['adjust_lr_fn'] not in ('original','match_rms_adamw'):
            raise ValueError('Unsupported group shape adjustment')
        if group['ns_dtype'] not in (torch.float16,torch.bfloat16,torch.float32):
            raise ValueError('Unsupported group NS precision')
        if not isinstance(group['ns_steps'],int) or not 0<=group['ns_steps']<100 or not math.isfinite(group['eps']) or group['eps']<=0:
            raise ValueError('Invalid group NS steps/epsilon')
        for p in group['params']:
            if p.ndim!=2 or p.dtype!=torch.float32 or p.is_complex():
                raise ValueError('Grouped Muon supports only real FP32 2-D parameters')

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        # Reject all unsupported gradients before changing any state/parameter.
        for group in self.param_groups:
            self._validate_group(group)
            for p in group['params']:
                if p.grad is not None and (p.grad.is_sparse or p.grad.dtype!=torch.float32):
                    raise ValueError('Expected dense FP32 gradients')
        for group in self.param_groups:
            banks={}
            for p in group['params']:
                if p.grad is None: continue
                state=self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer']=torch.zeros_like(p)
                buf=state['momentum_buffer']
                buf.lerp_(p.grad,1-group['momentum'])
                u=p.grad.lerp(buf,group['momentum']) if group['nesterov'] else buf
                key=(tuple(p.shape),p.device,p.dtype)
                banks.setdefault(key,[]).append((p,u))
            for (shape,device,param_dtype),items in banks.items():
                updates=self.orthogonalize_bank(torch.stack([u for _,u in items]),group).float()
                scale=math.sqrt(max(1,shape[0]/shape[1])) if group['adjust_lr_fn']=='original' else .2*math.sqrt(max(shape))
                for (p,_),u in zip(items,updates):
                    p.mul_(1-group['lr']*group['weight_decay'])
                    p.add_(u,alpha=-group['lr']*scale)
        return loss
