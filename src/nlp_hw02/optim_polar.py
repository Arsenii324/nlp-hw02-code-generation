"""Paper-v5 Polar Express, five maps, with explicit grouped FP16 adaptation.

Source: https://arxiv.org/html/2505.16932v5#appendix.A
First five rows all use the source's coefficient1.01 safety rescaling.
FP32 normalization before the selectable low-precision cast differs from the
paper's BF16-first code. Momentum/decay/LR/state layout remain BatchedMuon's.
"""
import math
import torch
from .optim_batched import BatchedMuon

RAW_COEFFICIENTS=(
    (8.28721201814563,-23.595886519098837,17.300387312530933),
    (4.107059111542203,-2.9478499167379106,0.5448431082926601),
    (3.9486908534822946,-2.908902115962949,0.5518191394370137),
    (3.3184196573706015,-2.488488024314874,0.51004894012372),
    (2.300652019954817,-1.6689039845747493,0.4188073119525673),
)
COEFFICIENTS=tuple((a/1.01,b/1.01**3,c/1.01**5) for a,b,c in RAW_COEFFICIENTS)


@torch.no_grad()
def polar_express(bank,*,dtype=torch.float16,steps=5,eps=1e-7):
    if bank.ndim!=3 or bank.dtype!=torch.float32 or min(bank.shape)<=0:
        raise ValueError('Expected nonempty[count,rows,cols] FP32 matrix bank')
    if dtype not in (torch.float16,torch.bfloat16,torch.float32) or steps!=5:
        raise ValueError('This pinned Polar Express implementation uses exactly five maps')
    if not math.isfinite(eps) or eps<=0:raise ValueError('Invalid epsilon')
    tall=bank.shape[-2]>bank.shape[-1]
    with torch.autocast(device_type=bank.device.type,enabled=False):
        x=bank/(bank.norm(dim=(-2,-1),keepdim=True)*1.01+eps)
        if tall:x=x.transpose(-2,-1)
        x=x.to(dtype)
        for a,b,c in COEFFICIENTS:
            gram=torch.bmm(x,x.transpose(-2,-1))
            polynomial=torch.baddbmm(gram,gram,gram,beta=b,alpha=c)
            x=torch.baddbmm(x,polynomial,x,beta=a)
        return x.transpose(-2,-1) if tall else x


class PolarMuon(BatchedMuon):
    orthogonalizer_name='polar_express_v5_5'

    @staticmethod
    def _validate_group(group):
        BatchedMuon._validate_group(group)
        if group['ns_steps']!=5:raise ValueError('PolarMuon requires five fixed paper-v5 maps')

    def orthogonalize_bank(self,bank,group):
        return polar_express(bank,dtype=group['ns_dtype'],steps=group['ns_steps'],eps=group['eps'])
