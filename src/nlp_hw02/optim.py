"""Controlled optimizer alternatives; Muon only receives hidden 2-D matrices."""
import torch


def make_optimizers(model,kind='adamw'):
    kwargs=dict(lr=6e-4,betas=(.9,.98),weight_decay=.01,fused=next(model.parameters()).is_cuda)
    if kind=='adamw':
        regular=[p for name,p in model.named_parameters() if not name.endswith('.xsa_alpha')]
        gates=[p for name,p in model.named_parameters() if name.endswith('.xsa_alpha')]
        groups=[{'params':regular}]
        if gates: groups.append({'params':gates,'weight_decay':0.})
        opt=torch.optim.AdamW(groups,**kwargs)
        for group in opt.param_groups: group['lr_multiplier']=1.
        return [opt],{'kind':'AdamW','peak_lr':6e-4,'betas':[.9,.98],'weight_decay':.01}
    if kind not in ('muon','muon_grouped','muon_grouped_pe'): raise ValueError(kind)
    matrices=[]; other=[]; names=[]; gates=[]; gate_names=[]
    for name,p in model.named_parameters():
        if name.endswith('.xsa_alpha'):
            gates.append(p); gate_names.append(name)
        elif p.ndim==2 and name.startswith(('encoder.','decoder.')):
            matrices.append(p); names.append(name)
        else: other.append(p)
    if kind in ('muon_grouped','muon_grouped_pe'):
        from .optim_batched import BatchedMuon
        if kind=='muon_grouped_pe':
            from .optim_polar import PolarMuon
            implementation=PolarMuon
        else:implementation=BatchedMuon
        muon=implementation(matrices,lr=.003,weight_decay=.01,momentum=.95,
            nesterov=True,ns_steps=5,adjust_lr_fn='match_rms_adamw',ns_dtype=torch.float16)
    else:
        muon=torch.optim.Muon(matrices,lr=.003,weight_decay=.01,momentum=.95,
            nesterov=True,ns_steps=5,adjust_lr_fn='match_rms_adamw')
    groups=[{'params':other}]
    if gates: groups.append({'params':gates,'weight_decay':0.})
    adam=torch.optim.AdamW(groups,**kwargs)
    for group in muon.param_groups: group['lr_multiplier']=5.
    for group in adam.param_groups: group['lr_multiplier']=1.
    return [muon,adam],{'kind':('hybrid grouped FP16 PolarExpress Muon+AdamW' if kind=='muon_grouped_pe' else
        'hybrid grouped FP16 Muon+AdamW' if kind=='muon_grouped' else 'hybrid Muon+AdamW'),
        'orthogonalizer':getattr(muon,'orthogonalizer_name','stock_PyTorch'),
        'coefficient_source':'arxiv2505.16932v5 AppendixA first5 with1.01 safety' if kind=='muon_grouped_pe' else None,
        'ns_precision':'FP16 with FP32 normalization' if kind.startswith('muon_grouped') else 'stock PyTorch BF16',
        'muon_peak_lr':.003,'adamw_peak_lr':6e-4,
        'muon_momentum':.95,'muon_nesterov':True,'muon_ns_steps':5,
        'adjust_lr_fn':'match_rms_adamw','weight_decay':.01,
        'matrix_names':names,'muon_parameters':sum(p.numel() for p in matrices),
        'adamw_parameters':sum(p.numel() for p in other),
        'xsa_gate_names':gate_names,'xsa_gate_parameters':sum(p.numel() for p in gates),
        'xsa_gate_peak_lr':6e-4 if gates else None,'xsa_gate_weight_decay':0. if gates else None,
        'lr_reason':'Muon normalized update RMS is ~0.2 under shape adjustment; 5x base LR aims to match early AdamW ~1 RMS, not a tuned optimum'}
