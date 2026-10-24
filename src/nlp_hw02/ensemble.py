"""Frozen token-probability mixture behind the existing cached beam interface."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ProbabilityEnsemble(nn.Module):
    """Independent encoders/KV banks; one shared prefix and beam-parent ordering."""
    def __init__(self,models,weights=None):
        super().__init__()
        models=list(models)
        if not models:raise ValueError('At least one model required')
        weights=[1./len(models)]*len(models) if weights is None else list(weights)
        if len(weights)!=len(models) or any(not math.isfinite(w) or w<=0 for w in weights):
            raise ValueError('One strictly positive finite weight per model required')
        total=sum(weights)
        if not math.isfinite(total):raise ValueError('Finite total weight required')
        config=models[0].cfg;device=next(models[0].parameters()).device
        for model in models:
            if any(getattr(model.cfg,k)!=getattr(config,k) for k in ('vocab_size','max_source','max_target')):
                raise ValueError('Models require matching token vocabulary and length caps')
            if any(p.device!=device for p in model.parameters()):raise ValueError('Models must share a device')
        self.models=nn.ModuleList(models);self.cfg=config
        self.log_weights=tuple(math.log(w/total) for w in weights)
        self.cache_counts=tuple(len(list(model.decoder_steps())) for model in models)

    def mix(self,logits):
        if len(logits)==1:return logits[0]
        terms=[F.log_softmax(value.float(),dim=-1)+weight for value,weight in zip(logits,self.log_weights)]
        return torch.logsumexp(torch.stack(terms),dim=0)

    def forward(self,src,tgt):
        return self.mix([model(src,tgt) for model in self.models])

    def encode(self,src):
        memories=[];valid=None
        for model in self.models:
            memory,current=model.encode(src);memories.append(memory)
            if valid is None:valid=current
            elif not torch.equal(valid,current):raise ValueError('Source validity masks differ')
        return memories,valid

    def new_cache(self,memories):
        if len(memories)!=len(self.models):raise ValueError('One memory per model required')
        cache=[]
        for model,memory,count in zip(self.models,memories,self.cache_counts):
            own=model.new_cache(memory)
            if len(own)!=count:raise ValueError('Decoder application count changed')
            cache.extend(own)
        return cache

    def decode_step(self,token,t,cache,valid):
        if len(cache)!=sum(self.cache_counts):raise ValueError('Wrong number of model cache banks')
        logits=[];offset=0
        for model,count in zip(self.models,self.cache_counts):
            logits.append(model.decode_step(token,t,cache[offset:offset+count],valid));offset+=count
        return self.mix(logits)
