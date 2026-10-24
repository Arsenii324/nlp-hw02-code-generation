"""Independent cached beam decoder; canonical measured inference stays greedy."""
import math
import torch
from torch.nn import functional as F
from .tokenizer import BOS,EOS,PAD


@torch.inference_mode()
def generate_beam(model,src,beam_size=4,length_penalty=0.,max_new_tokens=None):
    if model.training: raise RuntimeError('Call eval() before cached generation')
    if not isinstance(beam_size,int) or not 1<=beam_size<=model.cfg.vocab_size:
        raise ValueError('Invalid beam width')
    if not math.isfinite(length_penalty) or length_penalty<0: raise ValueError('Invalid length penalty')
    limit=model.cfg.max_target if max_new_tokens is None else max_new_tokens
    if not isinstance(limit,int) or limit<1: raise ValueError('Invalid token limit')
    limit=min(limit,model.cfg.max_target)
    # Preserve greedy's raw-logit tie rule and exact cache/layout path.
    if beam_size==1 and hasattr(model,'generate'):
        return model.generate(src,max_new_tokens=limit)
    batch=src.shape[0]; width=beam_size; device=src.device
    memory,valid=model.encode(src)
    base=model.new_cache(memory)
    cache=[tuple(x.repeat_interleave(width,dim=0) for x in layer) for layer in base]
    spare=[(torch.empty_like(sk),torch.empty_like(sv),ck,cv) for sk,sv,ck,cv in cache]
    valid=valid.repeat_interleave(width,dim=0)
    scores=torch.full((batch,width),-torch.inf,device=device,dtype=torch.float32); scores[:,0]=0
    done=torch.zeros(batch,width,device=device,dtype=torch.bool)
    lengths=torch.zeros(batch,width,device=device,dtype=torch.long)
    token=torch.full((batch*width,),BOS,device=device,dtype=torch.long)
    history=torch.full((batch*width,limit),PAD,device=device,dtype=torch.long)
    spare_history=torch.empty_like(history)
    offsets=torch.arange(batch,device=device)[:,None]*width
    vocab=model.cfg.vocab_size
    for t in range(limit):
        logits=model.decode_step(token,t,cache,valid)
        logprob=F.log_softmax(logits.float(),dim=-1).view(batch,width,vocab)
        logprob.masked_fill_(done[:,:,None],-torch.inf)
        logprob[:,:,EOS]=torch.where(done,torch.zeros_like(scores),logprob[:,:,EOS])
        candidates=scores[:,:,None]+logprob
        next_lengths=lengths+(~done).long()
        penalty=((5+next_lengths.float())/6).pow(length_penalty)
        ranking=candidates/penalty[:,:,None]
        if width==1:
            choices=logits.argmax(-1,keepdim=True)
            choices=torch.where(done,torch.full_like(choices,EOS),choices)
        else:
            _,choices=ranking.flatten(1).topk(width,dim=1)
        parents=choices.div(vocab,rounding_mode='floor')
        token=choices.remainder(vocab).flatten()
        global_parents=(parents+offsets).flatten()
        scores=candidates.flatten(1).gather(1,choices)
        dead=scores.eq(-torch.inf)
        token=torch.where(dead.flatten(),torch.full_like(token,EOS),token)
        lengths=next_lengths.gather(1,parents)
        done=done.gather(1,parents)|token.view(batch,width).eq(EOS)|dead
        if t: spare_history[:,:t].copy_(history[:,:t].index_select(0,global_parents))
        spare_history[:,t]=token
        history,spare_history=spare_history,history
        if done.all(): break
        # All parents remain in their source group: cross KV and valid masks
        # are identical across that group's beams and need no reordering.
        for (sk,sv,ck,cv),(next_sk,next_sv,_,_) in zip(cache,spare):
            next_sk[:,:,:t+1].copy_(sk[:,:,:t+1].index_select(0,global_parents))
            next_sv[:,:,:t+1].copy_(sv[:,:,:t+1].index_select(0,global_parents))
        cache,spare=spare,cache
    normalized=scores/((5+lengths.float())/6).pow(length_penalty)
    completed=normalized.masked_fill(~done,-torch.inf)
    best=torch.where(done.any(1),completed.argmax(1),normalized.argmax(1))
    return history[(best[:,None]+offsets[:,:1]).flatten(),:t+1]
