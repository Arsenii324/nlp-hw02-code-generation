"""Ordered beam inference with the same timing boundaries as greedy inference."""
import math
import time
import torch
from .beam import generate_beam
from .inference import padded_tokens
from .tokenizer import decode,EOS


@torch.inference_mode()
def generate_beam_questions(model,tok,questions,batch_size=256,beam_size=4,length_penalty=.6):
    if not isinstance(batch_size,int) or batch_size<1: raise ValueError('Batch size must be positive')
    if not isinstance(beam_size,int) or not 1<=beam_size<=model.cfg.vocab_size: raise ValueError('Invalid beam width')
    if not math.isfinite(length_penalty) or length_penalty<0: raise ValueError('Invalid length penalty')
    model.eval(); device=next(model.parameters()).device
    if device.type=='cuda': torch.cuda.synchronize()
    start=time.perf_counter(); predictions=[]; lengths=[]; capped=0
    for offset in range(0,len(questions),batch_size):
        src=padded_tokens(tok,questions[offset:offset+batch_size],model.cfg.max_source).to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            tokens=generate_beam(model,src,beam_size=beam_size,length_penalty=length_penalty)
        for ids in tokens.cpu().tolist():
            ended=EOS in ids; lengths.append(ids.index(EOS) if ended else len(ids)); capped+=not ended
            predictions.append(decode(tok,ids))
    if device.type=='cuda': torch.cuda.synchronize()
    assert len(predictions)==len(questions)
    return predictions,{'seconds':time.perf_counter()-start,'rows':len(predictions),
        'batch_size':batch_size,'beam_width':beam_size,'length_penalty':length_penalty,'max_new_tokens':model.cfg.max_target,
        'mean_generated_tokens':sum(lengths)/len(lengths) if lengths else 0,'capped_rows':capped,
        'includes':'tokenization, transfers, encoder, cached beam decoder, detokenization; excludes loading, warm-up and writing'}
