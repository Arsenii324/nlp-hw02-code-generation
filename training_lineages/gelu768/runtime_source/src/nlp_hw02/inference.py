"""Ordered batched inference; local JSONL is an internal artifact, not a bot schema."""
import ast
import time
import numpy as np
import torch
from .tokenizer import encode, decode, EOS


def padded_tokens(tok, texts, limit):
    ids=[encode(tok,s,limit) for s in texts]
    array=np.zeros((len(ids),max(map(len,ids))),dtype=np.int64)
    for i,row in enumerate(ids): array[i,:len(row)]=row
    return torch.from_numpy(array)


@torch.inference_mode()
def generate_questions(model,tok,questions,batch_size=256):
    model.eval()
    device=next(model.parameters()).device
    if device.type=='cuda': torch.cuda.synchronize()
    started=time.perf_counter()
    outputs=[]; lengths=[]; capped=0
    for start in range(0,len(questions),batch_size):
        src=padded_tokens(tok,questions[start:start+batch_size],model.cfg.max_source).to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            result=model.generate(src)
        for ids in result.cpu().tolist():
            ended=EOS in ids
            n=ids.index(EOS) if ended else len(ids)
            lengths.append(n); capped+=not ended
            outputs.append(decode(tok,ids))
    if device.type=='cuda': torch.cuda.synchronize()
    return outputs,{'seconds':time.perf_counter()-started,'rows':len(outputs),
        'batch_size':batch_size,'max_new_tokens':model.cfg.max_target,
        'mean_generated_tokens':float(np.mean(lengths)),'capped_rows':capped,
        'includes':'tokenization, host/device transfer, encoder, greedy cached decoder, detokenization; excludes checkpoint load and JSONL writing'}


def syntax_valid(code):
    if not code.strip(): return False
    try: ast.parse(code); return True
    except (SyntaxError, ValueError, RecursionError): return False
