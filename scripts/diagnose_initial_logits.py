"""CPU held-out diagnostic for tied-head initialization; no training or test use."""
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from nlp_hw02.model import Seq2Seq,ModelConfig
from nlp_hw02.inference import padded_tokens
from nlp_hw02.train import shifted
from nlp_hw02.tokenizer import PAD
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
cache=ROOT/'artifacts/kaggle/prepare-v2/cache';torch.set_num_threads(2)
rows=json.loads((cache/'validation.json').read_text())
indices=__import__('numpy').random.default_rng(20261002).permutation(len(rows))[:64]
rows=[rows[int(i)] for i in indices]
tok=Tokenizer.from_file(str(cache/'tokenizer.json'))
src=padded_tokens(tok,[r['question'] for r in rows],256)
tgt=padded_tokens(tok,[r['code'] for r in rows],192);previous=shifted(tgt);valid=tgt.ne(PAD)
configs={'B008':ModelConfig(d_model=384,ff=1536),
         'B011':ModelConfig(d_model=512,ff=2048),
         'B012':ModelConfig(d_model=384,ff=1536,encoder_layers=4,decoder_layers=4)}
result={'device':'CPU','dtype':'FP32','rows':64,'validation_sha256':hashlib.sha256((cache/'validation.json').read_bytes()).hexdigest(),
    'sample_seed':20261002,'model_seed':42,'source_sha256':hashlib.sha256((ROOT/'src/nlp_hw02/model.py').read_bytes()).hexdigest(),
    'note':'Same sampled held-out teacher-forced positions, freshly initialized models. Diagnostics only; not trained accuracy, not proof of final loss attribution or a muP-controlled width study. No test prompts used.','models':{}}
with torch.inference_mode():
 for label,cfg in configs.items():
  torch.manual_seed(42);model=Seq2Seq(cfg).eval();logits=model(src,previous)
  copied=logits.gather(-1,previous[:,:,None]).squeeze(-1)
  true=logits.gather(-1,tgt[:,:,None]).squeeze(-1)
  others=(logits.sum(-1)-copied)/(cfg.vocab_size-1)
  p=logits.softmax(-1).gather(-1,previous[:,:,None]).squeeze(-1)
  result['models'][label]={'parameters':sum(t.numel() for t in model.parameters()),
   'initial_ce':float(F.cross_entropy(logits.flatten(0,1),tgt.flatten(),ignore_index=PAD)),
   'mean_previous_token_logit':float(copied[valid].mean()),'mean_other_token_logit':float(others[valid].mean()),
   'mean_target_logit':float(true[valid].mean()),'previous_token_argmax_fraction':float((logits.argmax(-1)==previous)[valid].float().mean()),
   'mean_previous_token_probability':float(p[valid].mean()),'target_equals_previous_fraction':float((tgt==previous)[valid].float().mean()),
   'residual_coefficients':1.0,'embedding_std':.02,'input_embedding_scale':cfg.d_model**.5,'tied_output_scale':1.0}
  del model,logits
out=ROOT/'reports/evidence/initial_logits_diagnostic.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
