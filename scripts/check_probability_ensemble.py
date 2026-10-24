"""Arithmetic-mixture and cached/full-prefix beam oracles; no data or allocation."""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import json
import torch
from nlp_hw02.model import ModelConfig,Seq2Seq
from nlp_hw02.beam import generate_beam
from nlp_hw02.ensemble import ProbabilityEnsemble


def main():
 p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');a=p.parse_args()
 device='cuda' if a.cuda else 'cpu';torch.set_num_threads(2)
 if a.cuda:assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
 cfg=ModelConfig(vocab_size=32,d_model=16,heads=2,encoder_layers=1,decoder_layers=2,ff=32,dropout=0,max_source=8,max_target=8)
 torch.manual_seed(19);one=Seq2Seq(cfg).to(device).eval()
 torch.manual_seed(23);two=Seq2Seq(ModelConfig(**{**cfg.__dict__,'learnable_xsa':True})).to(device).eval()
 with torch.no_grad():
  for n,param in two.named_parameters():
   if n.endswith('xsa_alpha'):param.fill_(.7)
 ensemble=ProbabilityEnsemble([one,two],[.25,.75]).eval()
 src=torch.tensor([[4,2,0],[6,7,2]],device=device);target=torch.tensor([[1,11,12],[1,14,15]],device=device)
 with torch.inference_mode():
  for fp16 in ([False,True] if a.cuda else [False]):
   with torch.autocast('cuda',dtype=torch.float16,enabled=fp16) if a.cuda else nullcontext():
    expected=(.25*one(src,target).float().softmax(-1)+.75*two(src,target).float().softmax(-1)).log()
    actual=ensemble(src,target);torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    memory,valid=ensemble.encode(src);cache=ensemble.new_cache(memory)
    assert len(cache)==4 and cache[0][0].data_ptr()!=cache[2][0].data_ptr()
    for t in range(target.shape[1]):
     step=ensemble.decode_step(target[:,t],t,cache,valid)
     torch.testing.assert_close(step,actual[:,t],atol=4e-3 if fp16 else 3e-6,rtol=3e-4 if fp16 else 1e-5)
    class FullPrefix:
     training=False
     cfg=ensemble.cfg
     def encode(self,source):return source[:,:,None].float(),source.ne(0)
     def new_cache(self,memory):
      prefix=torch.empty(memory.shape[0],1,self.cfg.max_target,1,device=memory.device)
      return [(prefix,torch.empty_like(prefix),memory[:,None],memory[:,None])]
     def decode_step(self,token,t,cache,valid):
      sk,_,ck,_=cache[0];sk[:,:,t,0]=token[:,None].float()
      return ensemble(ck[:,0,:,0].long(),sk[:,0,:t+1,0].long())[:,-1]
    for width in [1,2,4,8]:
     for alpha in [0.,.6]:
      actual_tokens=generate_beam(ensemble,src,beam_size=width,length_penalty=alpha,max_new_tokens=4)
      expected_tokens=generate_beam(FullPrefix(),src,beam_size=width,length_penalty=alpha,max_new_tokens=4)
      assert torch.equal(actual_tokens,expected_tokens),(width,alpha,fp16)
    singleton=ProbabilityEnsemble([one]).eval();duplicates=ProbabilityEnsemble([one,deepcopy(one)]).eval()
    for width in [1,4]:
     reference=generate_beam(one,src,beam_size=width,length_penalty=.6,max_new_tokens=4)
     assert torch.equal(reference,generate_beam(singleton,src,beam_size=width,length_penalty=.6,max_new_tokens=4))
     assert torch.equal(reference,generate_beam(duplicates,src,beam_size=width,length_penalty=.6,max_new_tokens=4))
 bad=[]
 for models,weights in [([],None),([one,two],[1.]),([one,two],[-1.,2.]),([one,two],[0.,0.]),([one,two],[float('nan'),1.])]:
  try:ProbabilityEnsemble(models,weights)
  except ValueError:bad.append('rejected')
  else:raise AssertionError('invalid weights/models accepted')
 other=Seq2Seq(ModelConfig(**{**cfg.__dict__,'vocab_size':33})).to(device).eval()
 try:ProbabilityEnsemble([one,other])
 except ValueError:bad.append('vocabulary rejected')
 else:raise AssertionError('incompatible vocabulary')
 ensemble.train()
 try:generate_beam(ensemble,src)
 except RuntimeError:bad.append('training rejected')
 else:raise AssertionError('training ensemble accepted')
 print(json.dumps({'event':'probability_ensemble_checks_passed','device':device,'arithmetic_probability_mean':True,'full_and_cached_beam_equal':True,'distinct_cache_banks':True,'single_duplicate_parity':True,'rejections':len(bad)}),flush=True)


if __name__=='__main__':main()
