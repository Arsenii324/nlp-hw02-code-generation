"""Catch accidentally routing wider beam back to ordinary120k/180k model."""
import torch
from nlp_hw02.model import Seq2Seq,ModelConfig
from nlp_hw02.ensemble import ProbabilityEnsemble
import inference_mixture_probe as probe
assert hasattr(probe,'select_probe_model'), 'Missing declared-model lane routing'
torch.set_num_threads(2)
o=Seq2Seq(ModelConfig(vocab_size=32,d_model=16,heads=2,encoder_layers=1,decoder_layers=1,ff=32,dropout=0.,max_source=8,max_target=8)).eval()
l=Seq2Seq(ModelConfig(**{**o.cfg.__dict__,'learnable_xsa':True})).eval()
models={'b027':o,'b028':l}
ensemble=probe.select_probe_model(models,{'parents':['b027','b028'],'mechanism':'equal_token_probability_ensemble','weights':[.5,.5]})
assert isinstance(ensemble,ProbabilityEnsemble)
x=torch.tensor([[4,2]]);y=torch.tensor([[1,5]])
with torch.inference_mode():
 expected=(.5*o(x,y).float().softmax(-1)+.5*l(x,y).float().softmax(-1)).log()
 torch.testing.assert_close(ensemble(x,y),expected,atol=1e-6,rtol=1e-6)
assert probe.select_probe_model(models,{'parents':['b028'],'mechanism':'learned_wider_beam'}) is l
assert probe.select_probe_model(models,{'parents':['b027'],'mechanism':'ordinary_wider_beam'}) is o
for spec in [{'parents':['b027'],'mechanism':'equal_token_probability_ensemble','weights':[.5,.5]},{'parents':['b027','b028'],'mechanism':'learned_wider_beam'},{'parents':['b028'],'mechanism':'unknown'}]:
 try:probe.select_probe_model(models,spec)
 except ValueError:pass
 else:raise AssertionError('Invalid modelrouting accepted')
print('PASS learned widerbeam exact model / arithmeticensemble / legacyordinary / wrongrolesreject')
