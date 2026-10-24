"""Fixed/learned XSA restart oracle, including legacy config and gate routing."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from nlp_hw02.model import ModelConfig,Seq2Seq
from nlp_hw02.optim import make_optimizers
import check_training_resumption as oracle

p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');a=p.parse_args()
# Real model bank check on CPU: no weights/checkpoint or raw arrays downloaded.
m=Seq2Seq(ModelConfig(d_model=384,ff=1536,heads=4,encoder_layers=4,decoder_layers=4,learnable_xsa=True));opts,info=make_optimizers(m,'muon_grouped')
gates=[v for n,v in m.named_parameters() if n.endswith('xsa_alpha')]
assert sum(v.numel() for v in gates)==32 and info['xsa_gate_parameters']==32
for value in gates:
 groups=[g for o in opts for g in o.param_groups if any(v is value for v in g['params'])]
 assert len(groups)==1 and groups[0]['weight_decay']==0
assert sum(sum(v is gate for o in opts for g in o.param_groups for v in g['params']) for gate in gates)==len(gates)
del m,opts,gates
original_cfg=ModelConfig;original_save=torch.save
for variant in ('ordinary','fixed','learned'):
 def config(**kwargs):
  return original_cfg(**{**kwargs,'exclusive_self_attention':variant=='fixed','learnable_xsa':variant=='learned'})
 def legacy_save(value,path,*args,**kwargs):
  if isinstance(value,dict) and 'config' in value:
   cfg={**value['config']}
   for key in ('loop_order','loop_share_scope','loop_repeats','share_loop_weights','residual_scale'):cfg.pop(key,None)
   if not cfg.get('learnable_xsa'):cfg.pop('learnable_xsa',None)
   value={**value,'config':cfg}
  return original_save(value,path,*args,**kwargs)
 argv=['check_training_resumption','--no-polar-transfer-check']+(['--cuda'] if a.cuda else [])
 with patch.object(oracle,'ModelConfig',config),patch.object(sys,'argv',argv),patch.object(torch,'save',legacy_save):
  oracle.main()
 print(json.dumps({'event':'xsa_variant_resume_passed','variant':variant,'legacy_config_checked':True}),flush=True)
print(json.dumps({'event':'xsa_pair_resumption_checks_passed','device':'cuda' if a.cuda else 'cpu','real_gate_scalars':32,'gate_optimizer_unique_no_decay':True}),flush=True)
