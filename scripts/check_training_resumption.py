"""Stochastic CPU restart oracle; deliberately no extra GPU allocation."""
from copy import deepcopy
from dataclasses import asdict
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from nlp_hw02.model import ModelConfig,Seq2Seq
from nlp_hw02.optim import make_optimizers
from nlp_hw02.resuming import restore_training


def nested_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--no-polar-transfer-check',action='store_true');args=parser.parse_args()
    device='cuda' if args.cuda else 'cpu'
    if args.cuda:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    cfg=ModelConfig(vocab_size=40,d_model=16,heads=2,encoder_layers=1,
        decoder_layers=1,ff=32,dropout=.1,max_source=8,max_target=8,learnable_xsa=True)
    data={'dataset_revision':'cpu-resume-fixture','train_examples':100}
    seed=42;all_results=[]
    def initialize(kind):
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=Seq2Seq(cfg).to(device).train();opts,_=make_optimizers(model,kind)
        scaler=torch.amp.GradScaler(device,init_scale=128.)
        return model,opts,scaler,rng
    def steps(model,opts,scaler,rng,start,end):
        draws=[]
        for step in range(start+1,end+1):
            a=rng.integers(4,40,size=(4,5));b=rng.integers(4,40,size=(4,4));a[:,-1]=2;b[:,-1]=2
            draws.append((a.tolist(),b.tolist()))
            src=torch.tensor(a,device=device);target=torch.tensor(b,device=device);shifted=torch.cat([torch.ones_like(target[:,:1]),target[:,:-1]],1)
            for opt in opts:
                for group in opt.param_groups:group['lr']=6e-4*(step/200)*group['lr_multiplier']
                opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16) if args.cuda else nullcontext():
                loss=torch.nn.functional.cross_entropy(model(src,shifted).flatten(0,1),target.flatten())
            scaler.scale(loss).backward()
            for opt in opts:scaler.unscale_(opt)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(grad_norm)
            for opt in opts:scaler.step(opt)
            scaler.update()
        return draws
    for kind in ['adamw','muon_grouped']:
        model,opts,scaler,rng=initialize(kind)
        expected_draws=steps(model,opts,scaler,rng,0,6)
        expected={'model':deepcopy(model.state_dict()),'optimizers':deepcopy([o.state_dict() for o in opts]),
            'scaler':deepcopy(scaler.state_dict()),'rng_numpy':deepcopy(rng.bit_generator.state),'rng_torch':torch.get_rng_state().clone(),
            'rng_cuda':torch.cuda.get_rng_state_all() if args.cuda else []}
        first,first_opts,first_scaler,first_rng=initialize(kind)
        first_draws=steps(first,first_opts,first_scaler,first_rng,0,3)
        state={'config':asdict(cfg),'model':first.state_dict(),'optimizers':[o.state_dict() for o in first_opts],
            'scaler':first_scaler.state_dict(),'step':3,'seed':seed,'dataset_revision':data['dataset_revision'],
            'sampled_examples':3*4,'optimizer_updates':3,'skipped_updates':0,
            'validation_loss':1.,'best_validation_loss':1.,'rng_numpy_generator':first_rng.bit_generator.state,
            'rng_torch_cpu':torch.get_rng_state(),'rng_torch_cuda':torch.cuda.get_rng_state_all() if args.cuda else []}
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            _,info=make_optimizers(first,kind)
            metadata={'torch':str(torch.__version__),'cuda':torch.version.cuda,'data':data,'seed':seed,'batch_size':4,'optimizer':info}
            (directory/'run_config.json').write_text(json.dumps(metadata))
            torch.save(state,directory/'last_training.pt')
            best={k:v for k,v in state.items() if k in ['config','model','step','seed','dataset_revision','validation_loss']}
            torch.save(best,directory/'model.pt')
            resumed,resumed_opts,resumed_scaler,resumed_rng=initialize(kind)
            restored=restore_training(resumed,resumed_opts,resumed_scaler,resumed_rng,directory,seed=seed,data=data,batch_size=4)
            assert restored['step']==3 and restored['best_step']==3
            assert nested_equal(resumed.state_dict(),state['model'])
            assert nested_equal([o.state_dict() for o in resumed_opts],state['optimizers'])
            assert nested_equal(resumed_scaler.state_dict(),state['scaler'])
            assert nested_equal(resumed_rng.bit_generator.state,state['rng_numpy_generator'])
            assert torch.equal(torch.get_rng_state(),state['rng_torch_cpu'])
            assert nested_equal(torch.cuda.get_rng_state_all() if args.cuda else [],state['rng_torch_cuda'])
            remaining_draws=steps(resumed,resumed_opts,resumed_scaler,resumed_rng,3,6)
            actual={'model':resumed.state_dict(),'optimizers':[o.state_dict() for o in resumed_opts],
                'scaler':resumed_scaler.state_dict(),'rng_numpy':resumed_rng.bit_generator.state,'rng_torch':torch.get_rng_state(),
                'rng_cuda':torch.cuda.get_rng_state_all() if args.cuda else []}
            assert first_draws+remaining_draws==expected_draws
            assert nested_equal(actual,expected),kind+' interrupted/restarted path differs'
            # Reject known incompatible counters/settings/provenance before updates.
            rejections=0
            for field,value in [('step',4),('sampled_examples',1),('seed',43),('skipped_updates',2)]:
                bad={**state,field:value};torch.save(bad,directory/'last_training.pt')
                try:restore_training(resumed,resumed_opts,resumed_scaler,resumed_rng,directory,seed=seed,data=data,batch_size=4)
                except ValueError:rejections+=1
                else:raise AssertionError('failed to reject '+field)
            torch.save(state,directory/'last_training.pt')
            if kind=='muon_grouped' and not args.no_polar_transfer_check:
                changed,changed_opts,changed_scaler,changed_rng=initialize('muon_grouped_pe')
                try:restore_training(changed,changed_opts,changed_scaler,changed_rng,directory,seed=seed,data=data,batch_size=4)
                except ValueError:pass
                else:raise AssertionError('undeclared orthogonalizer change accepted')
                transfer=restore_training(changed,changed_opts,changed_scaler,changed_rng,directory,seed=seed,data=data,batch_size=4,allow_orthogonalizer_change=True)
                assert transfer['orthogonalizer_change']['from']=='newton_schulz'
                assert transfer['orthogonalizer_change']['to']=='polar_express_v5_5'
                assert nested_equal([o.state_dict() for o in changed_opts],state['optimizers'])
                assert nested_equal(changed.state_dict(),state['model'])
                steps(changed,changed_opts,changed_scaler,changed_rng,3,4)
                assert all(torch.isfinite(p).all() for p in changed.parameters())
            bad_meta={**metadata,'torch':'different'};(directory/'run_config.json').write_text(json.dumps(bad_meta))
            try:restore_training(resumed,resumed_opts,resumed_scaler,resumed_rng,directory,seed=seed,data=data,batch_size=4)
            except ValueError:rejections+=1
            else:raise AssertionError('failed to reject software mismatch')
            all_results.append({'optimizer':kind,'exact_draws_weights_states_scaler_rng':True,'rejections':rejections,
                'declared_NS_to_PE_state_transfer_and_finite_update':kind=='muon_grouped' and not args.no_polar_transfer_check,
                'immediate_all_state_equality':True})
    print(json.dumps({'event':'training_resumption_checks_passed','torch':str(torch.__version__),
        'device':device,'CUDA_restart_verified':args.cuda,'results':all_results}),flush=True)


if __name__=='__main__':main()
