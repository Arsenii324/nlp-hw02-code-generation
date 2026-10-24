"""Pinned-map oracle and finite precision smoke check; --cuda uses one visible T4."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile
import torch
from nlp_hw02.optim_batched import BatchedMuon,orthogonalize
from nlp_hw02.optim_polar import PolarMuon,polar_express,COEFFICIENTS


def reference(matrix,dtype=torch.float32):
    x=matrix/(matrix.norm()*1.01+1e-7);tall=matrix.shape[0]>matrix.shape[1]
    if tall:x=x.T
    x=x.to(dtype)
    for a,b,c in COEFFICIENTS:
        gram=x@x.T;x=a*x+(b*gram+c*(gram@gram))@x
    return x.T if tall else x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261002)
    device='cuda' if args.cuda else 'cpu'
    if args.cuda:
        assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    summaries=[]
    for rows,cols in [(16,32),(32,16),(32,32)]:
        bank=torch.randn(3,rows,cols,device=device)
        fp32=polar_express(bank,dtype=torch.float32)
        oracle=torch.stack([reference(matrix) for matrix in bank])
        torch.testing.assert_close(fp32,oracle,atol=3e-5,rtol=3e-5)
        u,s,vh=torch.linalg.svd(bank,full_matrices=False);exact=u@vh
        pe_error=float((fp32-exact).norm()/exact.norm())
        ns_error=float((orthogonalize(bank,dtype=torch.float32)-exact).norm()/exact.norm())
        assert pe_error<ns_error,'pinned PE failed this well-conditioned polar fixture'
        row={'shape':[3,rows,cols],'PE_relative_polar_error':pe_error,'NS_relative_polar_error':ns_error,
             'reference_max_absolute_difference':float((fp32-oracle).abs().max()),'precision':{}}
        for dtype in (torch.float16,torch.bfloat16):
            low=polar_express(bank,dtype=dtype).float();assert torch.isfinite(low).all()
            row['precision'][str(dtype)]={'relative_vs_FP32':float((low-fp32).norm()/fp32.norm()),
                'update_rms':float(low.square().mean().sqrt())}
        summaries.append(row)
        for scale in (0.,1e-12,1e5):
            result=polar_express(bank*scale);assert torch.isfinite(result).all()
            if scale==0:assert torch.count_nonzero(result)==0
        rank_one=torch.randn(3,rows,1,device=device)*torch.randn(3,1,cols,device=device)
        assert torch.isfinite(polar_express(rank_one)).all()
    # Same fixed gradients -> same momentum, a different orthogonalized update.
    initial=torch.randn(16,32,device=device);a=torch.nn.Parameter(initial.clone());b=torch.nn.Parameter(initial.clone())
    ns=BatchedMuon([a]);pe=PolarMuon([b]);gradient=torch.randn_like(initial)
    a.grad=gradient.clone();b.grad=gradient.clone();ns.step();pe.step()
    assert torch.equal(ns.state[a]['momentum_buffer'],pe.state[b]['momentum_buffer'])
    assert not torch.equal(a,b) and torch.isfinite(b).all()
    # Prove the default NS path remains the recovered implementation on CPU.
    preserved=Path('artifacts/kaggle/b011-b012-v1/training_source.bin')
    old_checked=False
    if preserved.exists() and not args.cuda:
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old_optim_batched.py'
            with zipfile.ZipFile(preserved) as archive:path.write_bytes(archive.read('src/nlp_hw02/optim_batched.py'))
            spec=importlib.util.spec_from_file_location('old_optim_batched',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            old_param=torch.nn.Parameter(initial.clone());new_param=torch.nn.Parameter(initial.clone())
            old=module.BatchedMuon([old_param]);new=BatchedMuon([new_param])
            for _ in range(3):
                grad=torch.randn_like(initial);old_param.grad=grad.clone();new_param.grad=grad.clone();old.step();new.step()
                assert torch.equal(old_param,new_param)
                assert torch.equal(old.state[old_param]['momentum_buffer'],new.state[new_param]['momentum_buffer'])
            old_checked=True
    # On actual T4, also check representative matrix sizes before a long run.
    if args.cuda:
        for shape in [(4,384,384),(4,1536,384),(4,384,1536)]:
            bank=torch.randn(shape,device=device);out=polar_express(bank)
            assert torch.isfinite(out).all();summaries.append({'representative_shape':list(shape),'finite':True,'update_rms':float(out.float().square().mean().sqrt())})
    print(json.dumps({'event':'polar_muon_checks_passed','device':device,'torch':str(torch.__version__),
        'CUDA_T4_verified':args.cuda,'coefficient_source':'arxiv2505.16932v5 AppendixA first5 safety',
        'default_NS_frozen_cpu_equivalence':old_checked,'same_fixed_gradient_momentum':True,'cases':summaries}),flush=True)


if __name__=='__main__':main()
