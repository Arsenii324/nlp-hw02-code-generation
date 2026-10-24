"""Focused correctness checks; run on Colab before the experiment."""
import tempfile
from pathlib import Path
import torch
from nlp_hw02.model import Seq2Seq, ModelConfig, load_model
from nlp_hw02.tokenizer import train_tokenizer, encode, decode, PAD, BOS, EOS


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as d:
        tokenizer = train_tokenizer(['a = 1\n', 'print(a)\n', 'write a program'], Path(d)/'tok.json', 300)
        s = 'x = "Привет"\n  print(x)\n'
        ids = encode(tokenizer, s, 100)
        assert ids[-1] == EOS and decode(tokenizer, ids) == s
        assert encode(tokenizer, s, 4)[-1] == EOS
        assert PAD == 0 and BOS == 1 and EOS == 2
    cfg = ModelConfig(vocab_size=300, d_model=32, heads=4, encoder_layers=1, decoder_layers=2, ff=64, dropout=0, max_source=16, max_target=16)
    m = Seq2Seq(cfg).eval()
    src = torch.tensor([[8, 9, EOS, PAD], [10, EOS, PAD, PAD]])
    tgt = torch.tensor([[BOS, 11, 12], [BOS, 13, 14]])
    with torch.no_grad():
        logits = m(src, tgt)
        changed = tgt.clone(); changed[:, -1] = 17
        torch.testing.assert_close(logits[:, :-1], m(src, changed)[:, :-1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(logits, m(torch.cat([src, torch.zeros(2, 2, dtype=torch.long)], 1), tgt), atol=1e-6, rtol=1e-5)
        memory, valid = m.encode(src)
        cache = m.new_cache(memory)
        for t in range(tgt.shape[1]):
            step = m.decode_step(tgt[:, t], t, cache, valid)
            torch.testing.assert_close(step, logits[:, t], atol=2e-6, rtol=1e-5)
        greedy = m.generate(src, max_new_tokens=6)
        ref = torch.full((2,1), BOS, dtype=torch.long)
        ended = torch.zeros(2,dtype=torch.bool)
        for _ in range(6):
            next_id = m(src, ref)[:, -1].argmax(-1)
            next_id = torch.where(ended, torch.full_like(next_id, EOS), next_id)
            ref = torch.cat([ref, next_id[:,None]], 1)
            ended |= next_id.eq(EOS)
            if ended.all(): break
        assert torch.equal(greedy, ref[:,1:])
    m.train()
    loss = torch.nn.functional.cross_entropy(m(src,tgt).flatten(0,1), tgt.flatten())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    from dataclasses import asdict
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'model.pt'
        torch.save({'config':asdict(cfg),'model':m.state_dict(),'step':1},path)
        restored,record=load_model(path,'cpu')
        m.eval()
        with torch.no_grad(): torch.testing.assert_close(restored(src,tgt),m(src,tgt))
        assert record['step']==1
    # A real model forced to choose EOS should stop all rows after one token.
    stop_model=Seq2Seq(cfg).eval()
    with torch.no_grad():
        for p in stop_model.parameters(): p.zero_()
        stop_model.dec_norm.weight.fill_(1)
        stop_model.embedding.weight[BOS,0]=1
        stop_model.embedding.weight[EOS,0]=2
        assert torch.equal(stop_model.generate(src,max_new_tokens=5),torch.full((2,1),EOS))
    from nlp_hw02.train import shifted
    assert torch.equal(shifted(torch.tensor([[8,EOS,PAD]])),torch.tensor([[BOS,8,EOS]]))
    from nlp_hw02.model import exclusive_projection
    y=torch.randn(2,4,5,8); own=torch.randn_like(y)
    z=exclusive_projection(y,own)
    assert (z*own).sum(-1).abs().max()<2e-6
    assert torch.isfinite(exclusive_projection(y,torch.zeros_like(own))).all()
    cfg.exclusive_self_attention=True
    xsa=Seq2Seq(cfg).eval()
    with torch.no_grad():
        full=xsa(src,tgt); memory,valid=xsa.encode(src); cache=xsa.new_cache(memory)
        for t in range(tgt.shape[1]):
            torch.testing.assert_close(xsa.decode_step(tgt[:,t],t,cache,valid),full[:,t],atol=2e-6,rtol=1e-5)
    from nlp_hw02.optim import make_optimizers
    muon_available=hasattr(torch.optim,'Muon')
    opts,info=make_optimizers(xsa,'muon' if muon_available else 'adamw')
    params=[p for opt in opts for group in opt.param_groups for p in group['params']]
    assert len(params)==len(list(xsa.parameters())) and len({id(p) for p in params})==len(params)
    if muon_available: assert all(p.ndim==2 for p in opts[0].param_groups[0]['params'])
    xsa.train(); xsa(src,tgt).sum().backward()
    for opt in opts: opt.step()
    assert all(torch.isfinite(p).all() for p in xsa.parameters())
    cfg.exclusive_self_attention=False
    cfg.activation='relu2'
    relu2=Seq2Seq(cfg).eval()
    with torch.no_grad():
        full=relu2(src,tgt); memory,valid=relu2.encode(src); cache=relu2.new_cache(memory)
        for t in range(tgt.shape[1]):
            torch.testing.assert_close(relu2.decode_step(tgt[:,t],t,cache,valid),full[:,t],atol=2e-6,rtol=1e-5)
    from nlp_hw02.model import SquaredReLU
    torch.testing.assert_close(SquaredReLU()(torch.tensor([-2.,0.,3.])),torch.tensor([0.,0.,9.]))
    # Flags introduce no new random parameters: matched seed means matched tensors.
    initial=[]
    for exclusive,activation in [(False,'gelu'),(True,'gelu'),(False,'relu2')]:
        torch.manual_seed(42)
        cfg.exclusive_self_attention=exclusive; cfg.activation=activation
        initial.append(Seq2Seq(cfg).state_dict())
    for other in initial[1:]:
        assert initial[0].keys()==other.keys()
        assert all(torch.equal(initial[0][k],other[k]) for k in other)
    print('PASS: tokenizer Unicode/whitespace/EOS; causal masking; source padding; cached logits and greedy equivalence; finite gradients; checkpoint round-trip; EOS stop; target shift')
    print('PASS: XSA orthogonality/zero-value safety/cached equivalence; optimizer exact parameter partition and finite update')
    print('Muon checks: PASS' if muon_available else 'Muon checks: SKIPPED (not present in installed torch; no Muon trial authorized by this check)')
    print('PASS: ReLU-squared operator and cached/full decoder equivalence; identical initial tensors across controlled variants')


if __name__ == '__main__': main()
