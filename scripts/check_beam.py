"""Deterministic cache/parent correctness checks; no data/targets or remote run."""
import torch
from types import SimpleNamespace
from nlp_hw02.beam import generate_beam
from nlp_hw02.model import Seq2Seq,ModelConfig
from nlp_hw02.tokenizer import EOS


class Toy:
    training=False
    cfg=SimpleNamespace(vocab_size=5,max_target=3)
    def encode(self,src): return src[:,:1,None].float(),src[:,:1].ne(-1)
    def new_cache(self,memory):
        sk=torch.empty(memory.shape[0],1,3,1)
        return [(sk,torch.empty_like(sk),memory[:,None],memory[:,None])]
    def decode_step(self,token,t,cache,valid):
        sk,sv,ck,cv=cache[0]; sk[:,:,t,0]=token[:,None].float(); sv[:,:,t,0]=token[:,None].float()
        result=torch.full((len(token),5),-torch.inf)
        for row in range(len(token)):
            if t==0: probabilities=[0.,0.,0.,.6,.4] if ck[row,0,0,0]==0 else [0.,0.,0.,.9,.1]
            elif t==1: probabilities=[0.,0.,.01,.51,.48] if token[row]==3 else [0.,0.,.005,.99,.005]
            else: probabilities=[0.,0.,.34,.33,.33] if sk[row,0,1,0]==3 else [0.,0.,.99,.005,.005]
            result[row]=torch.tensor(probabilities).log()
        return result


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    model=Seq2Seq(ModelConfig(vocab_size=32,d_model=16,heads=2,encoder_layers=1,
        decoder_layers=2,ff=32,dropout=0,max_source=8,max_target=8)).eval()
    src=torch.tensor([[4,EOS,0],[6,7,EOS]])
    for cap in [1,3,8]:
        assert torch.equal(generate_beam(model,src,beam_size=1,max_new_tokens=cap),model.generate(src,max_new_tokens=cap))
    class FullPrefix:
        training=False
        cfg=model.cfg
        def encode(self,source): return source[:,:,None].float(),source.ne(0)
        def new_cache(self,memory):
            prefix=torch.empty(memory.shape[0],1,self.cfg.max_target,1)
            return [(prefix,torch.empty_like(prefix),memory[:,None],memory[:,None])]
        def decode_step(self,token,t,cache,valid):
            sk,sv,ck,cv=cache[0]; sk[:,:,t,0]=token[:,None].float()
            return model(ck[:,0,:,0].long(),sk[:,0,:t+1,0].long())[:,-1]
    for width in [2,3]:
        for alpha in [0.,.6]:
            assert torch.equal(generate_beam(model,src,beam_size=width,length_penalty=alpha,max_new_tokens=4),
                generate_beam(FullPrefix(),src,beam_size=width,length_penalty=alpha,max_new_tokens=4))
    toy=Toy(); source=torch.tensor([[0],[1]])
    assert torch.equal(generate_beam(toy,source,beam_size=1),torch.tensor([[3,3,EOS],[3,3,EOS]]))
    assert torch.equal(generate_beam(toy,source,beam_size=2),torch.tensor([[4,3,EOS],[3,3,EOS]]))
    for alpha in [0.,.6]:
        batched=generate_beam(toy,source,beam_size=2,length_penalty=alpha)
        for index in range(2): assert torch.equal(batched[index:index+1],generate_beam(toy,source[index:index+1],beam_size=2,length_penalty=alpha))
    class Immediate(Toy):
        def decode_step(self,token,t,cache,valid):
            logits=torch.full((len(token),5),-torch.inf); logits[:,EOS]=0
            return logits
    assert generate_beam(Immediate(),source,beam_size=1).eq(EOS).all()
    assert torch.equal(generate_beam(Immediate(),source,beam_size=3),torch.full((2,1),EOS))
    class MixedLength(Toy):
        def decode_step(self,token,t,cache,valid):
            probabilities=[0.,0.,.4,.6,0.] if t==0 else ([0.,0.,.4,0.,.6] if t==1 else [0.,0.,1.,0.,0.])
            return torch.tensor(probabilities).log()[None].expand(len(token),-1)
    # P(EOS)=.4 > P(A,B,EOS)=.36, but length penalty.6 reverses the ranks.
    assert generate_beam(MixedLength(),source,beam_size=2,length_penalty=0.).eq(EOS).all()
    assert torch.equal(generate_beam(MixedLength(),source,beam_size=2,length_penalty=.6),torch.tensor([[3,4,EOS],[3,4,EOS]]))
    # At cap1 a completed EOS is retained and preferred to incomplete A.
    assert generate_beam(MixedLength(),source,beam_size=2,max_new_tokens=1).eq(EOS).all()
    class Tied(Toy):
        near=False
        def decode_step(self,token,t,cache,valid):
            logits=torch.full((len(token),5),-torch.inf); logits[:,3]=0.; logits[:,4]=1e-12 if self.near else 0.
            return logits
    tied=Tied()
    assert generate_beam(tied,source,beam_size=1,max_new_tokens=2).eq(3).all()
    tied.near=True
    assert generate_beam(tied,source,beam_size=1,max_new_tokens=2).eq(4).all()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
    assert torch.equal(generate_beam(model,src,beam_size=1),model.generate(src))
    for width,alpha in [(0,0.),(6,0.),(2,-.1)]:
        try: generate_beam(toy,source,beam_size=width,length_penalty=alpha)
        except ValueError: pass
        else: raise AssertionError('Invalid search settings accepted')
    print('PASS: beam1/greedy; analytic greedy trap; cache parent permutation; source isolation; length penalty; EOS; invalid settings')


if __name__=='__main__': main()
