"""Pre-LN encoder–decoder with SDPA and cached greedy inference."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .tokenizer import PAD, BOS, EOS


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 192
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    ff: int = 768
    dropout: float = .1
    max_source: int = 256
    max_target: int = 192
    exclusive_self_attention: bool = False
    learnable_xsa: bool = False
    activation: str = 'gelu'
    loop_repeats: int = 1
    share_loop_weights: bool = True
    residual_scale: float = 1.
    loop_order: str = 'stack'
    loop_share_scope: str = 'block'
    gated_repeats: bool = False
    repeat_residual_scale: float = .5

    def __post_init__(self):
        if type(self.loop_repeats) is not int or self.loop_repeats<1:
            raise ValueError('loop_repeats must be a positive integer')
        if not math.isfinite(self.residual_scale) or self.residual_scale<=0:
            raise ValueError('residual_scale must be positive and finite')
        if self.loop_order not in ('stack','adjacent'):
            raise ValueError('loop_order must be stack or adjacent')
        if self.loop_share_scope not in ('block','attention'):
            raise ValueError('loop_share_scope must be block or attention')
        if self.loop_share_scope=='attention' and (not self.share_loop_weights or self.loop_repeats==1):
            raise ValueError('Attention-only sharing requires shared loops with repeats>1')
        if not math.isfinite(self.repeat_residual_scale) or self.repeat_residual_scale<=0:
            raise ValueError('repeat_residual_scale must be positive and finite')
        if self.gated_repeats and self.loop_repeats<2:
            raise ValueError('Gated repeats require loop_repeats>=2')
        if self.gated_repeats and (not self.share_loop_weights or self.loop_share_scope!='attention'):
            raise ValueError('Gated repeats require attention-shared loops with independent block branches')


def residual_add(x, branch, scale):
    return x+branch if scale==1. else x+scale*branch


class SquaredReLU(nn.Module):
    def forward(self,x):
        return F.relu(x).square()


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_gate = nn.Linear(cfg.d_model, 2*cfg.ff)
        self.drop = nn.Dropout(cfg.dropout)
        self.down = nn.Linear(cfg.ff, cfg.d_model)

    def forward(self, x):
        gate, up = self.up_gate(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(gate)*up))


def exclusive_projection(y, own_value, alpha=None):
    """XSA (Zhai 2026): per-head projection before the output projection.

    FP32 normalization/reduction keeps zero vectors and FP16 values finite.
    This precision choice is explicit; encoder use extends the causal-LM paper.
    """
    unit=F.normalize(own_value.float(),dim=-1,eps=1e-6)
    projection=(y.float()*unit).sum(-1,keepdim=True)*unit
    if alpha is not None:
        projection=projection*alpha.float().tanh().view(1,-1,1,1)
    z=y.float()-projection
    return z.to(y.dtype)


class Attention(nn.Module):
    def __init__(self, cfg, exclusive=False):
        super().__init__()
        self.heads, self.dim = cfg.heads, cfg.d_model // cfg.heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        self.exclusive = exclusive
        if cfg.learnable_xsa and cfg.exclusive_self_attention:
            raise ValueError('Choose fixed or learnable XSA, not both')
        # Cross-attention is not XSA. Zero initialization consumes no RNG.
        self.register_parameter('xsa_alpha', nn.Parameter(torch.zeros(cfg.heads))
            if exclusive and cfg.learnable_xsa else None)

    def split(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.dim).transpose(1,2)

    def kv(self, x):
        return self.split(self.k(x)), self.split(self.v(x))

    def attend(self, q, k, v, valid=None, causal=False, own_value=None):
        mask = valid[:,None,None,:] if valid is not None else None
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0, is_causal=causal)
        if self.exclusive:
            z=exclusive_projection(z,v if own_value is None else own_value,self.xsa_alpha)
        return self.out(z.transpose(1,2).contiguous().flatten(2))

    def forward(self, x, context=None, valid=None, causal=False):
        k,v = self.kv(x if context is None else context)
        return self.attend(self.split(self.q(x)), k, v, valid, causal)


class Block(nn.Module):
    def __init__(self, cfg, decoder=False):
        super().__init__()
        self.self_attn = Attention(cfg, exclusive=cfg.exclusive_self_attention or cfg.learnable_xsa)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm_ff = nn.LayerNorm(cfg.d_model)
        if cfg.activation not in ['gelu','relu2','swiglu']: raise ValueError(cfg.activation)
        if cfg.activation=='swiglu':
            self.ff = SwiGLU(cfg)
        else:
            activation=nn.GELU() if cfg.activation=='gelu' else SquaredReLU()
            self.ff = nn.Sequential(nn.Linear(cfg.d_model,cfg.ff), activation, nn.Dropout(cfg.dropout), nn.Linear(cfg.ff,cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.decoder = decoder
        self.residual_scale = cfg.residual_scale
        if decoder:
            self.cross = Attention(cfg)
            self.norm_cross = nn.LayerNorm(cfg.d_model)

    def forward(self, x, memory=None, valid=None):
        x = residual_add(x,self.drop(self.self_attn(self.norm1(x), valid=None if self.decoder else valid, causal=self.decoder)),self.residual_scale)
        if self.decoder:
            x = residual_add(x,self.drop(self.cross(self.norm_cross(x), memory, valid)),self.residual_scale)
        return residual_add(x,self.drop(self.ff(self.norm_ff(x))),self.residual_scale)


class Seq2Seq(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size,cfg.d_model,padding_idx=PAD)
        self.source_pos = nn.Embedding(cfg.max_source,cfg.d_model)
        self.target_pos = nn.Embedding(cfg.max_target,cfg.d_model)
        self.encoder = nn.ModuleList([Block(cfg) for _ in range(cfg.encoder_layers)])
        self.decoder = nn.ModuleList([Block(cfg,True) for _ in range(cfg.decoder_layers)])
        self.encoder_repeat_gates = nn.Parameter(torch.zeros(cfg.encoder_layers*(cfg.loop_repeats-1))) if cfg.gated_repeats else None
        self.decoder_repeat_gates = nn.Parameter(torch.zeros(cfg.decoder_layers*(cfg.loop_repeats-1))) if cfg.gated_repeats else None
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # Scale tied embeddings down initially; no pretrained weights.
        nn.init.normal_(self.embedding.weight, std=.02)
        nn.init.normal_(self.source_pos.weight, std=.02)
        nn.init.normal_(self.target_pos.weight, std=.02)
        with torch.no_grad(): self.embedding.weight[PAD].zero_()
        if cfg.loop_repeats>1 and (not cfg.share_loop_weights or cfg.loop_share_scope=='attention'):
            # Identical initial paths; clones become independent after training.
            encoder_base=tuple(self.encoder);decoder_base=tuple(self.decoder)
            self.encoder.extend(deepcopy(layer) for _ in range(cfg.loop_repeats-1) for layer in encoder_base)
            self.decoder.extend(deepcopy(layer) for _ in range(cfg.loop_repeats-1) for layer in decoder_base)
            if cfg.gated_repeats:
                for layer in self.encoder[cfg.encoder_layers:]: layer.residual_scale=cfg.repeat_residual_scale
                for layer in self.decoder[cfg.decoder_layers:]: layer.residual_scale=cfg.repeat_residual_scale
            if cfg.loop_share_scope=='attention':
                for stack,base in ((self.encoder,encoder_base),(self.decoder,decoder_base)):
                    for index in range(len(base),len(stack)):
                        original=base[index%len(base)]
                        stack[index].self_attn=original.self_attn
                        if stack[index].decoder:stack[index].cross=original.cross

    def _steps(self,stack,base_count):
        if self.cfg.share_loop_weights and self.cfg.loop_share_scope=='block':
            if self.cfg.loop_order=='stack':return list(stack)*self.cfg.loop_repeats
            return [layer for layer in stack for _ in range(self.cfg.loop_repeats)]
        if self.cfg.loop_order=='stack':return stack
        return [stack[repeat*base_count+layer] for layer in range(base_count)
                for repeat in range(self.cfg.loop_repeats)]

    def encoder_steps(self):
        return self._steps(self.encoder,self.cfg.encoder_layers)

    def decoder_steps(self):
        return self._steps(self.decoder,self.cfg.decoder_layers)

    def _application(self,stack,base_count,gate_params):
        if not self.cfg.gated_repeats:
            for layer in self._steps(stack,base_count):
                yield layer,None
            return
        # Storage remains repeat-major; application order may be adjacent.
        indices=(range(len(stack)) if self.cfg.loop_order=='stack' else
                 [repeat*base_count+layer for layer in range(base_count)
                  for repeat in range(self.cfg.loop_repeats)])
        for index in indices:
            repeat,base=divmod(index,base_count)
            gate=gate_params[(repeat-1)*base_count+base] if repeat else None
            yield stack[index],gate

    @staticmethod
    def _gated_result(before,after,gate):
        return after if gate is None else before+gate*(after-before)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if self.cfg.loop_share_scope=='attention':
            if assign:raise ValueError('assign=True can break attention aliases')
            groups={}
            for name,param in self.named_parameters(remove_duplicate=False):
                groups.setdefault(id(param),[]).append(name)
            for names in groups.values():
                present=[name for name in names if name in state_dict]
                if len(present)>1:
                    first=state_dict[present[0]]
                    if any(not torch.equal(first,state_dict[name]) for name in present[1:]):
                        raise ValueError('Inconsistent shared attention checkpoint: '+str(present))
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def embed(self, ids, pos):
        return self.drop(self.embedding(ids)*math.sqrt(self.cfg.d_model) + pos(torch.arange(ids.shape[1],device=ids.device)))

    def encode(self,src):
        valid = src.ne(PAD)
        x = self.embed(src,self.source_pos)
        for layer,gate in self._application(self.encoder,self.cfg.encoder_layers,self.encoder_repeat_gates):
            before=x
            x=self._gated_result(before,layer(x,valid=valid),gate)
        return self.enc_norm(x),valid

    def forward(self,src,tgt):
        memory,valid = self.encode(src)
        x = self.embed(tgt,self.target_pos)
        for layer,gate in self._application(self.decoder,self.cfg.decoder_layers,self.decoder_repeat_gates):
            before=x
            x=self._gated_result(before,layer(x,memory,valid),gate)
        return F.linear(self.dec_norm(x),self.embedding.weight)

    def new_cache(self,memory):
        caches=[]
        for layer,gate in self._application(self.decoder,self.cfg.decoder_layers,self.decoder_repeat_gates):
            cross_k,cross_v = layer.cross.kv(memory)
            shape=(memory.shape[0],self.cfg.heads,self.cfg.max_target,self.cfg.d_model//self.cfg.heads)
            caches.append((torch.empty(shape,device=memory.device,dtype=cross_k.dtype),
                           torch.empty(shape,device=memory.device,dtype=cross_k.dtype),cross_k,cross_v))
        return caches

    def decode_step(self,token,t,cache,valid):
        x = self.embedding(token[:,None])*math.sqrt(self.cfg.d_model) + self.target_pos.weight[t]
        applications=self._application(self.decoder,self.cfg.decoder_layers,self.decoder_repeat_gates)
        for (layer,gate),(sk,sv,ck,cv) in zip(applications,cache):
            before=x
            z=layer.norm1(x)
            k,v=layer.self_attn.kv(z)
            sk[:,:,t:t+1]=k; sv[:,:,t:t+1]=v
            x=residual_add(x,layer.self_attn.attend(layer.self_attn.split(layer.self_attn.q(z)),sk[:,:,:t+1],sv[:,:,:t+1],own_value=v),layer.residual_scale)
            z=layer.norm_cross(x)
            x=residual_add(x,layer.cross.attend(layer.cross.split(layer.cross.q(z)),ck,cv,valid),layer.residual_scale)
            x=residual_add(x,layer.ff(layer.norm_ff(x)),layer.residual_scale)
            x=self._gated_result(before,x,gate)
        return F.linear(self.dec_norm(x[:,0]),self.embedding.weight)

    @torch.inference_mode()
    def generate(self,src,max_new_tokens=None):
        if self.training: raise RuntimeError('Call eval() before cached generation')
        limit=min(max_new_tokens or self.cfg.max_target,self.cfg.max_target)
        memory,valid=self.encode(src)
        cache=self.new_cache(memory)
        token=torch.full((src.shape[0],),BOS,device=src.device,dtype=torch.long)
        done=torch.zeros_like(token,dtype=torch.bool)
        result=[]
        for t in range(limit):
            token=self.decode_step(token,t,cache,valid).argmax(-1)
            token=torch.where(done,torch.full_like(token,EOS),token)
            result.append(token)
            done |= token.eq(EOS)
            if done.all(): break
        return torch.stack(result,1)


def load_model(path,device='cuda'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=Seq2Seq(ModelConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint
