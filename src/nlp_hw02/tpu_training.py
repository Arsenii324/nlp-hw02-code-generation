"""JAX supervised encoder-decoder TPU probe, with PyTorch-shaped checkpoints.

No framework imports until required by the CLI mode. This is a distinct BF16,
fixed-shape, data-parallel recipe; neither TPU throughput nor quality is assumed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

PINNED_CACHE_SHA = '41efd8551f4e9d679a4f06b4e6f86163c15af52c5f6711deacd765b7c06d9a9b'
B012_CONFIG = dict(vocab_size=2048, d_model=384, heads=4, encoder_layers=4,
                   decoder_layers=4, ff=1536, dropout=.1, max_source=256,
                   max_target=192, exclusive_self_attention=False, activation='gelu')


def validate_supported_config(cfg):
    """Reject architecture fields this port cannot faithfully implement.

    Missing prospective fields retain their original single-pass defaults.
    These checks use static config values, never device values or model state.
    """
    if cfg.get('learnable_xsa', False):
        raise ValueError('TPU port does not implement learnable_xsa')
    if cfg.get('activation') not in ('gelu', 'relu2'):
        raise ValueError('TPU port supports activation=gelu or relu2 only')
    if cfg.get('loop_repeats', 1) != 1:
        raise ValueError('TPU port supports loop_repeats=1 only')
    if cfg.get('residual_scale', 1) != 1:
        raise ValueError('TPU port supports residual_scale=1 only')


def initial_parameters(cfg, seed):
    """From scratch; shapes and initialization laws follow existing Seq2Seq.

    NumPy RNG differs from torch RNG; same seed is not same initialization.
    """
    validate_supported_config(cfg)
    rng = np.random.default_rng(seed)
    d = cfg['d_model']
    p = {}
    for name, n in [('embedding', cfg['vocab_size']), ('source_pos', cfg['max_source']),
                    ('target_pos', cfg['max_target'])]:
        p[name+'.weight'] = rng.normal(0, .02, (n, d)).astype(np.float32)
    p['embedding.weight'][0] = 0
    def norm(name):
        p[name+'.weight'] = np.ones(d, np.float32)
        p[name+'.bias'] = np.zeros(d, np.float32)
    def linear(name, ni, no):
        bound = 1/math.sqrt(ni)
        p[name+'.weight'] = rng.uniform(-bound, bound, (no, ni)).astype(np.float32)
        p[name+'.bias'] = rng.uniform(-bound, bound, no).astype(np.float32)
    for stack, count in [('encoder', cfg['encoder_layers']), ('decoder', cfg['decoder_layers'])]:
        for i in range(count):
            name = f'{stack}.{i}'
            for attn in (['self_attn', 'cross'] if stack == 'decoder' else ['self_attn']):
                for proj in ('q', 'k', 'v', 'out'):
                    linear(name+'.'+attn+'.'+proj, d, d)
            for n in (['norm1', 'norm_ff', 'norm_cross'] if stack == 'decoder' else ['norm1', 'norm_ff']):
                norm(name+'.'+n)
            linear(name+'.ff.0', d, cfg['ff'])
            linear(name+'.ff.3', cfg['ff'], d)
    norm('enc_norm'); norm('dec_norm')
    return p


def forward(p, src, tgt, cfg, key, training=True, compute_dtype=None):
    validate_supported_config(cfg)
    import jax
    import jax.numpy as jnp
    dt = compute_dtype or jnp.bfloat16
    drop = cfg['dropout'] if training else 0.
    d, heads = cfg['d_model'], cfg['heads']
    def dropout(x):
        nonlocal key
        if not drop:
            return x
        key, use = jax.random.split(key)
        return jnp.where(jax.random.bernoulli(use, 1-drop, x.shape), x/(1-drop), 0)
    def linear(x, name):
        y = jnp.matmul(x.astype(dt), p[name+'.weight'].T.astype(dt))
        return (y + p[name+'.bias'].astype(dt)).astype(dt)
    def norm(x, name):
        x = x.astype(jnp.float32)
        mean = x.mean(-1, keepdims=True)
        var = ((x-mean)**2).mean(-1, keepdims=True)
        return (x-mean)*jax.lax.rsqrt(var+1e-5)*p[name+'.weight']+p[name+'.bias']
    def embedding(ids, pos):
        e = p['embedding.weight'][ids]
        # padding_idx suppresses input-gather gradient, while tied output retains
        # its gradient for PAD's row (the existing torch behavior).
        e = jnp.where((ids == 0)[..., None], jax.lax.stop_gradient(e), e)
        return dropout(e*math.sqrt(d)+p[pos+'.weight'][jnp.arange(ids.shape[1])])
    def attention(x, context, name, valid=None, causal=False, exclusive=False):
        def split(y):
            return y.reshape(y.shape[0], y.shape[1], heads, d//heads).transpose(0, 2, 1, 3)
        q, k, v = split(linear(x, name+'.q')), split(linear(context, name+'.k')), split(linear(context, name+'.v'))
        score = jnp.einsum('bhqd,bhkd->bhqk', q, k, preferred_element_type=jnp.float32)/math.sqrt(d//heads)
        if valid is not None:
            score = jnp.where(valid[:, None, None, :], score, -jnp.inf)
        if causal:
            mask = jnp.arange(q.shape[2])[:, None] >= jnp.arange(k.shape[2])[None, :]
            score = jnp.where(mask, score, -jnp.inf)
        prob = dropout(jax.nn.softmax(score.astype(jnp.float32), axis=-1)).astype(dt)
        y = jnp.einsum('bhqk,bhkd->bhqd', prob, v).astype(dt)
        if exclusive:
            vf = v.astype(jnp.float32)
            # Clamp squared norm before sqrt: reverse-mode sqrt(0) otherwise
            # produces NaN gradients even when a later maximum masks its value.
            unit = vf/jnp.sqrt(jnp.maximum((vf*vf).sum(-1, keepdims=True), 1e-12))
            yf = y.astype(jnp.float32)
            y = (yf-(yf*unit).sum(-1, keepdims=True)*unit).astype(dt)
        return linear(y.transpose(0, 2, 1, 3).reshape(x.shape), name+'.out')
    def ff(x, name):
        y = linear(norm(x, name+'.norm_ff'), name+'.ff.0')
        y = jax.nn.gelu(y, approximate=False) if cfg['activation'] == 'gelu' else jax.nn.relu(y)**2
        return dropout(linear(dropout(y), name+'.ff.3'))
    valid = src != 0
    memory = embedding(src, 'source_pos')
    for i in range(cfg['encoder_layers']):
        n = f'encoder.{i}'
        z = norm(memory, n+'.norm1')
        memory = memory+dropout(attention(z, z, n+'.self_attn', valid,
                                         exclusive=cfg['exclusive_self_attention']))
        memory = memory+ff(memory, n)
    memory = norm(memory, 'enc_norm')
    x = embedding(tgt, 'target_pos')
    for i in range(cfg['decoder_layers']):
        n = f'decoder.{i}'
        z = norm(x, n+'.norm1')
        x = x+dropout(attention(z, z, n+'.self_attn', causal=True,
                               exclusive=cfg['exclusive_self_attention']))
        x = x+dropout(attention(norm(x, n+'.norm_cross'), memory, n+'.cross', valid))
        x = x+ff(x, n)
    return jnp.matmul(norm(x, 'dec_norm').astype(dt), p['embedding.weight'].T.astype(dt)).astype(jnp.float32)


def token_loss(p, src, target, cfg, key, training=True, compute_dtype=None):
    import jax
    import jax.numpy as jnp
    dec = jnp.concatenate([jnp.ones((target.shape[0], 1), jnp.int32), target[:, :-1]], axis=1)
    logits = forward(p, src, dec, cfg, key, training, compute_dtype)
    ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), target[..., None], -1)[..., 0]
    tokens = (target != 0).sum()
    loss = jnp.where(target != 0, ce, 0).sum()/jnp.maximum(tokens, 1)
    return loss, tokens


def is_muon(name, value, kind):
    return kind == 'muon' and value.ndim == 2 and name.startswith(('encoder.', 'decoder.'))


def init_optimizer(p, kind):
    import jax.numpy as jnp
    return {'count': jnp.asarray(0, jnp.int32),
            'm': {n: jnp.zeros_like(v) for n, v in p.items()},
            'v': {n: jnp.zeros_like(v) for n, v in p.items() if not is_muon(n, v, kind)}}


def optimizer_update(p, state, grads, step, loss_finite, kind='muon', ns_dtype=None):
    import jax
    import jax.numpy as jnp
    ns_dtype = ns_dtype or jnp.bfloat16
    gn = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32)**2) for g in grads.values()))
    finite = loss_finite & jnp.isfinite(gn)
    grads = {n: g*jnp.minimum(1., 1./(gn+1e-6)) for n, g in grads.items()}
    sf = step.astype(jnp.float32)
    lr = 6e-4*jnp.minimum(sf/200., jnp.sqrt(200./jnp.maximum(sf, 200.)))
    count = state['count']+1
    m, v, updates, groups = {}, {}, {}, {}
    for n, value in p.items():
        g = grads[n]
        if is_muon(n, value, kind):
            m[n] = .95*state['m'][n]+.05*g
            u = .05*g+.95*m[n]
            groups.setdefault(value.shape, []).append((n, u))
        else:
            m[n] = .9*state['m'][n]+.1*g
            v[n] = .98*state['v'][n]+.02*g*g
            mh = m[n]/(1-jnp.power(.9, count))
            vh = v[n]/(1-jnp.power(.98, count))
            updates[n] = mh/(jnp.sqrt(vh)+1e-8)
    for shape, items in groups.items():
        x = jnp.stack([u for _, u in items])
        x = x/jnp.maximum(jnp.sqrt((x*x).sum((-2, -1), keepdims=True)), 1e-7)
        tall = shape[0] > shape[1]
        if tall:
            x = x.transpose(0, 2, 1)
        x = x.astype(ns_dtype)
        for _ in range(5):
            a = jnp.matmul(x, x.transpose(0, 2, 1))
            b = -4.775*a+2.0315*jnp.matmul(a, a)
            x = 3.4445*x+jnp.matmul(b, x)
        if tall:
            x = x.transpose(0, 2, 1)
        x = x.astype(jnp.float32)*(.2*math.sqrt(max(shape)))
        for i, (n, _) in enumerate(items):
            updates[n] = x[i]
    candidate = {n: value*(1-.01*lr*(5 if is_muon(n, value, kind) else 1))
                    -lr*(5 if is_muon(n, value, kind) else 1)*updates[n]
                 for n, value in p.items()}
    finite = finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in candidate.values()]))
    selected = jax.tree.map(lambda new, old: jnp.where(finite, new, old), candidate, p)
    newstate = {'count': count, 'm': m, 'v': v}
    newstate = jax.tree.map(lambda new, old: jnp.where(finite, new, old), newstate, state)
    rms = jnp.sqrt(jnp.mean((selected['encoder.0.self_attn.q.weight']-p['encoder.0.self_attn.q.weight'])**2))
    return selected, newstate, {'finite': finite, 'grad_norm': gn, 'lr': lr, 'update_rms': rms}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def make_compiled_chunk(cfg, kind, replicated, batched):
    """One executable shape; schedule/step/RNG are device values, never static."""
    validate_supported_config(cfg)
    import jax
    import jax.numpy as jnp
    traces = [0]
    def chunk(p, opt, offset, key, aa, bb):
        traces[0] += 1
        def one(carry, batch):
            p, opt, step, key = carry
            key, use = jax.random.split(key)
            (loss, tokens), grads = jax.value_and_grad(token_loss, has_aux=True)(p, batch[0], batch[1], cfg, use)
            p, opt, met = optimizer_update(p, opt, grads, step+1, jnp.isfinite(loss), kind)
            return (p, opt, step+1, key), {**met, 'loss': loss, 'tokens': tokens}
        return jax.lax.scan(one, (p, opt, offset, key), (aa, bb))
    compiled = jax.jit(chunk, in_shardings=(replicated, replicated, replicated, replicated, batched, batched),
                       out_shardings=((replicated, replicated, replicated, replicated), replicated), donate_argnums=(0, 1))
    return compiled, traces


def save_checkpoint(output, p, cfg, step, metadata):
    validate_supported_config(cfg)
    import jax
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    arrays = {n: np.asarray(v, dtype=np.float32) for n, v in jax.device_get(p).items()}
    with (output/'weights.npz.tmp').open('wb') as f:
        np.savez(f, **arrays)
    (output/'weights.npz.tmp').replace(output/'weights.npz')
    atomic_json(output/'checkpoint.json', {'config': cfg, 'step': step, 'metadata': metadata,
                                         'weights_sha256': sha(output/'weights.npz')})


def convert_checkpoint(root, output, verify=True):
    """No transpose or renaming; strict load and FP32 forward parity required."""
    import torch
    from .model import ModelConfig, Seq2Seq
    root = Path(root)
    meta = json.loads((root/'checkpoint.json').read_text())
    validate_supported_config(meta['config'])
    if sha(root/'weights.npz') != meta['weights_sha256']:
        raise RuntimeError('NPZ hash mismatch')
    with np.load(root/'weights.npz', allow_pickle=False) as f:
        arrays = {n: np.array(f[n], dtype=np.float32) for n in f.files}
    model = Seq2Seq(ModelConfig(**meta['config'])).eval()
    model.load_state_dict({n: torch.from_numpy(v) for n, v in arrays.items()}, strict=True)
    result = {'strict_load': True, 'step': meta['step'], 'tpu_probe_only': True}
    if verify:
        import jax
        import jax.numpy as jnp
        cfg = meta['config']; rng = np.random.default_rng(867)
        src = rng.integers(4, cfg['vocab_size'], (2, min(8, cfg['max_source'])), dtype=np.int32)
        src[:, -2:] = 0
        dec = rng.integers(4, cfg['vocab_size'], (2, min(6, cfg['max_target'])), dtype=np.int32)
        dec[:, 0] = 1
        jf = forward(jax.tree.map(jnp.asarray, arrays), jnp.asarray(src), jnp.asarray(dec), cfg,
                     jax.random.PRNGKey(0), False, jnp.float32)
        with torch.inference_mode():
            tf = model(torch.from_numpy(src).long(), torch.from_numpy(dec).long()).numpy()
        np.testing.assert_allclose(np.asarray(jf), tf, rtol=5e-4, atol=2e-4)
        result['fp32_logits_max_abs_error'] = float(np.max(np.abs(np.asarray(jf)-tf)))
    tmp = Path(str(output)+'.tmp')
    torch.save({'config': meta['config'], 'model': model.state_dict(), 'step': meta['step'],
                'tpu_probe_only': True, 'conversion': result}, tmp)
    tmp.replace(output)
    return result


def run_probe(args):
    started = time.monotonic()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    def emit(event, **record):
        value = {'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'script_seconds': time.monotonic()-started, **record}
        with (output/'train.jsonl').open('a') as f:
            f.write(json.dumps(value, allow_nan=False)+'\n')
        print(json.dumps(value, allow_nan=False), flush=True)
        atomic_json(output/'progress.json', value)
    emit('startup')
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import jaxlib
    devices = jax.devices()
    if jax.default_backend() != 'tpu' or jax.process_count() != 1 or jax.local_device_count() != len(devices):
        raise RuntimeError('Requires actual TPU backend and all devices on one host/process')
    if len(devices) < 2 or args.batch % len(devices):
        raise RuntimeError('Need >=2 devices and global batch divisible by actual device count')
    if not 20 <= args.steps <= 200 or args.steps % args.chunk or args.batch != 64 or not 1 <= args.chunk <= 20:
        raise ValueError('Probe: batch64, 20..200 steps divisible by chunk1..20')
    cfg = dict(B012_CONFIG); cfg['exclusive_self_attention'] = args.xsa
    emit('hardware', jax=jax.__version__, jaxlib=jaxlib.__version__,
         devices=[{'id': d.id, 'platform': d.platform, 'kind': d.device_kind,
                   'process_index': d.process_index} for d in devices],
         process_count=jax.process_count(), global_batch=args.batch,
         per_device_batch=args.batch//len(devices), image_commit=(Path('/etc/git_commit').read_text().strip()
                                                                if Path('/etc/git_commit').exists() else None))
    root = Path(args.cache)
    manifest = root/'cache_manifest.json'
    if sha(manifest) != PINNED_CACHE_SHA:
        raise RuntimeError('Unpinned cache manifest')
    cm = json.loads(manifest.read_text())
    if not cm['complete']:
        raise RuntimeError('Incomplete cache')
    for name, digest in cm['checksums'].items():
        if name.endswith('.json') and sha(root/name) != digest:
            raise RuntimeError('Cache metadata checksum mismatch: '+name)
    # Inspect headers; immutable CPU-v2 hashes already covered full arrays.
    arrays = [np.load(root/name, mmap_mode='r', allow_pickle=False)
              for name in ('train_source.npy', 'train_target.npy')]
    for arr, shape in zip(arrays, [(2_000_000, 256), (2_000_000, 192)]):
        if arr.shape != shape or arr.dtype != np.uint16 or not arr.flags.c_contiguous:
            raise RuntimeError('Unexpected cache array header')
    mesh = Mesh(np.asarray(devices), ('data',))
    replicated = NamedSharding(mesh, P())
    batched = NamedSharding(mesh, P(None, 'data', None))
    emit('initializing', config=cfg, optimizer=args.optimizer, precision='BF16 products; FP32 master/state/LN/softmax/CE/NS norms',
         cache_manifest_sha256=sha(manifest), resident_arrays=False)
    params = jax.device_put(initial_parameters(cfg, args.seed), replicated)
    state = jax.device_put(init_optimizer(params, args.optimizer), replicated)
    compiled, traces = make_compiled_chunk(cfg, args.optimizer, replicated, batched)
    rng = np.random.default_rng(args.seed)
    def batch():
        idx = rng.integers(0, len(arrays[0]), size=(args.chunk, args.batch))
        data = [np.asarray(a[idx], dtype=np.int32) for a in arrays]
        if any(np.max(a) >= cfg['vocab_size'] for a in data) or np.any((data[0] != 0).sum(-1) == 0):
            raise RuntimeError('Invalid cache token IDs/empty source')
        return [jax.device_put(a, batched) for a in data]
    offset = jax.device_put(jnp.asarray(0, jnp.int32), replicated)
    key = jax.device_put(jax.random.PRNGKey(args.seed), replicated)
    first_batch = batch()
    emit('compile_start', trace_count=traces[0], source_shape=list(first_batch[0].shape), target_shape=list(first_batch[1].shape),
         shards=[{'device': str(s.device), 'index': str(s.index)} for s in first_batch[0].addressable_shards])
    warm_start = time.monotonic()
    lowered = compiled.lower(params, state, offset, key, *first_batch)
    lowered_at = time.monotonic()
    executable = lowered.compile()
    compiled_at = time.monotonic()
    carry, met = executable(params, state, offset, key, *first_batch)
    jax.block_until_ready((carry, met))
    warm_seconds = time.monotonic()-warm_start
    def record_metrics(met):
        host = jax.device_get(met)
        if not np.all(host['finite']) or not all(np.all(np.isfinite(v)) for v in host.values()):
            raise RuntimeError('Nonfinite loss/gradient/update; abort before more quota use')
        return {'loss_mean': float(np.mean(host['loss'])), 'grad_norm_max': float(np.max(host['grad_norm'])),
                'update_rms_min': float(np.min(host['update_rms'])), 'finite_updates': int(np.sum(host['finite'])),
                'nonpad_target_tokens': int(np.sum(host['tokens'])), 'lr_last': float(host['lr'][-1])}
    summary = record_metrics(met)
    emit('compile_complete', compile_and_first_chunk_seconds=warm_seconds,
         lowering_seconds=lowered_at-warm_start, compile_seconds=compiled_at-lowered_at,
         first_chunk_execution_seconds=time.monotonic()-compiled_at, step=args.chunk,
         trace_count=traces[0], **summary)
    records = []; measured_start = time.monotonic()
    for done in range(args.chunk, args.steps, args.chunk):
        if time.monotonic()-started > args.seconds:
            raise RuntimeError('Training script wall budget exhausted')
        chunk_start = time.monotonic(); aa, bb = batch()
        carry, met = executable(*carry, aa, bb)
        jax.block_until_ready((carry, met))
        seconds = time.monotonic()-chunk_start
        summary = record_metrics(met)
        records.append({'seconds': seconds, **summary})
        emit('training_chunk', step=done+args.chunk, synchronized_chunk_seconds=seconds,
             examples_per_second=args.chunk*args.batch/seconds, trace_count=traces[0], **summary)
    measured_seconds = time.monotonic()-measured_start
    if traces[0] != 1:
        raise RuntimeError('Unexpected recompilation')
    emit('checkpoint_start', step=args.steps)
    p = carry[0]
    # Compare from-scratch initial weights only at final checkpoint, no per-step host sync.
    initial = initial_parameters(cfg, args.seed)
    host_p = jax.device_get(p)
    change = float(np.sqrt(np.mean((np.asarray(host_p['encoder.0.self_attn.q.weight'])-initial['encoder.0.self_attn.q.weight'])**2)))
    if not np.isfinite(change) or change <= 0:
        raise RuntimeError('Weights did not change finitely')
    report = {'complete': True, 'probe_only': True, 'steps': args.steps, 'warm_steps': args.chunk,
              'measured_steps': args.steps-args.chunk, 'compile_and_first_chunk_seconds': warm_seconds,
              'lowering_seconds': lowered_at-warm_start, 'compile_seconds': compiled_at-lowered_at,
              'measured_seconds_including_host_input_logging': measured_seconds,
              'examples_per_second': (args.steps-args.chunk)*args.batch/measured_seconds,
              'synchronized_chunk_seconds': [r['seconds'] for r in records],
              'nonpad_tokens_per_second': sum(r['nonpad_target_tokens'] for r in records)/measured_seconds,
              'all_updates_finite': True, 'weight_change_rms': change, 'trace_count': traces[0],
              'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'devices': [str(d) for d in devices],
              'config': cfg, 'optimizer': args.optimizer, 'cache_manifest_sha256': PINNED_CACHE_SHA,
              'recipe_differs_from_b012': ['BF16 compute/NS', 'NumPy initialization/RNG', 'fixed padding',
                                          'data parallel global64', 'compiled updates'],
              'optimizer_state_saved': False, 't4_inference_verified': False}
    save_checkpoint(output, p, cfg, args.steps, report)
    for name in ('tokenizer.json', 'data_stats.json', 'validation.json'):
        import shutil
        shutil.copyfile(root/name, output/name)
    atomic_json(output/'probe_result.json', report)
    emit('finished', step=args.steps, weight_change_rms=change, trace_count=traces[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    probe = sub.add_parser('probe')
    probe.add_argument('--cache', required=True); probe.add_argument('--output', required=True)
    probe.add_argument('--steps', type=int, default=120); probe.add_argument('--chunk', type=int, default=10)
    probe.add_argument('--batch', type=int, default=64); probe.add_argument('--seconds', type=int, default=440)
    probe.add_argument('--seed', type=int, default=42); probe.add_argument('--xsa', action='store_true')
    probe.add_argument('--optimizer', choices=['muon', 'adamw'], default='muon')
    convert = sub.add_parser('convert'); convert.add_argument('--root', required=True)
    convert.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.mode == 'probe':
        run_probe(args)
    else:
        print(json.dumps(convert_checkpoint(args.root, args.output)), flush=True)


if __name__ == '__main__':
    main()
