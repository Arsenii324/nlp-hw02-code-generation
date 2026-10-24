"""Focused CPU parity/optimizer checks; no data, installs or cloud calls."""
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import jax
import jax.numpy as jnp
import numpy as np
import torch
from nlp_hw02.model import ModelConfig, Seq2Seq, load_model
from nlp_hw02.tpu_training import (forward, token_loss, init_optimizer,
                                  optimizer_update, initial_parameters,
                                  save_checkpoint, convert_checkpoint, make_compiled_chunk,
                                  validate_supported_config)


def main():
    torch.set_num_threads(1)
    cfg = dict(vocab_size=32, d_model=16, heads=2, encoder_layers=1,
               decoder_layers=1, ff=32, dropout=0., max_source=8, max_target=6,
               exclusive_self_attention=False, activation='gelu')
    src = np.array([[4, 5, 2, 0], [6, 7, 8, 2]], dtype=np.int32)
    target = np.array([[9, 10, 2], [11, 2, 0]], dtype=np.int32)
    dec = np.concatenate([np.ones((2, 1), np.int32), target[:, :-1]], 1)
    records = []
    unsupported = [('learnable_xsa', True), ('activation', 'swiglu'),
                   ('activation', 'unknown'), ('loop_repeats', 2), ('residual_scale', .5)]
    for name, value in unsupported:
        badcfg = {**cfg, name: value}
        # Reject before weights or a compiled executable can be created, and
        # reject direct forward/checkpoint routes as well.
        calls = [lambda: initial_parameters(badcfg, 42),
                 lambda: forward({}, None, None, badcfg, None),
                 lambda: make_compiled_chunk(badcfg, 'muon', None, None)]
        with tempfile.TemporaryDirectory() as td:
            calls.append(lambda: save_checkpoint(Path(td), {}, badcfg, 0, {}))
            Path(td, 'checkpoint.json').write_text(json.dumps({'config': badcfg}))
            calls.append(lambda: convert_checkpoint(Path(td), Path(td)/'model.pt'))
            for call in calls:
                try:
                    call()
                except ValueError as error:
                    assert name in str(error), str(error)
                else:
                    raise AssertionError('Unsupported TPU configuration accepted: '+name)
    for activation in ('gelu', 'relu2'):
        for xsa in (False, True):
            validate_supported_config({**cfg, 'activation': activation,
                'exclusive_self_attention': xsa, 'learnable_xsa': False,
                'loop_repeats': 1, 'residual_scale': 1.})
    records.append({'unsupported_configuration_guards': 'passed',
                    'rejected': unsupported, 'entry_points_per_case': 5,
                    'supported_activation_attention_combinations': 4})
    for xsa in (False, True):
        cfg['exclusive_self_attention'] = xsa
        p = initial_parameters(cfg, 42)
        m = Seq2Seq(ModelConfig(**cfg)).eval()
        m.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in p.items()}, strict=True)
        jp = jax.tree.map(jnp.asarray, p)
        jl = forward(jp, jnp.asarray(src), jnp.asarray(dec), cfg,
                     jax.random.PRNGKey(1), training=False, compute_dtype=jnp.float32)
        tl = m(torch.from_numpy(src).long(), torch.from_numpy(dec).long())
        np.testing.assert_allclose(np.asarray(jl), tl.detach().numpy(), rtol=3e-5, atol=3e-6)
        (loss, _), grads = jax.value_and_grad(token_loss, has_aux=True)(
            jp, jnp.asarray(src), jnp.asarray(target), cfg, jax.random.PRNGKey(1),
            False, jnp.float32)
        tloss = torch.nn.functional.cross_entropy(tl.flatten(0, 1),
                    torch.from_numpy(target).long().flatten(), ignore_index=0)
        tloss.backward()
        np.testing.assert_allclose(np.asarray(loss), tloss.detach().numpy(), rtol=3e-6)
        max_grad = 0.
        for name, param in m.named_parameters():
            np.testing.assert_allclose(np.asarray(grads[name]), param.grad.numpy(), rtol=2e-3, atol=8e-6,
                                       err_msg=name)
            max_grad = max(max_grad, float(np.max(np.abs(np.asarray(grads[name])-param.grad.numpy()))))
        if xsa:
            zero = dict(jp)
            for name in zero:
                if '.self_attn.v.' in name:
                    zero[name] = jnp.zeros_like(zero[name])
            zero_grads = jax.grad(lambda q: token_loss(q, jnp.asarray(src), jnp.asarray(target), cfg,
                                                     jax.random.PRNGKey(1), False, jnp.float32)[0])(zero)
            assert all(np.all(np.isfinite(np.asarray(g))) for g in zero_grads.values()), 'XSA zero-value gradients'
        state = init_optimizer(jp, 'muon')
        updated, newstate, metrics = jax.jit(lambda q, s, g: optimizer_update(
            q, s, g, jnp.asarray(1, jnp.int32), jnp.asarray(True), 'muon', jnp.float32))(jp, state, grads)
        assert bool(metrics['finite']) and float(metrics['update_rms']) > 0
        # FP32 NS lets the existing PyTorch grouped implementation be an oracle.
        from nlp_hw02.optim_batched import BatchedMuon
        matrices = [param for name, param in m.named_parameters()
                    if param.ndim == 2 and name.startswith(('encoder.', 'decoder.'))]
        others = [param for name, param in m.named_parameters()
                  if not (param.ndim == 2 and name.startswith(('encoder.', 'decoder.')))]
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        base = 6e-4/200
        BatchedMuon(matrices, lr=5*base, ns_dtype=torch.float32).step()
        torch.optim.AdamW(others, lr=base, betas=(.9, .98), weight_decay=.01).step()
        for name, param in m.named_parameters():
            np.testing.assert_allclose(np.asarray(updated[name]), param.detach().numpy(), rtol=2e-4, atol=2e-6,
                                       err_msg=name)
        badgrads = dict(grads)
        badgrads['embedding.weight'] = grads['embedding.weight'].at[0, 0].set(jnp.nan)
        skipped, skipstate, skipmetrics = optimizer_update(jp, state, badgrads,
            jnp.asarray(1), jnp.asarray(True), 'muon', jnp.float32)
        assert not bool(skipmetrics['finite'])
        for name in jp:
            np.testing.assert_array_equal(np.asarray(skipped[name]), np.asarray(jp[name]))
        for a, b in zip(jax.tree.leaves(skipstate), jax.tree.leaves(state)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            save_checkpoint(td, updated, cfg, 1, {'probe_only': True})
            converted = convert_checkpoint(td, td/'model.pt', verify=True)
            loaded, checkpoint = load_model(td/'model.pt', device='cpu')
            assert checkpoint['step'] == 1 and converted['strict_load']
            for name, param in loaded.named_parameters():
                np.testing.assert_array_equal(param.detach().numpy(), np.asarray(updated[name]))
        records.append({'xsa': xsa, 'loss': float(loss), 'max_gradient_abs_error': max_grad,
                        'forward_gradient_muon_adamw_skip_conversion': 'passed'})
    # Force2 CPU devices with XLA_FLAGS=--xla_force_host_platform_device_count=2
    # to exercise real sharding/collectives and one executable reused at new steps.
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    rep = NamedSharding(mesh, P()); bat = NamedSharding(mesh, P(None, 'data', None))
    cfg['exclusive_self_attention'] = False; cfg['dropout'] = .1
    jp = jax.device_put(initial_parameters(cfg, 42), rep)
    opt = jax.device_put(init_optimizer(jp, 'muon'), rep)
    aa = jax.device_put(np.stack([src]*2), bat)
    bb = jax.device_put(np.stack([target]*2), bat)
    step = jax.device_put(jnp.asarray(0, jnp.int32), rep)
    key = jax.device_put(jax.random.PRNGKey(1), rep)
    # Unequal token counts (3 vs2) exercise a globally weighted token reduction.
    expected_loss, _ = token_loss(jax.tree.map(jnp.asarray, initial_parameters(cfg, 42)),
                                 jnp.asarray(src), jnp.asarray(target), cfg,
                                 jax.random.split(jax.random.PRNGKey(1))[1])
    chunk, traces = make_compiled_chunk(cfg, 'muon', rep, bat)
    exe = chunk.lower(jp, opt, step, key, aa, bb).compile()
    carry, metrics = exe(jp, opt, step, key, aa, bb)
    jax.block_until_ready((carry, metrics))
    assert np.all(np.asarray(metrics['finite'])) and np.all(np.asarray(metrics['update_rms']) > 0)
    first_loss = np.asarray(metrics['loss']).tolist()
    np.testing.assert_allclose(first_loss[0], float(expected_loss), rtol=2e-4, atol=1e-4)
    carry, metrics = exe(*carry, aa, bb)
    jax.block_until_ready((carry, metrics))
    assert int(carry[2]) == 4 and traces[0] == 1 and np.all(np.asarray(metrics['finite']))
    assert len(aa.addressable_shards) == len(jax.devices())
    records.append({'bf16_compiled_chunks': 'passed', 'trace_count': traces[0],
                    'steps': 4, 'data_shards': len(aa.addressable_shards), 'first_loss': first_loss})
    print(json.dumps({'event': 'local_parity_checks', 'jax': jax.__version__,
                      'torch': torch.__version__, 'devices': [str(d) for d in jax.devices()],
                      'records': records}))


if __name__ == '__main__':
    main()
