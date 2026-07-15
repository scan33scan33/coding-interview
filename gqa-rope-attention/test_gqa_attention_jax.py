import jax
import jax.numpy as jnp
import numpy as np

from gqa_attention_jax import apply_rope, gqa_forward, init_gqa, rope_tables


def test_rope_preserves_norm():
    cos, sin = rope_tables(8, 32)
    x = jax.random.normal(jax.random.key(0), (2, 3, 16, 8))
    y = apply_rope(x, cos, sin)
    np.testing.assert_allclose(jnp.linalg.norm(x, axis=-1),
                               jnp.linalg.norm(y, axis=-1), atol=1e-5)


def test_rope_is_relative():
    cos, sin = rope_tables(16, 64)
    kq, kk = jax.random.split(jax.random.key(1))
    q = jax.random.normal(kq, (1, 1, 1, 16))
    k = jax.random.normal(kk, (1, 1, 1, 16))
    dots = []
    for shift in [0, 5, 23]:
        qr = apply_rope(q, cos, sin, offset=3 + shift)
        kr = apply_rope(k, cos, sin, offset=7 + shift)
        dots.append(float((qr * kr).sum()))
    assert abs(dots[0] - dots[1]) < 1e-4
    assert abs(dots[0] - dots[2]) < 1e-4


def test_causal_masking():
    params = init_gqa(jax.random.key(0), dim=32, n_heads=4, n_kv_heads=2)
    x = jax.random.normal(jax.random.key(2), (1, 8, 32))
    out_a, _ = gqa_forward(params, x)
    x2 = x.at[0, 5].add(10.0)
    out_b, _ = gqa_forward(params, x2)
    np.testing.assert_allclose(out_a[0, :5], out_b[0, :5], atol=1e-5)
    assert not np.allclose(out_a[0, 5:], out_b[0, 5:], atol=1e-5)


def test_kv_cache_matches_full_forward():
    params = init_gqa(jax.random.key(0), dim=48, n_heads=6, n_kv_heads=2)
    x = jax.random.normal(jax.random.key(3), (2, 12, 48))
    full, _ = gqa_forward(params, x)

    cache, steps = None, []
    for t in range(x.shape[1]):
        out, cache = gqa_forward(params, x[:, t:t + 1], cache)
        steps.append(out)
    inc = jnp.concatenate(steps, axis=1)
    np.testing.assert_allclose(full, inc, atol=1e-4)


def test_cache_shapes_use_kv_heads():
    params = init_gqa(jax.random.key(0), dim=64, n_heads=8, n_kv_heads=2)
    _, (k, v) = gqa_forward(params, jax.random.normal(jax.random.key(4), (1, 5, 64)))
    assert k.shape == (1, 2, 5, 8)
    assert v.shape == (1, 2, 5, 8)
