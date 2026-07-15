"""One parametrized suite runs the same invariants against the PyTorch and
JAX ports (the NumPy original keeps its own test file)."""

import numpy as np
import pytest

rng = np.random.default_rng(0)


def torch_impl():
    import torch
    import online_attention_torch as m
    return m, lambda a: torch.tensor(a, dtype=torch.float64), np.asarray


def jax_impl():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import online_attention_jax as m
    return m, lambda a: jnp.asarray(a, dtype=jnp.float64), np.asarray


@pytest.fixture(params=["torch", "jax"])
def impl(request):
    return torch_impl() if request.param == "torch" else jax_impl()


@pytest.mark.parametrize("block", [3, 16, 1000])
def test_matches_naive_any_block_size(impl, block):
    m, to, back = impl
    q, k, v = (rng.normal(size=s) for s in [(9, 8), (37, 8), (37, 8)])
    got = back(m.attention_tiled(to(q), to(k), to(v), block=block))
    want = back(m.attention_naive(to(q), to(k), to(v)))
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_causal_matches_naive(impl):
    m, to, back = impl
    q, k, v = (rng.normal(size=(20, 16)) for _ in range(3))
    got = back(m.attention_tiled(to(q), to(k), to(v), block=7, causal=True))
    want = back(m.attention_naive(to(q), to(k), to(v), causal=True))
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_numerically_stable_with_huge_logits(impl):
    m, to, back = impl
    q = rng.normal(size=(4, 8)) * 200
    k = rng.normal(size=(50, 8)) * 200
    v = rng.normal(size=(50, 8))
    got = back(m.attention_tiled(to(q), to(k), to(v), block=8))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, back(m.attention_naive(to(q), to(k), to(v))),
                               atol=1e-8)


def test_rescaling_actually_happens(impl):
    m, to, back = impl
    q = np.ones((1, 8))
    k = np.zeros((33, 8)); k[-1] = 10.0
    v = rng.normal(size=(33, 8))
    got = back(m.attention_tiled(to(q), to(k), to(v), block=8))
    want = back(m.attention_naive(to(q), to(k), to(v)))
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_fully_masked_blocks_dont_nan(impl):
    m, to, back = impl
    q, k, v = (rng.normal(size=(6, 4)) for _ in range(3))
    got = back(m.attention_tiled(to(q), to(k), to(v), block=2, causal=True))
    np.testing.assert_allclose(got[0], v[0], atol=1e-12)
