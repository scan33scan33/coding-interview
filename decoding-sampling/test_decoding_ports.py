"""Parametrized suite running the same invariants against the PyTorch and
JAX ports (the NumPy original keeps its own tests)."""

import numpy as np
import pytest


class TorchImpl:
    name = "torch"

    def __init__(self):
        import torch
        import decoding_torch as m
        self.m, self.torch = m, torch
        self.gen = torch.Generator().manual_seed(0)

    def arr(self, x):
        return self.torch.tensor(np.asarray(x, dtype=np.float64))

    def finite(self, x):
        return np.isfinite(np.asarray(x))

    def sample(self, logits, **kw):
        return self.m.sample_token(self.arr(logits), self.gen, **kw)


class JaxImpl:
    name = "jax"

    def __init__(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import decoding_jax as m
        self.m, self.jax, self.jnp = m, jax, jnp
        self._key = jax.random.key(0)

    def arr(self, x):
        return self.jnp.asarray(x, dtype=self.jnp.float64)

    def finite(self, x):
        return np.isfinite(np.asarray(x))

    def sample(self, logits, **kw):
        self._key, sub = self.jax.random.split(self._key)
        return self.m.sample_token(self.arr(logits), sub, **kw)


@pytest.fixture(params=["torch", "jax"])
def impl(request):
    return TorchImpl() if request.param == "torch" else JaxImpl()


def test_top_k_keeps_exactly_k(impl):
    logits = impl.arr([5.0, 1.0, 3.0, 4.0, 2.0])
    out = impl.m.top_k_filter(logits, 2)
    assert impl.finite(out).sum() == 2
    assert impl.finite(out)[[0, 3]].all()
    assert impl.finite(impl.m.top_k_filter(logits, 10)).all()


def test_top_p_boundary_semantics(impl):
    logits = impl.arr(np.log([0.5, 0.25, 0.125, 0.125]))
    assert impl.finite(impl.m.top_p_filter(logits, 0.7)).sum() == 2
    assert impl.finite(impl.m.top_p_filter(logits, 0.5)).sum() == 1
    assert impl.finite(impl.m.top_p_filter(logits, 0.51)).sum() == 2
    assert impl.finite(impl.m.top_p_filter(logits, 1e-9)).sum() == 1


def test_repetition_penalty_reduces_prob_for_any_sign(impl):
    logits = impl.arr([2.0, -2.0, 0.5])
    out = impl.m.apply_repetition_penalty(logits, [0, 1], 1.5)
    out_np, in_np = np.asarray(out), np.asarray(logits)

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    before, after = softmax(in_np), softmax(out_np)
    assert after[0] < before[0]
    assert after[1] < before[1]


def test_sampling_matches_softmax_distribution(impl):
    logits = np.array([0.0, 1.0, 2.0])
    want = np.exp(logits) / np.exp(logits).sum()
    draws = [impl.sample(logits) for _ in range(8000)]
    got = np.bincount(draws, minlength=3) / len(draws)
    np.testing.assert_allclose(got, want, atol=0.03)


def make_trap_model(impl):
    table = {
        (): [np.log(0.6), np.log(0.4), -np.inf],
        (0,): [np.log(0.5), np.log(0.49), np.log(0.01)],
        (1,): [-np.inf, -np.inf, 0.0],
    }

    def step_fn(seq):
        key = tuple(seq[1:])[:1] if len(seq) > 1 else ()
        return impl.arr(table.get(key, table[(0,)]))
    return step_fn


def test_beam_search_escapes_greedy_trap(impl):
    step_fn = make_trap_model(impl)
    greedy = impl.m.greedy_decode(step_fn, bos=3, eos=2, max_len=5)
    assert greedy[1] == 0
    best, _ = impl.m.beam_search(step_fn, bos=3, eos=2, beam_width=2, max_len=5)
    assert best == [3, 1, 2]
