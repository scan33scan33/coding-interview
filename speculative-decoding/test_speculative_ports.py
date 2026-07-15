"""Parametrized suite for the PyTorch and JAX speculative-decoding ports."""

import numpy as np
import pytest

V = 5


def markov_table(seed):
    g = np.random.default_rng(seed)
    return g.dirichlet(np.ones(V) * 0.8, size=V + 1)  # row V = empty prefix


class TorchImpl:
    def __init__(self):
        import torch
        import speculative_torch as m
        self.m, self.torch = m, torch

    def model(self, table):
        t = self.torch.tensor(table)
        return lambda prefix: t[prefix[-1] if prefix else V]

    def step(self, target, draft, prefix, gamma, seed):
        gen = self.torch.Generator().manual_seed(seed)
        return self.m.speculative_step(target, draft, prefix, gamma, gen)

    def gen(self, target, draft, prefix, n, gamma, seed):
        g = self.torch.Generator().manual_seed(seed)
        return self.m.generate(target, draft, prefix, n, gamma, g)


class JaxImpl:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        import speculative_jax as m
        self.m, self.jax, self.jnp = m, jax, jnp

    def model(self, table):
        t = self.jnp.asarray(table)
        return lambda prefix: t[prefix[-1] if prefix else V]

    def step(self, target, draft, prefix, gamma, seed):
        return self.m.speculative_step(target, draft, prefix, gamma,
                                       self.jax.random.key(seed))

    def gen(self, target, draft, prefix, n, gamma, seed):
        return self.m.generate(target, draft, prefix, n, gamma,
                               self.jax.random.key(seed))


@pytest.fixture(params=["torch", "jax"], scope="module")
def impl(request):
    return TorchImpl() if request.param == "torch" else JaxImpl()


def test_exactness_with_bad_draft(impl):
    # The theorem: first-token distribution == target's, unrelated draft.
    target_t, draft_t = markov_table(1), markov_table(2)
    target, draft = impl.model(target_t), impl.model(draft_t)
    counts = np.zeros(V)
    n = 8000
    for trial in range(n):
        toks, _ = impl.step(target, draft, [], 3, seed=trial)
        counts[toks[0]] += 1
    np.testing.assert_allclose(counts / n, target_t[V], atol=0.03)


def test_perfect_draft_accepts_everything(impl):
    target = impl.model(markov_table(1))
    _, stats = impl.step(target, target, [], 4, seed=0)
    assert stats["accepted"] == 4


def test_every_round_emits_at_least_one_token(impl):
    target = impl.model(markov_table(1))
    draft = impl.model(markov_table(2))
    for trial in range(100):
        toks, stats = impl.step(target, draft, [2], 3, seed=trial)
        assert 1 <= len(toks) <= 4
        assert len(toks) == stats["accepted"] + 1


def test_acceptance_tracks_draft_quality(impl):
    target = impl.model(markov_table(1))
    same = impl.model(markov_table(1))
    diff = impl.model(markov_table(9))
    _, rate_same = impl.gen(target, same, [], 200, 3, seed=0)
    _, rate_diff = impl.gen(target, diff, [], 200, 3, seed=0)
    assert rate_same == 1.0
    assert rate_diff < rate_same
