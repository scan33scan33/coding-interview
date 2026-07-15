"""Parametrized suite for the PyTorch and JAX ring all-reduce ports."""

import numpy as np
import pytest

from allreduce import naive_allreduce_volume, ring_volume_per_worker

rng = np.random.default_rng(0)


@pytest.fixture(params=["torch", "jax"])
def impl(request):
    if request.param == "torch":
        import torch
        import allreduce_torch as m
        return m, lambda a: torch.tensor(a)
    import allreduce_jax as m
    import jax.numpy as jnp
    return m, jnp.asarray


@pytest.mark.parametrize("p,n", [(2, 10), (4, 64), (5, 17), (8, 8)])
def test_every_worker_gets_the_sum(impl, p, n):
    m, to = impl
    grads = [rng.normal(size=n).astype(np.float32) for _ in range(p)]
    want = np.sum(grads, axis=0)
    results, _ = m.ring_allreduce([to(g) for g in grads])
    for r in results:
        np.testing.assert_allclose(np.asarray(r), want, atol=1e-4)


def test_communication_volume_is_bandwidth_optimal(impl):
    m, to = impl
    p, n = 8, 800
    grads = [rng.normal(size=n).astype(np.float32) for _ in range(p)]
    _, log = m.ring_allreduce([to(g) for g in grads])
    assert len(log) == 2 * (p - 1) * p
    sent = ring_volume_per_worker(log, p)
    expected = 2 * n * (p - 1) / p
    for s in sent:
        assert abs(s - expected) <= p
    assert max(sent) < naive_allreduce_volume(p, n) / 3


def test_only_neighbor_communication(impl):
    m, to = impl
    p = 5
    grads = [rng.normal(size=20).astype(np.float32) for _ in range(p)]
    _, log = m.ring_allreduce([to(g) for g in grads])
    for _, _, src, dst, _ in log:
        assert dst == (src + 1) % p
