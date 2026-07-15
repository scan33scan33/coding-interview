"""Parametrized suite for the PyTorch and JAX k-means ports."""

import numpy as np
import pytest


def make_blobs(k=4, per=50, spread=0.3, sep=10.0, d=2, seed=1):
    g = np.random.default_rng(seed)
    centers = g.normal(size=(k, d)) * sep
    x = np.concatenate([c + g.normal(size=(per, d)) * spread for c in centers])
    labels = np.repeat(np.arange(k), per)
    return x.astype(np.float32), labels


class TorchImpl:
    def __init__(self):
        import torch
        import kmeans_torch as m
        self.m, self.torch = m, torch

    def run(self, x, k, seed=0, **kw):
        g = self.torch.Generator().manual_seed(seed)
        c, l, h = self.m.kmeans(self.torch.tensor(x), k, g, **kw)
        return np.asarray(c), np.asarray(l), h


class JaxImpl:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        import kmeans_jax as m
        self.m, self.jax, self.jnp = m, jax, jnp

    def run(self, x, k, seed=0, **kw):
        c, l, h = self.m.kmeans(self.jnp.asarray(x), k, self.jax.random.key(seed), **kw)
        return np.asarray(c), np.asarray(l), h


@pytest.fixture(params=["torch", "jax"])
def impl(request):
    return TorchImpl() if request.param == "torch" else JaxImpl()


def test_inertia_monotonically_decreases(impl):
    x, _ = make_blobs()
    _, _, history = impl.run(x, 4)
    assert all(a >= b - 1e-3 for a, b in zip(history, history[1:]))


def test_recovers_separated_blobs(impl):
    x, true_labels = make_blobs(spread=0.2, sep=20.0)
    _, labels, _ = impl.run(x, 4)
    for j in range(4):
        assert len(set(true_labels[labels == j])) == 1


def test_deterministic_given_seed(impl):
    x, _ = make_blobs()
    c1, l1, h1 = impl.run(x, 4, seed=7)
    c2, l2, h2 = impl.run(x, 4, seed=7)
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_allclose(c1, c2, atol=1e-6)


def test_empty_cluster_is_reseeded(impl):
    x = np.array([[0.0, 0], [0.1, 0], [50.0, 0], [50.1, 0], [50.2, 0],
                  [49.9, 0]], dtype=np.float32)
    _, labels, _ = impl.run(x, 3, init="random")
    assert len(set(labels.tolist())) == 3
