"""Parametrized suite for the PyTorch and JAX IVF ports."""

import numpy as np
import pytest

rng = np.random.default_rng(0)


def make_corpus(n=2000, d=32, n_clusters=20, seed=1):
    g = np.random.default_rng(seed)
    centers = g.normal(size=(n_clusters, d)) * 5
    x = centers[g.integers(0, n_clusters, n)] + g.normal(size=(n, d)) * 0.6
    return x.astype(np.float32)


class TorchImpl:
    def __init__(self):
        import torch
        import ivf_torch as m
        self.m, self.to = m, lambda a: torch.tensor(a)

    def norm(self, x):
        return np.asarray(x)


class JaxImpl:
    def __init__(self):
        import jax.numpy as jnp
        import ivf_jax as m
        self.m, self.to = m, jnp.asarray

    def norm(self, x):
        return np.asarray(x)


@pytest.fixture(params=["torch", "jax"], scope="module")
def impl(request):
    return TorchImpl() if request.param == "torch" else JaxImpl()


@pytest.fixture(scope="module")
def built(impl):
    x = make_corpus()
    ix = impl.m.IVFIndex(n_lists=20)
    xt = impl.to(x)
    ix.train(xt)
    ix.add(xt)
    return impl, x, xt, ix


def test_full_probe_equals_brute_force(built):
    impl, x, xt, ix = built
    q = impl.to((x[7] + 0.01).astype(np.float32))
    ids, dists, scanned = ix.search(q, k=10, nprobe=ix.n_lists)
    true_ids, true_dists = impl.m.brute_force(xt, q, k=10)
    assert scanned == len(x)
    np.testing.assert_array_equal(np.sort(impl.norm(ids)),
                                  np.sort(impl.norm(true_ids)))
    np.testing.assert_allclose(np.sort(impl.norm(dists)),
                               np.sort(impl.norm(true_dists)), rtol=1e-4)


def test_partition_integrity(built):
    impl, x, xt, ix = built
    all_ids = np.concatenate([impl.norm(ids) for ids, _ in ix.lists])
    assert len(all_ids) == len(x)
    assert len(np.unique(all_ids)) == len(x)


def test_recall_increases_with_nprobe(built):
    impl, x, xt, ix = built
    queries = x[rng.choice(len(x), 30)] + rng.normal(size=(30, x.shape[1])) * 0.1
    recalls = []
    for nprobe in [1, 4, 20]:
        r = []
        for q in queries.astype(np.float32):
            qt = impl.to(q)
            ids, _, _ = ix.search(qt, k=10, nprobe=nprobe)
            true_ids, _ = impl.m.brute_force(xt, qt, k=10)
            r.append(len(set(impl.norm(ids).tolist())
                         & set(impl.norm(true_ids).tolist())) / 10)
        recalls.append(np.mean(r))
    assert recalls[0] <= recalls[1] <= recalls[2]
    assert recalls[2] == 1.0


def test_small_nprobe_scans_small_fraction(built):
    impl, x, xt, ix = built
    _, _, scanned = ix.search(impl.to(x[0]), k=5, nprobe=1)
    assert scanned < len(x) * 0.25
