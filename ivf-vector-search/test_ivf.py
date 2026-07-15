import numpy as np

from ivf import IVFIndex, brute_force, recall_at_k

rng = np.random.default_rng(0)


def make_corpus(n=4000, d=32, n_clusters=25, seed=1):
    g = np.random.default_rng(seed)
    centers = g.normal(size=(n_clusters, d)) * 5
    x = centers[g.integers(0, n_clusters, n)] + g.normal(size=(n, d)) * 0.6
    return x.astype(np.float32)


def build(x, n_lists=25):
    ix = IVFIndex(n_lists=n_lists)
    ix.train(x)
    ix.add(x)
    return ix


def test_full_probe_equals_brute_force():
    x = make_corpus()
    ix = build(x)
    q = x[7] + 0.01
    ids, dists, scanned = ix.search(q, k=10, nprobe=ix.n_lists)
    true_ids, true_dists = brute_force(x, q, k=10)
    assert scanned == len(x)                       # probed everything
    np.testing.assert_array_equal(np.sort(ids), np.sort(true_ids))
    np.testing.assert_allclose(np.sort(dists), np.sort(true_dists), rtol=1e-5)


def test_recall_increases_with_nprobe():
    x = make_corpus()
    ix = build(x)
    queries = x[rng.choice(len(x), 50)] + rng.normal(size=(50, x.shape[1])) * 0.1
    recalls = []
    for nprobe in [1, 4, 25]:
        r = []
        for q in queries:
            ids, _, _ = ix.search(q, k=10, nprobe=nprobe)
            true_ids, _ = brute_force(x, q, k=10)
            r.append(recall_at_k(ids, true_ids))
        recalls.append(np.mean(r))
    assert recalls[0] <= recalls[1] <= recalls[2]
    assert recalls[2] == 1.0                       # full probe is exact
    assert recalls[1] > 0.8                        # few probes ~ high recall


def test_small_nprobe_scans_small_fraction():
    x = make_corpus()
    ix = build(x)
    _, _, scanned = ix.search(x[0], k=5, nprobe=1)
    assert scanned < len(x) * 0.2


def test_every_vector_lands_in_exactly_one_list():
    x = make_corpus(n=500)
    ix = build(x, n_lists=10)
    all_ids = np.concatenate([ids for ids, _ in ix.lists])
    assert len(all_ids) == 500
    assert len(np.unique(all_ids)) == 500


def test_returns_true_neighbor_for_easy_query():
    x = make_corpus()
    ix = build(x)
    q = x[123] + 1e-4                              # essentially a corpus point
    ids, _, _ = ix.search(q, k=1, nprobe=2)
    assert ids[0] == 123


def test_boundary_query_needs_more_probes():
    # A query midway between two cluster centers is the failure mode of
    # nprobe=1: its true neighbors straddle two lists.
    g = np.random.default_rng(3)
    a = g.normal(size=(300, 8)) * 0.4
    b = g.normal(size=(300, 8)) * 0.4 + 4.0
    x = np.concatenate([a, b]).astype(np.float32)
    ix = build(x, n_lists=2)
    q = np.full(8, 2.0, dtype=np.float32)          # midpoint
    true_ids, _ = brute_force(x, q, k=20)
    ids1, _, _ = ix.search(q, k=20, nprobe=1)
    ids2, _, _ = ix.search(q, k=20, nprobe=2)
    assert recall_at_k(ids2, true_ids) == 1.0
    assert recall_at_k(ids1, true_ids) < 1.0
