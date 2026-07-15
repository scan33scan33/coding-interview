import numpy as np

from kmeans import kmeans, kmeans_pp_init, pairwise_sq_dists

rng = np.random.default_rng(0)


def make_blobs(k=4, per=50, spread=0.3, sep=10.0, d=2, seed=1):
    g = np.random.default_rng(seed)
    centers = g.normal(size=(k, d)) * sep
    x = np.concatenate([c + g.normal(size=(per, d)) * spread for c in centers])
    labels = np.repeat(np.arange(k), per)
    return x, labels, centers


def test_pairwise_dists_match_naive():
    x_ND, c_KD = rng.normal(size=(30, 5)), rng.normal(size=(4, 5))
    naive = ((x_ND[:, None, :] - c_KD[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(pairwise_sq_dists(x_ND, c_KD), naive, atol=1e-9)


def test_inertia_monotonically_decreases():
    x, _, _ = make_blobs()
    _, _, history = kmeans(x, k=4, rng=np.random.default_rng(0))
    assert all(a >= b - 1e-9 for a, b in zip(history, history[1:]))


def test_recovers_separated_blobs():
    x, true_labels, _ = make_blobs(k=4, spread=0.2, sep=20.0)
    _, labels, _ = kmeans(x, k=4, rng=np.random.default_rng(0))
    # Cluster ids are arbitrary: check the PARTITION matches, i.e. every
    # found cluster contains points of exactly one true blob.
    for j in range(4):
        assert len(set(true_labels[labels == j])) == 1


def test_kmeanspp_beats_random_init_on_average():
    # Adversarial-ish data: unequal, well-separated blobs. Random init often
    # puts 2 centers in one big blob and merges two others; ++ rarely does.
    g = np.random.default_rng(2)
    x = np.concatenate([
        g.normal(size=(200, 2)) * 0.3,                    # big blob at origin
        g.normal(size=(10, 2)) * 0.3 + [30, 0],
        g.normal(size=(10, 2)) * 0.3 + [0, 30],
        g.normal(size=(10, 2)) * 0.3 + [30, 30],
    ])
    final = {"++": [], "random": []}
    for seed in range(15):
        for init in final:
            _, _, hist = kmeans(x, k=4, rng=np.random.default_rng(seed), init=init)
            final[init].append(hist[-1])
    assert np.mean(final["++"]) < np.mean(final["random"])


def test_empty_cluster_is_reseeded():
    # k=3 with two tight far-apart pairs and centers engineered so one
    # cluster starves: after convergence every center must still own points.
    x = np.array([[0.0, 0], [0.1, 0], [50.0, 0], [50.1, 0], [50.2, 0], [49.9, 0]])
    _, labels, _ = kmeans(x, k=3, rng=np.random.default_rng(0), init="random")
    assert len(set(labels)) == 3


def test_deterministic_given_seed():
    x, _, _ = make_blobs()
    c1, l1, h1 = kmeans(x, k=4, rng=np.random.default_rng(7))
    c2, l2, h2 = kmeans(x, k=4, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_allclose(c1, c2)
    assert h1 == h2


def test_pp_init_spreads_centers():
    x, _, true_centers = make_blobs(k=4, spread=0.2, sep=20.0)
    c_KD = kmeans_pp_init(x, 4, np.random.default_rng(0))
    # Each ++ center should land near a DIFFERENT true blob.
    nearest = pairwise_sq_dists(c_KD, true_centers).argmin(axis=1)
    assert len(set(nearest.tolist())) == 4
