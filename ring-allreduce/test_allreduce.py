import numpy as np
import pytest

from allreduce import (chunk_bounds, naive_allreduce_volume, ring_allreduce,
                       ring_volume_per_worker)

rng = np.random.default_rng(0)


@pytest.mark.parametrize("p,n", [(2, 10), (4, 64), (5, 17), (8, 8), (3, 1000)])
def test_every_worker_gets_the_sum(p, n):
    grads = [rng.normal(size=n) for _ in range(p)]
    want = np.sum(grads, axis=0)
    results, _ = ring_allreduce(grads)
    for r in results:
        np.testing.assert_allclose(r, want, atol=1e-10)


def test_uneven_chunks_cover_everything():
    bounds = chunk_bounds(17, 5)          # 17 = 4+4+3+3+3
    assert bounds[0] == (0, 4)
    assert bounds[-1] == (14, 17)
    sizes = [hi - lo for lo, hi in bounds]
    assert sum(sizes) == 17
    assert max(sizes) - min(sizes) <= 1


def test_message_count_is_2_p_minus_1_rounds():
    p, n = 6, 60
    _, log = ring_allreduce([rng.normal(size=n) for _ in range(p)])
    # Each of the 2(p-1) rounds has exactly p messages (one per worker).
    assert len(log) == 2 * (p - 1) * p
    rs = [m for m in log if m[0] == "reduce_scatter"]
    ag = [m for m in log if m[0] == "all_gather"]
    assert len(rs) == len(ag) == (p - 1) * p


def test_communication_volume_is_bandwidth_optimal():
    # Per-worker send volume must be ~ 2N(p-1)/p — independent of p in the
    # large-p limit, vs (p-1)*N for naive all-to-all.
    p, n = 8, 800
    _, log = ring_allreduce([rng.normal(size=n) for _ in range(p)])
    sent = ring_volume_per_worker(log, p)
    expected = 2 * n * (p - 1) / p
    for s in sent:
        assert abs(s - expected) <= p          # rounding from uneven chunks
    assert max(sent) < naive_allreduce_volume(p, n) / 3


def test_only_neighbor_communication():
    p = 5
    _, log = ring_allreduce([rng.normal(size=20) for _ in range(p)])
    for _, _, src, dst, _ in log:
        assert dst == (src + 1) % p            # ring edges only


def test_gradient_averaging_use_case():
    # Data-parallel: each worker computes grads on its shard; after
    # all-reduce / P every worker steps with the same averaged gradient.
    p, n = 4, 32
    grads = [rng.normal(size=n) for _ in range(p)]
    results, _ = ring_allreduce(grads)
    avg = [r / p for r in results]
    want = np.mean(grads, axis=0)
    for a in avg:
        np.testing.assert_allclose(a, want, atol=1e-10)


def test_n_smaller_than_p_still_works():
    p, n = 8, 3                                # some chunks are empty
    grads = [rng.normal(size=n) for _ in range(p)]
    results, _ = ring_allreduce(grads)
    for r in results:
        np.testing.assert_allclose(r, np.sum(grads, axis=0), atol=1e-12)
