import random
from collections import Counter

from cms import CountMinSketch, HeavyHitters


def zipf_stream(n, n_keys=2000, alpha=1.2, seed=0):
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** alpha for i in range(n_keys)]
    keys = [f"key{i}" for i in range(n_keys)]
    return rng.choices(keys, weights=weights, k=n)


def test_never_undercounts():
    stream = zipf_stream(20000)
    cms, truth = CountMinSketch(eps=0.005, delta=0.01), Counter(stream)
    for k in stream:
        cms.add(k)
    assert all(cms.query(k) >= truth[k] for k in truth)


def test_error_bound_holds():
    # estimate <= true + eps*N for (nearly) all keys; the guarantee is
    # per-key with prob 1-delta, so allow a delta-sized failure fraction.
    stream = zipf_stream(20000)
    eps = 0.005
    cms, truth = CountMinSketch(eps=eps, delta=0.01), Counter(stream)
    for k in stream:
        cms.add(k)
    n = len(stream)
    violations = sum(cms.query(k) > truth[k] + eps * n for k in truth)
    assert violations / len(truth) <= 0.02


def test_conservative_update_dominates_standard():
    stream = zipf_stream(20000)
    std = CountMinSketch(eps=0.01, delta=0.05, seed=3)
    con = CountMinSketch(eps=0.01, delta=0.05, seed=3, conservative=True)
    for k in stream:
        std.add(k)
        con.add(k)
    truth = Counter(stream)
    # Conservative estimates are never below truth and never above standard.
    for k in truth:
        assert truth[k] <= con.query(k) <= std.query(k)
    total_std = sum(std.query(k) - truth[k] for k in truth)
    total_con = sum(con.query(k) - truth[k] for k in truth)
    assert total_con < total_std          # strictly less total overestimation


def test_weighted_adds():
    cms = CountMinSketch(eps=0.01, delta=0.01)
    cms.add("a", 10)
    cms.add("a", 5)
    assert cms.query("a") >= 15
    assert cms.query("zzz") >= 0


def test_memory_independent_of_cardinality():
    cms = CountMinSketch(eps=0.01, delta=0.01)
    cells_before = sum(len(row) for row in cms.table)
    for i in range(50000):                # 50k distinct keys
        cms.add(f"unique-{i}")
    assert sum(len(row) for row in cms.table) == cells_before


def test_heavy_hitters_finds_the_head():
    stream = zipf_stream(50000, alpha=1.5, seed=1)
    hh = HeavyHitters(k=10)
    for k in stream:
        hh.add(k)
    truth = Counter(stream)
    true_top5 = {k for k, _ in truth.most_common(5)}
    found = {k for k, _ in hh.top()}
    # The unambiguous head of a steep zipf must all be tracked.
    assert true_top5 <= found


def test_heavy_hitters_bounded_memory():
    hh = HeavyHitters(k=5)
    for i in range(20000):
        hh.add(f"key-{i % 1000}")
    assert len(hh.tracked) <= 5
