import numpy as np
import pytest

from speculative import expected_speedup, generate, speculative_step

V = 5


def make_markov(seed):
    """Random first-order Markov model: next-token dist depends on last token."""
    g = np.random.default_rng(seed)
    table_VV = g.dirichlet(np.ones(V) * 0.8, size=V + 1)  # row V = empty prefix

    def fn(prefix):
        return table_VV[prefix[-1] if prefix else V]
    return fn


def empirical_first_token(target_fn, draft_fn, n=40000, gamma=3, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros(V)
    for _ in range(n):
        toks, _ = speculative_step(target_fn, draft_fn, [], gamma, rng)
        counts[toks[0]] += 1
    return counts / n


def test_exactness_with_bad_draft():
    # THE theorem: output distribution == target's, even when the draft is
    # a completely unrelated model.
    target, draft = make_markov(1), make_markov(2)
    got = empirical_first_token(target, draft)
    np.testing.assert_allclose(got, target([]), atol=0.02)


def test_exactness_with_adversarial_draft():
    # Draft nearly deterministic on one (wrong) token — worst case for
    # acceptance, still must be unbiased.
    target = make_markov(1)

    def draft(prefix):
        q = np.full(V, 0.01 / (V - 1))
        q[3] = 0.99
        return q

    got = empirical_first_token(target, draft)
    np.testing.assert_allclose(got, target([]), atol=0.02)


def test_perfect_draft_accepts_everything():
    target = make_markov(1)
    rng = np.random.default_rng(0)
    _, stats = speculative_step(target, target, [], gamma=4, rng=rng)
    assert stats["accepted"] == 4     # p/q == 1 -> always accepted


def test_acceptance_tracks_draft_quality():
    target = make_markov(1)
    similar = make_markov(1)          # identical
    different = make_markov(9)
    rng = np.random.default_rng(0)
    _, rate_similar = generate(target, similar, [], 400, gamma=3, rng=rng)
    _, rate_diff = generate(target, different, [], 400, gamma=3, rng=rng)
    assert rate_similar == 1.0
    assert rate_diff < rate_similar


def test_every_round_emits_at_least_one_token():
    target, draft = make_markov(1), make_markov(2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        toks, stats = speculative_step(target, draft, [2], gamma=3, rng=rng)
        assert 1 <= len(toks) <= 4          # gamma accepted + 1 bonus max
        assert len(toks) == stats["accepted"] + 1


def test_generated_sequence_matches_target_markov_stats():
    # Long-run bigram statistics of speculative output must match direct
    # ancestral sampling from the target.
    target, draft = make_markov(1), make_markov(2)
    rng = np.random.default_rng(0)
    spec_toks, _ = generate(target, draft, [0], 30000, gamma=3, rng=rng)

    direct_rng = np.random.default_rng(1)
    ctx, direct_toks = [0], []
    for _ in range(30000):
        tok = int(direct_rng.choice(V, p=target(ctx)))
        direct_toks.append(tok)
        ctx.append(tok)

    def bigram(toks):
        m = np.zeros((V, V))
        for a, b in zip(toks, toks[1:]):
            m[a, b] += 1
        return m / m.sum()

    np.testing.assert_allclose(bigram(spec_toks), bigram(direct_toks), atol=0.01)


def test_speedup_formula_sanity():
    # Perfect acceptance, free draft -> gamma+1 tokens per target call.
    assert expected_speedup(1.0, gamma=4, cost_ratio=0.0) == pytest.approx(5.0)
    # Zero acceptance -> every round pays for the draft and emits 1 token.
    assert expected_speedup(0.0, gamma=4, cost_ratio=0.1) == pytest.approx(1 / 1.4)
    # Higher acceptance always helps.
    assert (expected_speedup(0.9, 4, 0.1) > expected_speedup(0.5, 4, 0.1)
            > expected_speedup(0.1, 4, 0.1))
