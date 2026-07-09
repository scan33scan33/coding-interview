import numpy as np
import pytest

from online_attention import attention_naive, attention_tiled

rng = np.random.default_rng(0)


@pytest.mark.parametrize("block", [1, 3, 16, 64, 1000])
def test_matches_naive_any_block_size(block):
    # Correctness must not depend on tiling granularity — including block
    # sizes that don't divide T and a single block covering everything.
    q_LK = rng.normal(size=(9, 8))
    k_TK = rng.normal(size=(37, 8))
    v_TK = rng.normal(size=(37, 8))
    got = attention_tiled(q_LK, k_TK, v_TK, block=block)
    want = attention_naive(q_LK, k_TK, v_TK)
    np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("block", [4, 7, 32])
def test_causal_matches_naive(block):
    q_LK = rng.normal(size=(20, 16))
    k_TK = rng.normal(size=(20, 16))
    v_TK = rng.normal(size=(20, 16))
    got = attention_tiled(q_LK, k_TK, v_TK, block=block, causal=True)
    want = attention_naive(q_LK, k_TK, v_TK, causal=True)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_numerically_stable_with_huge_logits():
    # Scores around +/-500: a softmax without max-subtraction overflows
    # (exp(500) = inf -> nan). The online recurrence must stay finite.
    q_LK = rng.normal(size=(4, 8)) * 200
    k_TK = rng.normal(size=(50, 8)) * 200
    v_TK = rng.normal(size=(50, 8))
    got = attention_tiled(q_LK, k_TK, v_TK, block=8)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, attention_naive(q_LK, k_TK, v_TK), atol=1e-8)


def test_rescaling_actually_happens():
    # Put the max-scoring key in the LAST block, so every earlier block's
    # accumulator must be rescaled when the running max jumps. An
    # implementation that forgets alpha gets this badly wrong.
    K = 8
    q_LK = np.ones((1, K))
    k_TK = np.zeros((33, K))
    k_TK[-1] = 10.0  # dominant key arrives last
    v_TK = rng.normal(size=(33, K))
    got = attention_tiled(q_LK, k_TK, v_TK, block=8)
    want = attention_naive(q_LK, k_TK, v_TK)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_causal_first_row_attends_only_to_itself():
    # Row 0 may only see key 0, so its output is exactly v[0] — also
    # exercises fully-masked blocks (every block after the first).
    q_LK = rng.normal(size=(6, 4))
    k_TK = rng.normal(size=(6, 4))
    v_TK = rng.normal(size=(6, 4))
    got = attention_tiled(q_LK, k_TK, v_TK, block=2, causal=True)
    np.testing.assert_allclose(got[0], v_TK[0], atol=1e-12)


def test_attention_output_is_convex_combination():
    # Softmax weights are nonnegative and sum to 1, so each output row must
    # lie inside the per-dimension range of V.
    q_LK = rng.normal(size=(5, 8))
    k_TK = rng.normal(size=(40, 8))
    v_TK = rng.normal(size=(40, 8))
    got = attention_tiled(q_LK, k_TK, v_TK, block=8)
    assert (got <= v_TK.max(axis=0) + 1e-12).all()
    assert (got >= v_TK.min(axis=0) - 1e-12).all()
