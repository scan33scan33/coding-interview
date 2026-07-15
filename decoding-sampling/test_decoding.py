import numpy as np
import pytest

from decoding import (apply_repetition_penalty, apply_temperature, beam_search,
                      greedy_decode, sample_token, softmax, top_k_filter,
                      top_p_filter)

rng = np.random.default_rng(0)


def test_temperature_preserves_argmax_and_flattens():
    logits = np.array([1.0, 3.0, 2.0])
    for t in [0.1, 1.0, 10.0]:
        assert np.argmax(apply_temperature(logits, t)) == 1
    p_cold = softmax(apply_temperature(logits, 0.1))
    p_hot = softmax(apply_temperature(logits, 10.0))
    assert p_cold[1] > softmax(logits)[1] > p_hot[1]  # cold sharpens, hot flattens


def test_top_k_keeps_exactly_k():
    logits = np.array([5.0, 1.0, 3.0, 4.0, 2.0])
    out = top_k_filter(logits, 2)
    assert np.isfinite(out).sum() == 2
    assert np.isfinite(out[[0, 3]]).all()          # the two largest survive
    out_all = top_k_filter(logits, 10)             # k >= V is a no-op
    assert np.isfinite(out_all).all()


def test_top_p_keeps_smallest_covering_set():
    # probs = [0.5, 0.25, 0.125, 0.125]; p=0.7 needs {0.5, 0.25}.
    logits = np.log(np.array([0.5, 0.25, 0.125, 0.125]))
    out = top_p_filter(logits, 0.7)
    assert np.isfinite(out[[0, 1]]).all()
    assert np.isinf(out[[2, 3]]).all()


def test_top_p_includes_boundary_token_and_top1():
    logits = np.log(np.array([0.5, 0.25, 0.125, 0.125]))
    out = top_p_filter(logits, 0.5)     # top-1 alone reaches p exactly
    assert np.isfinite(out).sum() == 1
    out = top_p_filter(logits, 0.51)    # crossing token must be INCLUDED
    assert np.isfinite(out).sum() == 2
    out = top_p_filter(logits, 1e-9)    # never returns an empty set
    assert np.isfinite(out).sum() == 1


def test_sampling_matches_softmax_distribution():
    logits = np.array([0.0, 1.0, 2.0])
    want = softmax(logits)
    draws = [sample_token(logits, rng) for _ in range(20000)]
    got = np.bincount(draws, minlength=3) / len(draws)
    np.testing.assert_allclose(got, want, atol=0.02)


def test_repetition_penalty_reduces_prob_for_any_sign():
    # The classic bug: naive division INCREASES probability of tokens with
    # negative logits. Both signs must go DOWN.
    logits = np.array([2.0, -2.0, 0.5])
    before = softmax(logits)
    after = softmax(apply_repetition_penalty(logits, [0, 1], penalty=1.5))
    assert after[0] < before[0]
    assert after[1] < before[1]


def make_trap_model():
    """3-token vocab {0,1,eos=2}, deterministic table model.

    From BOS, token 0 looks best (p=0.6) but leads to a dead end where the
    model dithers; token 1 (p=0.4) leads to immediate confident EOS. The
    full-sequence probability of [1, eos] beats anything starting with 0 —
    greedy falls into the trap, beam search must not.
    """
    table = {
        (): [np.log(0.6), np.log(0.4), -np.inf],
        (0,): [np.log(0.5), np.log(0.49), np.log(0.01)],
        (1,): [-np.inf, -np.inf, 0.0],  # p(eos)=1
    }

    def step_fn(seq):
        key = tuple(seq[1:])[:1] if len(seq) > 1 else ()
        return np.array(table.get(key, table[(0,)]))
    return step_fn


def test_beam_search_escapes_greedy_trap():
    step_fn = make_trap_model()
    greedy = greedy_decode(step_fn, bos=3, eos=2, max_len=5)
    assert greedy[1] == 0                                    # greedy takes the bait
    best, _ = beam_search(step_fn, bos=3, eos=2, beam_width=2, max_len=5)
    assert best == [3, 1, 2]                                 # beam finds [1, eos]


def test_beam_width_one_is_greedy():
    step_fn = make_trap_model()
    best, _ = beam_search(step_fn, bos=3, eos=2, beam_width=1, max_len=5)
    greedy = greedy_decode(step_fn, bos=3, eos=2, max_len=5)
    assert best[:len(greedy)] == greedy


def test_length_normalization_prefers_longer_completion():
    # Model offers: eos now (p=0.4) or token 0 then certain eos (p=0.6*~1).
    # Raw scores: log .4 = -0.92 vs log .6 = -0.51 -- longer already wins;
    # make the short one slightly better raw, and alpha>0 flip the ranking.
    table = {
        (): [np.log(0.45), -np.inf, np.log(0.55)],   # short: eos p=.55
        (0,): [-np.inf, -np.inf, 0.0],               # long: 0 then eos p=1
    }

    def step_fn(seq):
        # default: any other prefix ends immediately (dead-end filler beams)
        return np.array(table.get(tuple(seq[1:]), [-np.inf, -np.inf, 0.0]))

    short, _ = beam_search(step_fn, bos=9, eos=2, beam_width=2, max_len=4,
                           length_alpha=0.0)
    long, _ = beam_search(step_fn, bos=9, eos=2, beam_width=2, max_len=4,
                          length_alpha=1.0)
    assert short == [9, 2]           # raw log-prob prefers immediate EOS
    assert long == [9, 0, 2]         # per-token normalization prefers longer


def test_pipeline_composes():
    logits = rng.normal(size=50)
    tok = sample_token(logits, rng, temperature=0.8, top_k=10, top_p=0.9,
                       prev_ids=[3, 7], repetition_penalty=1.3)
    assert 0 <= tok < 50
