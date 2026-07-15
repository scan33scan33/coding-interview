import numpy as np
import pytest

from metrics import (average_precision, expected_calibration_error,
                     precision_recall_curve, roc_auc)

rng = np.random.default_rng(0)


def test_auc_perfect_reversed_random():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    y_big = rng.integers(0, 2, 5000)
    s_random = rng.uniform(size=5000)
    assert abs(roc_auc(y_big, s_random) - 0.5) < 0.03


def test_auc_equals_pairwise_probability():
    # AUC must equal the empirical P(score_pos > score_neg) + 0.5*P(equal),
    # computed by brute force over all pos/neg pairs.
    y = rng.integers(0, 2, 200)
    s = np.round(rng.uniform(size=200), 1)        # coarse scores force ties
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    want = (wins + 0.5 * ties) / (len(pos) * len(neg))
    assert abs(roc_auc(y, s) - want) < 1e-12


def test_auc_tie_handling_hand_case():
    # All scores identical: every pair is a tie -> AUC exactly 0.5.
    y = np.array([0, 1, 0, 1, 1])
    s = np.ones(5)
    assert roc_auc(y, s) == 0.5


def test_auc_invariant_to_monotone_transform():
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300)
    a = roc_auc(y, s)
    assert abs(roc_auc(y, 1 / (1 + np.exp(-3 * s + 1))) - a) < 1e-12
    assert abs(roc_auc(y, s ** 3) - a) < 1e-12    # cube preserves order


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc(np.ones(5), rng.uniform(size=5))


def test_pr_curve_hand_case():
    # scores desc: [1(P), 0.8(N), 0.6(P)] ->
    # depth1: P=1/1 R=1/2; depth2: P=1/2 R=1/2; depth3: P=2/3 R=1
    y = np.array([1, 0, 1])
    s = np.array([1.0, 0.8, 0.6])
    precision, recall = precision_recall_curve(y, s)
    np.testing.assert_allclose(precision, [1, 1 / 2, 2 / 3])
    np.testing.assert_allclose(recall, [1 / 2, 1 / 2, 1])
    # AP = 1.0*(0.5-0) + 0.5*(0.5-0.5) + (2/3)*(1-0.5) = 0.8333...
    assert abs(average_precision(y, s) - (0.5 + 1 / 3)) < 1e-9


def test_ap_perfect_ranking_is_one():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert abs(average_precision(y, s) - 1.0) < 1e-12


def test_ap_reflects_imbalance_but_auc_does_not():
    # Same ranking quality, positives diluted 10x with easy negatives:
    # AUC stays high; AP (which ignores true negatives) drops.
    y_bal = np.concatenate([np.ones(50), np.zeros(50)])
    s_bal = np.concatenate([rng.normal(1.0, 1.0, 50), rng.normal(0.0, 1.0, 50)])
    y_imb = np.concatenate([np.ones(50), np.zeros(2000)])
    s_imb = np.concatenate([rng.normal(1.0, 1.0, 50), rng.normal(0.0, 1.0, 2000)])
    assert abs(roc_auc(y_bal, s_bal) - roc_auc(y_imb, s_imb)) < 0.06
    assert average_precision(y_imb, s_imb) < average_precision(y_bal, s_bal) - 0.2


def test_ece_perfectly_calibrated_is_small():
    # Labels drawn WITH probability equal to the prediction -> near-zero ECE.
    probs = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < probs).astype(float)
    assert expected_calibration_error(y, probs) < 0.02


def test_ece_overconfident_model_is_flagged():
    # Predicts 0.95 but is right only 60% of the time -> ECE ~ 0.35.
    probs = np.full(2000, 0.95)
    y = (rng.uniform(size=2000) < 0.6).astype(float)
    ece = expected_calibration_error(y, probs)
    assert abs(ece - 0.35) < 0.05


def test_good_auc_bad_calibration():
    # Squash well-ranked scores into [0.45, 0.55]: ranking (AUC) unaffected,
    # but confident labels make it badly calibrated vs its own probabilities.
    y = rng.integers(0, 2, 4000)
    s = np.where(y == 1, rng.normal(2.0, 1.0, 4000), rng.normal(-2.0, 1.0, 4000))
    squashed = 0.5 + 0.05 * np.tanh(s)
    assert roc_auc(y, squashed) > 0.9
    assert expected_calibration_error(y, squashed) > 0.2   # underconfident
