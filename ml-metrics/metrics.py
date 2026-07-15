"""Classifier evaluation metrics from scratch: ROC-AUC (rank-based, with
correct tie handling), average precision (AP), and expected calibration
error (ECE). No sklearn.

The classic MLE screen: everyone can CALL roc_auc_score; this checks you
know what the number is.
"""

import numpy as np


def _average_ranks(scores):
    """1-based ranks; tied scores share the average of their rank range."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1     # average of ranks i+1..j+1
        i = j + 1
    return ranks


def roc_auc(y_true, scores):
    """AUC = P(score(random positive) > score(random negative))
           + 0.5 * P(equal)   — the Mann-Whitney U statistic.

    Rank-based: AUC = (sum of positive ranks - npos(npos+1)/2) / (npos*nneg).
    Average ranks make ties contribute exactly 1/2, matching the
    probabilistic definition.
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    npos, nneg = int(y_true.sum()), int((~y_true).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("AUC needs both classes present")
    ranks = _average_ranks(scores)
    return (ranks[y_true].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def precision_recall_curve(y_true, scores):
    """Precision/recall at every score threshold, high threshold first."""
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp)
    recall = tp / y_true.sum()
    return precision, recall


def average_precision(y_true, scores):
    """AP = sum over positive-hit positions of precision-at-that-depth,
    divided by the number of positives. Step-integral of the PR curve; the
    standard summary that (unlike ROC-AUC) doesn't credit true negatives —
    the right metric under heavy class imbalance."""
    precision, recall = precision_recall_curve(y_true, scores)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE: bin predictions by confidence; weighted mean |accuracy - conf|.

    A model can have great AUC and terrible calibration (AUC only sees
    ranking). ECE is what tells you whether "0.9" means 90%.
    """
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if not in_bin.any():
            continue
        conf = probs[in_bin].mean()
        acc = y_true[in_bin].mean()
        ece += in_bin.mean() * abs(acc - conf)
    return float(ece)
