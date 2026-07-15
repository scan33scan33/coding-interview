"""PyTorch port of the classifier metrics (see metrics.py for the NumPy
original and full commentary)."""

import torch


def _average_ranks(scores):
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float64)
    i = 0
    s = scores[order]
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(y_true, scores):
    y_true = y_true.bool()
    npos, nneg = int(y_true.sum()), int((~y_true).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("AUC needs both classes present")
    ranks = _average_ranks(scores.double())
    return float((ranks[y_true].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def precision_recall_curve(y_true, scores):
    y_true = y_true.double()
    order = torch.argsort(-scores, stable=True)
    y_sorted = y_true[order]
    tp = torch.cumsum(y_sorted, 0)
    fp = torch.cumsum(1 - y_sorted, 0)
    return tp / (tp + fp), tp / y_true.sum()


def average_precision(y_true, scores):
    precision, recall = precision_recall_curve(y_true, scores)
    recall_prev = torch.cat([torch.zeros(1, dtype=recall.dtype), recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def expected_calibration_error(y_true, probs, n_bins=10):
    y_true = y_true.double()
    probs = probs.double()
    edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
        if not bool(in_bin.any()):
            continue
        conf = float(probs[in_bin].mean())
        acc = float(y_true[in_bin].mean())
        ece += float(in_bin.double().mean()) * abs(acc - conf)
    return float(ece)
