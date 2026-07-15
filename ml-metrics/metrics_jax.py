"""JAX port of the classifier metrics (see metrics.py for the NumPy
original and full commentary). Tie-averaged ranks are computed with a
vectorized double-argsort + segment trick instead of a while loop."""

import jax.numpy as jnp


def _average_ranks(scores):
    """1-based average ranks; ties share the mean of their rank range.

    Vectorized: sort once, then for each run of equal values assign the
    average of first/last ordinal rank in the run.
    """
    order = jnp.argsort(scores)
    s = scores[order]
    n = len(scores)
    idx = jnp.arange(n)
    # start[i] = first index of the run containing i; end[i] = last index.
    new_run = jnp.concatenate([jnp.array([True]), s[1:] != s[:-1]])
    run_id = jnp.cumsum(new_run) - 1
    # Scatter-min needs an identity of +inf (here: n); scatter-max needs 0.
    first = jnp.full(n, n, dtype=idx.dtype).at[run_id].min(idx)
    last = jnp.zeros(n, dtype=idx.dtype).at[run_id].max(idx)
    avg = (first[run_id] + last[run_id]) / 2 + 1
    return jnp.zeros(n, dtype=jnp.float64).at[order].set(avg)


def roc_auc(y_true, scores):
    y_true = y_true.astype(bool)
    npos = int(y_true.sum())
    nneg = int((~y_true).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("AUC needs both classes present")
    ranks = _average_ranks(scores.astype(jnp.float64))
    return float((ranks[y_true].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def precision_recall_curve(y_true, scores):
    y_true = y_true.astype(jnp.float64)
    order = jnp.argsort(-scores, stable=True)
    y_sorted = y_true[order]
    tp = jnp.cumsum(y_sorted)
    fp = jnp.cumsum(1 - y_sorted)
    return tp / (tp + fp), tp / y_true.sum()


def average_precision(y_true, scores):
    precision, recall = precision_recall_curve(y_true, scores)
    recall_prev = jnp.concatenate([jnp.zeros(1, dtype=recall.dtype), recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def expected_calibration_error(y_true, probs, n_bins=10):
    y_true = y_true.astype(jnp.float64)
    probs = probs.astype(jnp.float64)
    edges = jnp.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
        cnt = float(in_bin.sum())
        if cnt == 0:
            continue
        conf = float(probs[in_bin].mean())
        acc = float(y_true[in_bin].mean())
        ece += (cnt / len(probs)) * abs(acc - conf)
    return float(ece)
