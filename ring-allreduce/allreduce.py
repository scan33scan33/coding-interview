"""Ring all-reduce, simulated: P workers each hold a gradient vector; after
the collective, every worker holds the elementwise SUM, having communicated
only ~2N/P per step instead of naive gather-everything.

The algorithm behind data-parallel gradient sync (NCCL, Horovod):
  Phase 1 (reduce-scatter): P-1 steps; after it, worker i holds the fully
    reduced chunk (i+1) mod P.
  Phase 2 (all-gather): P-1 steps circulating the finished chunks.

Every message is logged so tests can verify the communication volume, not
just the arithmetic.

Shape suffix convention: P = workers, N = gradient length
"""

import numpy as np


def chunk_bounds(n, p):
    """Split [0, n) into p contiguous chunks, sizes differing by at most 1."""
    base, extra = divmod(n, p)
    bounds, start = [], 0
    for i in range(p):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def ring_allreduce(grads_PN):
    """grads_PN: list of P equal-length float arrays (one per worker).

    Returns (results, log): results[i] is worker i's final buffer (== sum of
    all inputs); log is a list of (phase, step, src, dst, n_elems) records.
    Workers only ever exchange data with ring neighbors i -> (i+1) mod P.
    """
    p = len(grads_PN)
    n = len(grads_PN[0])
    buf_PN = [g.astype(np.float64).copy() for g in grads_PN]
    bounds = chunk_bounds(n, p)
    log = []

    def send(phase, step, src, dst, chunk):
        lo, hi = bounds[chunk]
        log.append((phase, step, src, dst, hi - lo))
        return buf_PN[src][lo:hi].copy(), lo, hi

    # Phase 1: reduce-scatter. At step s, worker i sends chunk (i - s) mod p
    # to its right neighbor, which ADDS it into its own buffer. After p-1
    # steps, worker i holds the complete sum for chunk (i + 1) mod p.
    for s in range(p - 1):
        transfers = []
        for i in range(p):
            chunk = (i - s) % p
            payload, lo, hi = send("reduce_scatter", s, i, (i + 1) % p, chunk)
            transfers.append(((i + 1) % p, lo, hi, payload))
        for dst, lo, hi, payload in transfers:   # apply after all sends:
            buf_PN[dst][lo:hi] += payload        # steps are synchronous

    # Phase 2: all-gather. Worker i starts owning finished chunk (i+1) mod p
    # and forwards it around the ring; receivers OVERWRITE (no adding).
    for s in range(p - 1):
        transfers = []
        for i in range(p):
            chunk = (i + 1 - s) % p
            payload, lo, hi = send("all_gather", s, i, (i + 1) % p, chunk)
            transfers.append(((i + 1) % p, lo, hi, payload))
        for dst, lo, hi, payload in transfers:
            buf_PN[dst][lo:hi] = payload

    return buf_PN, log


def naive_allreduce_volume(p, n):
    """Everyone sends its full vector to everyone: per-worker send volume."""
    return (p - 1) * n


def ring_volume_per_worker(log, p):
    """Per-worker elements SENT according to the message log."""
    sent = [0] * p
    for _, _, src, _, n_elems in log:
        sent[src] += n_elems
    return sent
