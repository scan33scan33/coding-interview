"""PyTorch port of ring all-reduce (see allreduce.py for the NumPy original
and full commentary)."""

import torch

from allreduce import chunk_bounds  # pure python, shared


def ring_allreduce(grads_PN):
    p = len(grads_PN)
    n = len(grads_PN[0])
    buf_PN = [g.double().clone() for g in grads_PN]
    bounds = chunk_bounds(n, p)
    log = []

    def send(phase, step, src, dst, chunk):
        lo, hi = bounds[chunk]
        log.append((phase, step, src, dst, hi - lo))
        return buf_PN[src][lo:hi].clone(), lo, hi

    for s in range(p - 1):
        transfers = []
        for i in range(p):
            chunk = (i - s) % p
            payload, lo, hi = send("reduce_scatter", s, i, (i + 1) % p, chunk)
            transfers.append(((i + 1) % p, lo, hi, payload))
        for dst, lo, hi, payload in transfers:
            buf_PN[dst][lo:hi] += payload

    for s in range(p - 1):
        transfers = []
        for i in range(p):
            chunk = (i + 1 - s) % p
            payload, lo, hi = send("all_gather", s, i, (i + 1) % p, chunk)
            transfers.append(((i + 1) % p, lo, hi, payload))
        for dst, lo, hi, payload in transfers:
            buf_PN[dst][lo:hi] = payload

    return buf_PN, log
