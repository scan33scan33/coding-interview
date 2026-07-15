"""JAX port of ring all-reduce (see allreduce.py for the NumPy original and
full commentary).

JAX arrays are immutable, so each worker's buffer is kept as a LIST OF
CHUNKS — the natural layout for this algorithm anyway (real implementations
also operate on registered chunk buffers, not one mutable slab). Receives
build new chunk arrays instead of writing in place.
"""

import jax.numpy as jnp

from allreduce import chunk_bounds  # pure python, shared


def ring_allreduce(grads_PN):
    p = len(grads_PN)
    n = len(grads_PN[0])
    bounds = chunk_bounds(n, p)
    # buf[i][c] = worker i's copy of chunk c.
    buf = [[jnp.asarray(g[lo:hi], dtype=jnp.float32) for lo, hi in bounds]
           for g in grads_PN]
    log = []

    def send(phase, step, src, dst, chunk):
        lo, hi = bounds[chunk]
        log.append((phase, step, src, dst, hi - lo))
        return buf[src][chunk]

    for s in range(p - 1):
        transfers = []
        for i in range(p):
            chunk = (i - s) % p
            payload = send("reduce_scatter", s, i, (i + 1) % p, chunk)
            transfers.append(((i + 1) % p, chunk, payload))
        for dst, chunk, payload in transfers:
            buf[dst][chunk] = buf[dst][chunk] + payload

    for s in range(p - 1):
        transfers = []
        for i in range(p):
            chunk = (i + 1 - s) % p
            payload = send("all_gather", s, i, (i + 1) % p, chunk)
            transfers.append(((i + 1) % p, chunk, payload))
        for dst, chunk, payload in transfers:
            buf[dst][chunk] = payload

    return [jnp.concatenate(chunks) if n else jnp.zeros(0)
            for chunks in buf], log
