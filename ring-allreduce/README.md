# Ring All-Reduce (Data-Parallel Gradient Sync)

**Problem:** Implement **ring all-reduce** — the collective behind
data-parallel training (NCCL, Horovod) — as a step-synchronous simulation:
P workers each hold a gradient vector; after the collective **every worker
holds the sum**, and every message goes only to a ring neighbor.

The tests grade the *systems* content, not just arithmetic: message counts,
per-worker communication volume (bandwidth optimality), and the
neighbors-only constraint are all verified from a message log.

---

## Core Requirements

1. **Chunking (`chunk_bounds`)** — split length-N into P contiguous chunks,
   sizes differing by ≤ 1 (N need not divide evenly; N < P must work —
   empty chunks are legal).

2. **Phase 1 — reduce-scatter (P−1 steps).** At step s, worker i sends chunk
   `(i − s) mod P` to worker `(i+1) mod P`, which **adds** it into its
   buffer. Afterwards, worker i holds the *complete sum* of chunk
   `(i+1) mod P` — each chunk finished its trip around the ring exactly
   where the schedule says.

3. **Phase 2 — all-gather (P−1 steps).** The finished chunks circulate;
   receivers **overwrite** (adding here double-counts — the classic bug).

4. **Message log** — every send records `(phase, step, src, dst, size)`.
   From the log, verify: `2(P−1)` rounds of exactly P messages, per-worker
   send volume `≈ 2N(P−1)/P`, and ring-edges-only traffic.

---

## Why this is the algorithm everyone uses

Naive all-to-all costs `(P−1)·N` per worker. Ring costs `2N(P−1)/P ≈ 2N` —
**independent of P**: adding workers doesn't add per-worker bandwidth. That's
bandwidth-optimal for large N; the price is `2(P−1)` latency rounds, which is
why small-message collectives (and small P) sometimes use trees instead.

---

## Behavior Notes / Gotchas

- **Add in phase 1, overwrite in phase 2.** Mixing them up still produces
  plausible-looking numbers on symmetric inputs; the random-input sum test
  catches it immediately.
- **The chunk schedule is not arbitrary.** `(i − s) mod P` guarantees each
  chunk visits every worker exactly once during reduce-scatter and lands
  finished at the right owner. Off-by-one in the schedule = some chunk
  reduced P−2 times.
- **Synchronous step semantics:** all sends in a round read buffer state
  *before* any receive of the same round is applied (collect transfers,
  then apply). Applying in-loop lets a payload contain a same-round
  update — a race you'd otherwise only see on real hardware.
- **Uneven N:** chunk sizes differ by one; volume accounting in the test
  allows that slack. N < P produces empty chunks that must ship 0 elements
  gracefully.
- **This is why gradients are averaged AFTER the collective** — all-reduce
  gives the sum; divide by P locally (tested as the data-parallel use case).

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_allreduce.py -v
```

| Test | Validates |
|------|-----------|
| `test_every_worker_gets_the_sum` | Correctness across P/N combos (incl. uneven) |
| `test_uneven_chunks_cover_everything` | Chunking partition |
| `test_message_count_is_2_p_minus_1_rounds` | Exactly 2(P−1) rounds × P messages |
| `test_communication_volume_is_bandwidth_optimal` | ~2N(P−1)/P per worker, ≪ naive |
| `test_only_neighbor_communication` | Ring topology respected |
| `test_gradient_averaging_use_case` | The data-parallel workflow |
| `test_n_smaller_than_p_still_works` | Empty-chunk edge case |

---

## Discussion Questions (interview follow-ups)

- **Latency vs bandwidth** — total time ≈ `2(P−1)·α + 2N(P−1)/P·β`. When do
  tree all-reduce or recursive halving-doubling win? Why does NCCL switch
  algorithms by message size?
- **Overlap** — how does bucketed all-reduce overlap communication with the
  backward pass (PyTorch DDP gradient buckets, and why bucket order is
  reversed)?
- **Reduce-scatter + all-gather as first-class ops** — ZeRO/FSDP use the two
  phases *separately*. What does each phase alone give you, and why does
  sharded training want exactly that split?
- **Numerical nondeterminism** — different reduction orders give different
  float sums. Why can two identical runs diverge, and when do you care?
- **Failure** — one worker stalls: what happens to a ring vs a tree, and
  what do elastic trainers do about it?
