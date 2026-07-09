# Online Softmax / Tiled Attention (the FlashAttention core)

**Problem:** Compute `softmax(QKᵀ/√d)·V` **without ever materializing the
L × T attention matrix**. Keys and values stream in blocks; per-row running
statistics are updated as each block arrives. This is the *online softmax*
recurrence at the heart of FlashAttention — implemented in NumPy so the
algorithm (not CUDA) is what's being tested.

The must-pass correctness bar: bit-for-bit-close agreement with a naive
reference for **any block size**, including blocks that don't divide T, and
with scores large enough to overflow an unsafe softmax.

---

## Core Requirements

1. **Reference (`attention_naive`)** — full L × T scores, max-subtracted
   (safe) softmax, optional causal mask.

2. **Streaming version (`attention_tiled`)**
   - Iterate over key/value blocks of size `C`. Peak extra memory O(L·C),
     independent of T.
   - Maintain, per query row:
     - `m` — running max of scores seen so far (numerical stability),
     - `l` — running softmax denominator `Σ exp(s − m)`,
     - `o` — **unnormalized** output accumulator `Σ exp(s − m)·V`.
   - **The rescaling step:** when a new block raises the max `m → m′`,
     everything accumulated so far was scaled by the *old* max, so multiply
     `l` and `o` by `α = exp(m − m′)` before adding the new block.
   - Divide `o` by `l` **once, at the end**.

3. **Causal masking under tiling** — mask by absolute positions
   (`key_pos > query_pos`), and survive **fully masked blocks**: a row whose
   entire block is `−inf` must not produce `exp(−inf − (−inf)) = nan`.

---

## Interface

```python
def attention_naive(q_LK, k_TK, v_TK, causal=False) -> o_LK
def attention_tiled(q_LK, k_TK, v_TK, block=16, causal=False) -> o_LK
```

Shape-suffix naming convention:

```
L = query length   T = key length   K = head dim   C = key block size
```

---

## Behavior Notes / Gotchas

- **Why this is possible at all:** softmax looks global (the denominator
  needs every score), but `(m, l, o)` form an associative summary — merging
  two blocks' summaries gives the summary of their union. That's what lets
  the computation tile.
- **Forgetting `α` is the classic bug.** If the dominant key arrives in a
  *late* block, all earlier accumulation is scaled wrong by `exp(m_old − m_new)`
  — which can be orders of magnitude. `test_rescaling_actually_happens`
  places the dominant key in the last block to force this.
- **Normalize once.** Dividing by the running `l` inside the loop and again at
  the end (or renormalizing per block) gives subtly wrong results that shrink
  with block count — the worst kind of bug to debug.
- **The `−inf` corner.** With causal masking, some (row, block) pairs are
  fully masked. `m` may still be `−inf` after such a block; guard the
  rescale factor (`α = 1` when nothing has been seen) or nans propagate.
- **What the real kernel adds:** an outer loop over query blocks, fusion into
  one CUDA kernel so `s` lives in SRAM (the actual memory win — this NumPy
  version still *computes* per-block scores, it just never stores L × T),
  and a backward pass that recomputes attention instead of storing it.

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_online_attention.py -v
```

| Test | Validates |
|------|-----------|
| `test_matches_naive_any_block_size` | Tiling granularity never changes the answer |
| `test_causal_matches_naive` | Causal mask correct across block boundaries |
| `test_numerically_stable_with_huge_logits` | No overflow at scores ~±500 |
| `test_rescaling_actually_happens` | The α-rescale when the max arrives late |
| `test_causal_first_row_attends_only_to_itself` | Fully masked blocks don't nan |
| `test_attention_output_is_convex_combination` | Output rows stay inside V's hull |

---

## Discussion Questions (interview follow-ups)

- **Memory vs FLOPs** — FlashAttention does *more* arithmetic than naive
  attention (rescaling, recomputation in backward). Why is it still much
  faster on a GPU?
- **Associativity** — write the merge operation for two `(m, l, o)` summaries
  and show it's associative. How does this enable split-K / ring attention
  across devices?
- **Backward pass** — the L × T matrix was never stored. What gets
  recomputed in the backward pass, and what small per-row statistics must be
  saved from the forward?
- **Decode-time variant** — with a single query (L=1) and a long cache,
  how does this become FlashDecoding (parallelize over key blocks, then
  merge summaries)?
- **Precision** — real kernels accumulate in fp32 while inputs are bf16.
  Which of `m`, `l`, `o` most needs the extra precision, and why?
