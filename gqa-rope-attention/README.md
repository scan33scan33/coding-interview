# Grouped-Query Attention with RoPE and a KV Cache

**Problem:** Implement the attention block used by modern decoder LLMs
(Llama, Mistral, Qwen, ...): **causal grouped-query attention (GQA)** with
**rotary position embeddings (RoPE)** and an **incremental KV cache** for
autoregressive decoding.

The must-pass correctness bar: decoding one token at a time through the cache
produces *exactly* the same outputs as a single full-sequence forward pass.

---

## Core Requirements

1. **RoPE (`rope_tables`, `apply_rope`)**
   - Precompute cos/sin tables from the standard `base^(-2j/K)` frequencies.
   - Rotate consecutive (even, odd) feature pairs by a position-dependent angle.
   - Support a **position `offset`** — during incremental decoding the new token
     is at absolute position `cache_len`, not 0.

2. **GQA (`GQAttention`)**
   - `H` query heads share `G` key/value heads (`H % G == 0`).
     `G == H` recovers vanilla MHA; `G == 1` is multi-query attention.
   - Projections sized accordingly: `wq: D → H·K`, `wk/wv: D → G·K`.
   - Expand kv heads with `repeat_interleave` so each group of `H/G` query
     heads reads the same kv head.

3. **KV cache**
   - `forward(x, cache)` appends this call's keys/values to the cache and
     returns `(out, new_cache)`.
   - The cache stores **G** heads (that's the memory win of GQA).
   - RoPE is applied **before** caching, so cached keys never need re-rotation.

4. **Causal mask with offset**
   - Query at absolute position `offset + i` may attend to keys `0 .. offset+i`.
   - A plain `torch.triu` of size `L × L` is wrong once a cache exists —
     compare absolute query positions against absolute key positions.

---

## Interface

```python
class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_len=2048): ...
    def forward(self, x_BLD, cache=None) -> (out_BLD, (k_BGTK, v_BGTK)): ...

def rope_tables(head_dim, max_len, base=10000.0) -> (cos_TJ, sin_TJ)
def apply_rope(x_BhLK, cos_TJ, sin_TJ, offset=0) -> x_BhLK
```

Shape-suffix naming convention:

```
B = batch   L = new (query) length   T = total length (cache + new)
H = query heads   G = kv heads   K = head dim   D = model dim (H·K)
```

---

## Behavior Notes / Gotchas

- **RoPE is a norm-preserving rotation.** Each feature pair is rotated in 2D;
  vector norms are unchanged, and `⟨rope(q, i), rope(k, j)⟩` depends only on
  the *relative* offset `i − j`. Both properties are tested.
- **The offset bug is the classic one.** Forget to shift RoPE positions (or
  the causal mask) by the cache length and single-step decoding silently
  diverges from the full forward — generation quality degrades with no error.
- **Cache before or after RoPE?** Cache *rotated* keys. If you cached raw keys
  you'd have to re-rotate the whole cache every step (or store both), and
  queries/keys would disagree on positions.
- **`repeat_interleave` vs `repeat`.** Head order matters: query heads
  `[0..H/G)` must map to kv head 0, etc. `repeat` interleaves them wrongly and
  still runs — only numerics catch it.
- **Why GQA at all?** At inference the KV cache dominates memory
  (`B·T·G·K·2·layers`), and decode is memory-bandwidth-bound on reading it.
  Cutting `G` shrinks the cache and speeds decoding with minor quality loss.

---

## Running the Smoke Test

```bash
pip install torch pytest
python -m pytest test_gqa_attention.py -v
```

| Test | Validates |
|------|-----------|
| `test_rope_preserves_norm` | RoPE is a pure rotation |
| `test_rope_is_relative` | q·k after RoPE depends only on relative position |
| `test_gqa_equals_mha_when_groups_equal_heads` | `G == H` degenerates to MHA |
| `test_causal_masking` | Future tokens never affect past outputs |
| `test_kv_cache_matches_full_forward` | Incremental decode ≡ full forward |
| `test_cache_shapes_use_kv_heads` | Cache stores G heads, not H |

---

## Discussion Questions (interview follow-ups)

- **MQA vs GQA vs MHA** — quantify KV-cache memory for a 70B model at 8k
  context. Why did Llama-2-70B pick G=8?
- **RoPE extrapolation** — why does RoPE degrade beyond the training length,
  and how do position-interpolation / NTK-aware scaling / YaRN address it?
- **Paged attention** — how does vLLM's block-table cache layout change this
  implementation?
- **Sliding-window attention** — what changes in the mask and cache to support
  Mistral-style windows, and how do you evict?
