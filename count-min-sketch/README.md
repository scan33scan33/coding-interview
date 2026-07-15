# Count-Min Sketch & Streaming Heavy Hitters

**Problem:** Implement a **count-min sketch** — approximate frequency counts
over an unbounded stream in **sublinear memory** — plus a **heavy hitters**
tracker (top-k most frequent items) on top of it. The applied/systems-flavored
coding question reported in OpenAI-style "practical coding" rounds
("count token frequencies over a firehose you can't hold in RAM").

**The guarantee to implement correctly** (Cormode & Muthukrishnan 2005):
with width `w = ⌈e/ε⌉` and depth `d = ⌈ln(1/δ)⌉`,

```
true  ≤  estimate  ≤  true + ε·N     with probability ≥ 1 − δ
```

Estimates **never undercount** — that one-sided error is what makes the
min-across-rows trick work and what every test leans on.

---

## Core Requirements

1. **The sketch (`CountMinSketch`)**
   - `d × w` counter table; one hash function per row from a **universal
     family** (`((a·x + b) mod p) mod w`, Mersenne prime p = 2⁶¹−1) — not
     `hash(key) + row`, which correlates rows and voids the analysis.
   - `add(key, count)` increments one cell per row; `query(key)` returns the
     **min** across rows (each row only ever overcounts, so min is tightest).
   - **Conservative update** option: only raise counters to
     `min(cells) + count` — strictly less overestimation, same
     never-undercount guarantee.

2. **Heavy hitters (`HeavyHitters`)**
   - Track candidate top-k with a bounded map; on each arrival, re-query the
     sketch and evict the smallest tracked entry if beaten.
   - Memory: `O(w·d + k)` — never `O(#distinct keys)`.

---

## Behavior Notes / Gotchas

- **Testing a probabilistic bound needs statistics, not point asserts.** The
  `ε·N` bound holds *per key with probability 1−δ*; the test checks the
  violation *fraction* is ≤ δ-ish, not that no key violates.
- **Conservative update sandwich:** `true ≤ conservative ≤ standard` for
  every key (tested). The tradeoff: conservative sketches can't be merged by
  cell-wise addition — a dealbreaker for distributed aggregation.
- **Row independence is load-bearing.** The failure probability δ comes from
  each row being an independent chance to avoid collisions. Deriving rows
  from one hash (`h+0, h+1, ...`) makes them collide together.
- **Why min and not mean/median?** Errors are one-sided (counters only grow),
  so the minimum is simultaneously the tightest and still an overestimate.
  Count-*sketch* (the ± variant) has two-sided error and uses the median.
- **Heavy hitters re-queries on every arrival** because a tracked key's
  estimate changes as the stream evolves; caching the entry-time estimate
  under-ranks keys that heat up later.

---

## Running the Smoke Test

```bash
pip install pytest
python -m pytest test_cms.py -v
```

| Test | Validates |
|------|-----------|
| `test_never_undercounts` | The one-sided error guarantee |
| `test_error_bound_holds` | `≤ true + ε·N` for ≥ 1−δ of keys |
| `test_conservative_update_dominates_standard` | `true ≤ conservative ≤ standard` |
| `test_weighted_adds` | Weighted increments |
| `test_memory_independent_of_cardinality` | 50k distinct keys, fixed table |
| `test_heavy_hitters_finds_the_head` | Zipf head recovered |
| `test_heavy_hitters_bounded_memory` | Candidate set stays ≤ k |

---

## Discussion Questions (interview follow-ups)

- **Derive the bound.** Expected collision mass in one cell is N/w; Markov
  gives P(error > ε·N) ≤ 1/e per row; independence across d rows gives δ.
  Walk through it.
- **CMS vs Count-Sketch vs Misra-Gries** — error type, memory, mergeability.
  When is deterministic Misra-Gries strictly better?
- **Distributed streams** — why does cell-wise addition merge standard
  sketches exactly, and how do you shard by key vs replicate the sketch?
- **Where this shows up in ML** — feature hashing, n-gram counting for
  tokenizer training, deduplication heuristics, and DP noise on sketch cells.
- **Decay** — how do you adapt CMS for sliding windows or exponential decay
  (hint: timestamped counters or hokusai-style shifted sketches)?
