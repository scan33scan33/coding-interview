# IVF Approximate Nearest-Neighbor Index

**Problem:** Implement an **inverted-file (IVF) index** — the coarse
quantization scheme inside FAISS `IVF-Flat` and most production vector
databases: cluster the corpus with k-means, store each vector in its nearest
centroid's inverted list, and at query time **probe only the `nprobe`
closest lists** with exact scoring inside them.

This is the retrieval-infrastructure question behind every RAG system:
"you have 100M embeddings; brute force is too slow — what do you build?"

---

## Core Requirements

1. **Train** — learn `C` centroids with Lloyd's algorithm (the coarse
   quantizer).
2. **Add** — assign every vector to exactly one list (its nearest centroid),
   storing `(ids, vectors)` per list.
3. **Search** — rank centroids by distance to the query, take the top
   `nprobe` lists, score their members exactly, return top-k **plus the
   number of vectors scanned** — `scanned/N` is the speedup knob and must be
   observable.
4. **Evaluation** — `recall@k` against brute force; the accuracy/speed
   tradeoff curve is the deliverable, not a single number.

---

## Behavior Notes / Gotchas

- **`nprobe = C` must equal brute force exactly.** This is the strongest
  correctness test an ANN index has — the approximation must come *only*
  from probing fewer lists, never from scoring errors.
- **The failure mode of `nprobe=1` is boundary queries.** A query midway
  between two centroids has true neighbors straddling both lists; probing
  one list caps recall no matter how good the clustering is. There's a test
  constructing exactly this geometry. This is why real deployments run
  `nprobe` 8–64, not 1.
- **Recall is monotone in `nprobe`** (probing a superset of lists can only
  add candidates) — cheap invariant, worth testing.
- **Every vector lands in exactly one list** — partition, not cover. Losing
  vectors at add time (empty-cluster edge cases) is silent recall loss.
- **Train on representative data.** The quantizer is trained on the corpus
  distribution; heavy distribution shift between train and add degrades
  everything downstream (same lesson as any ML model).

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_ivf.py -v
```

| Test | Validates |
|------|-----------|
| `test_full_probe_equals_brute_force` | Zero approximation with all lists probed |
| `test_recall_increases_with_nprobe` | Monotone recall; high recall at small nprobe |
| `test_small_nprobe_scans_small_fraction` | The index actually skips work |
| `test_every_vector_lands_in_exactly_one_list` | Partition integrity |
| `test_returns_true_neighbor_for_easy_query` | Sanity |
| `test_boundary_query_needs_more_probes` | The nprobe=1 failure geometry |

---

## Discussion Questions (interview follow-ups)

- **Sizing:** the `C ≈ √N` heuristic — derive the per-query cost
  `O(C·D + nprobe·(N/C)·D)` and minimize over C.
- **IVF-PQ:** what does product quantization compress inside each list, what
  are asymmetric distance tables, and why re-rank with exact vectors?
- **IVF vs HNSW:** memory, build time, recall-QPS curves, filtering support,
  and why disk-based systems (DiskANN aside) favor IVF layouts.
- **Deletions and updates** — what breaks when the corpus drifts from the
  trained quantizer, and when do you retrain?
- **GPU batching** — how does the two-stage structure (centroid scan, then
  list scan) map onto batched matmuls?
