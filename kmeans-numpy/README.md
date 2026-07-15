# K-Means From Scratch (Vectorized NumPy)

**Problem:** Implement **Lloyd's algorithm** with **k-means++ initialization**
and **empty-cluster repair**, fully vectorized (no Python loop over points).
The classical-ML staple of MLE coding rounds (reported at DeepMind, Meta, and
in nearly every "implement from scratch" interview list) — the bar is not
"does it cluster" but the numerical and edge-case details below.

---

## Core Requirements

1. **Vectorized distances (`pairwise_sq_dists`)**
   - Use the expansion `‖x−c‖² = ‖x‖² − 2x·c + ‖c‖²` — one matmul, no
     `(N, K, D)` broadcast intermediate.
   - Clamp tiny negative results (floating-point cancellation) to 0.

2. **k-means++ init (`kmeans_pp_init`)**
   - First center uniform; each next center sampled ∝ squared distance to
     the nearest chosen center (D² weighting).
   - Handle the degenerate case where all remaining distances are 0.

3. **Lloyd iterations (`kmeans`)**
   - Assignment step, update step, inertia history, convergence on center
     movement < tol.
   - **Empty clusters happen.** Reseed a dead centroid at the point farthest
     from its assigned center (splits the worst-served region) rather than
     leaving it or crashing on `mean` of an empty slice.
   - All randomness through an injected `Generator` — same seed, same result.

---

## Interface

```python
def pairwise_sq_dists(x_ND, c_KD) -> d_NK
def kmeans_pp_init(x_ND, k, rng) -> c_KD
def kmeans(x_ND, k, rng, init="++", max_iters=100, tol=1e-9)
    -> (centers_KD, labels_N, inertia_history)
```

Shape-suffix naming convention:

```
N = points      D = feature dim      K = clusters
```

---

## Behavior Notes / Gotchas

- **Inertia must be monotone non-increasing.** Both steps can only lower it:
  assignment moves each point to a closer center; the mean minimizes summed
  squared distance to members. A history that ever ticks up is the fastest
  possible correctness check — it's a test here.
- **The matmul distance trick can go slightly negative** for a point equal to
  a center (`a − 2b + c` cancellation). Downstream `sqrt` or `choice(p=...)`
  then explodes. Clamp.
- **Empty-cluster repair matters on real data.** With random init on
  imbalanced blobs a centroid routinely ends up owning nothing; `np.mean` of
  an empty slice returns NaN and poisons everything after.
- **Label permutation.** Cluster ids are arbitrary — a correct test compares
  *partitions* (or uses Hungarian matching), never raw label equality.
- **Evaluating init quality is statistical.** k-means++ is better *in
  expectation*; the test averages over 15 seeds rather than asserting a
  single run.

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_kmeans.py -v
```

| Test | Validates |
|------|-----------|
| `test_pairwise_dists_match_naive` | Matmul trick == broadcast reference |
| `test_inertia_monotonically_decreases` | The Lloyd invariant |
| `test_recovers_separated_blobs` | Partition (not label) equality on easy data |
| `test_kmeanspp_beats_random_init_on_average` | D² seeding wins statistically |
| `test_empty_cluster_is_reseeded` | No dead centroids / NaN means |
| `test_deterministic_given_seed` | Injected RNG, reproducible |
| `test_pp_init_spreads_centers` | ++ lands one center per blob |

---

## Discussion Questions (interview follow-ups)

- **Complexity** — O(NKD) per iteration. What changes with mini-batch
  k-means, and when is the approximation acceptable?
- **Choosing k** — elbow vs. silhouette vs. gap statistic; why is inertia
  alone insufficient (it always decreases in k)?
- **Failure modes** — non-spherical clusters, unequal variances, mismatched
  scales. Which are fixed by standardization vs. needing GMM/spectral?
- **k-means as compression** — the connection to vector quantization and
  product quantization in ANN indexes (FAISS IVF-PQ) — this exact code is
  how retrieval indexes get built.
- **Local optima** — why do warm restarts (best of n inits) remain standard
  even with k-means++?
