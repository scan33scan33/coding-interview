"""IVF (inverted-file) approximate nearest-neighbor index — the coarse
quantization scheme inside FAISS's IVF-Flat: cluster the corpus with k-means,
store each vector in its nearest centroid's list, and at query time search
only the `nprobe` closest lists exactly.

Shape suffix convention:
  N = corpus vectors, D = dim, C = centroids/lists, Q = queries
"""

import numpy as np


def sq_dists(a_ND, b_MD):
    a2 = (a_ND ** 2).sum(1, keepdims=True)
    b2 = (b_MD ** 2).sum(1)
    return np.maximum(a2 - 2.0 * a_ND @ b_MD.T + b2, 0.0)


class IVFIndex:
    def __init__(self, n_lists, n_train_iters=15, seed=0):
        self.n_lists = n_lists
        self.n_train_iters = n_train_iters
        self.rng = np.random.default_rng(seed)
        self.centroids_CD = None
        self.lists = None        # list of (ids array, vectors array)

    def train(self, x_ND):
        """Learn the coarse quantizer (plain Lloyd's; ++-init omitted for
        brevity — the retrieval math is the point here)."""
        idx = self.rng.choice(len(x_ND), size=self.n_lists, replace=False)
        c_CD = x_ND[idx].copy()
        for _ in range(self.n_train_iters):
            assign_N = sq_dists(x_ND, c_CD).argmin(axis=1)
            for j in range(self.n_lists):
                members = assign_N == j
                if members.any():
                    c_CD[j] = x_ND[members].mean(axis=0)
        self.centroids_CD = c_CD

    def add(self, x_ND):
        assert self.centroids_CD is not None, "train() before add()"
        assign_N = sq_dists(x_ND, self.centroids_CD).argmin(axis=1)
        self.lists = []
        for j in range(self.n_lists):
            ids = np.nonzero(assign_N == j)[0]
            self.lists.append((ids, x_ND[ids]))

    def search(self, q_D, k, nprobe=1):
        """Probe the nprobe nearest lists; exact scoring inside them.

        Returns (ids, dists, n_scanned). n_scanned / N is the fraction of
        the corpus touched — the whole point of the index.
        """
        q_1D = q_D[None, :]
        probe_order = sq_dists(q_1D, self.centroids_CD)[0].argsort()[:nprobe]
        cand_ids, cand_vecs = [], []
        for j in probe_order:
            ids, vecs = self.lists[j]
            if len(ids):
                cand_ids.append(ids)
                cand_vecs.append(vecs)
        ids = np.concatenate(cand_ids)
        vecs = np.concatenate(cand_vecs)
        d = sq_dists(q_1D, vecs)[0]
        top = d.argsort()[:k]
        return ids[top], d[top], len(ids)


def brute_force(x_ND, q_D, k):
    d = sq_dists(q_D[None, :], x_ND)[0]
    top = d.argsort()[:k]
    return top, d[top]


def recall_at_k(found_ids, true_ids):
    return len(set(found_ids) & set(true_ids)) / len(true_ids)
