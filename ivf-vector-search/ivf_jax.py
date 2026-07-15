"""JAX port of the IVF index (see ivf.py for the NumPy original and full
commentary). Index structures (per-list id arrays) stay host-side python;
distance math is jnp."""

import jax
import jax.numpy as jnp
import numpy as np


def sq_dists(a_ND, b_MD):
    a2 = (a_ND ** 2).sum(1, keepdims=True)
    b2 = (b_MD ** 2).sum(1)
    return jnp.maximum(a2 - 2.0 * a_ND @ b_MD.T + b2, 0.0)


class IVFIndex:
    def __init__(self, n_lists, n_train_iters=15, seed=0):
        self.n_lists = n_lists
        self.n_train_iters = n_train_iters
        self.key = jax.random.key(seed)
        self.centroids_CD = None
        self.lists = None

    def train(self, x_ND):
        self.key, sub = jax.random.split(self.key)
        idx = jax.random.choice(sub, len(x_ND), (self.n_lists,), replace=False)
        c_CD = x_ND[idx]
        for _ in range(self.n_train_iters):
            assign_N = sq_dists(x_ND, c_CD).argmin(axis=1)
            rows = []
            for j in range(self.n_lists):
                members = assign_N == j
                if bool(members.any()):
                    w = members.astype(x_ND.dtype)[:, None]
                    rows.append((x_ND * w).sum(0) / w.sum())
                else:
                    rows.append(c_CD[j])
            c_CD = jnp.stack(rows)
        self.centroids_CD = c_CD

    def add(self, x_ND):
        assert self.centroids_CD is not None, "train() before add()"
        assign_N = np.asarray(sq_dists(x_ND, self.centroids_CD).argmin(axis=1))
        self.lists = []
        for j in range(self.n_lists):
            ids = np.nonzero(assign_N == j)[0]
            self.lists.append((ids, x_ND[ids]))

    def search(self, q_D, k, nprobe=1):
        q_1D = q_D[None, :]
        probe_order = np.asarray(
            sq_dists(q_1D, self.centroids_CD)[0].argsort())[:nprobe]
        cand_ids, cand_vecs = [], []
        for j in probe_order:
            ids, vecs = self.lists[j]
            if len(ids):
                cand_ids.append(ids)
                cand_vecs.append(vecs)
        ids = np.concatenate(cand_ids)
        vecs = jnp.concatenate(cand_vecs)
        d = np.asarray(sq_dists(q_1D, vecs)[0])
        top = d.argsort()[:k]
        return ids[top], d[top], len(ids)


def brute_force(x_ND, q_D, k):
    d = np.asarray(sq_dists(q_D[None, :], x_ND)[0])
    top = d.argsort()[:k]
    return top, d[top]
