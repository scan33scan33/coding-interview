"""PyTorch port of the IVF index (see ivf.py for the NumPy original and
full commentary)."""

import torch


def sq_dists(a_ND, b_MD):
    a2 = (a_ND ** 2).sum(1, keepdim=True)
    b2 = (b_MD ** 2).sum(1)
    return (a2 - 2.0 * a_ND @ b_MD.T + b2).clamp(min=0.0)


class IVFIndex:
    def __init__(self, n_lists, n_train_iters=15, seed=0):
        self.n_lists = n_lists
        self.n_train_iters = n_train_iters
        self.generator = torch.Generator().manual_seed(seed)
        self.centroids_CD = None
        self.lists = None

    def train(self, x_ND):
        idx = torch.randperm(len(x_ND), generator=self.generator)[:self.n_lists]
        c_CD = x_ND[idx].clone()
        for _ in range(self.n_train_iters):
            assign_N = sq_dists(x_ND, c_CD).argmin(dim=1)
            for j in range(self.n_lists):
                members = assign_N == j
                if bool(members.any()):
                    c_CD[j] = x_ND[members].mean(dim=0)
        self.centroids_CD = c_CD

    def add(self, x_ND):
        assert self.centroids_CD is not None, "train() before add()"
        assign_N = sq_dists(x_ND, self.centroids_CD).argmin(dim=1)
        self.lists = []
        for j in range(self.n_lists):
            ids = torch.nonzero(assign_N == j, as_tuple=True)[0]
            self.lists.append((ids, x_ND[ids]))

    def search(self, q_D, k, nprobe=1):
        q_1D = q_D[None, :]
        probe_order = sq_dists(q_1D, self.centroids_CD)[0].argsort()[:nprobe]
        cand_ids, cand_vecs = [], []
        for j in probe_order.tolist():
            ids, vecs = self.lists[j]
            if len(ids):
                cand_ids.append(ids)
                cand_vecs.append(vecs)
        ids = torch.cat(cand_ids)
        vecs = torch.cat(cand_vecs)
        d = sq_dists(q_1D, vecs)[0]
        top = d.argsort()[:k]
        return ids[top], d[top], len(ids)


def brute_force(x_ND, q_D, k):
    d = sq_dists(q_D[None, :], x_ND)[0]
    top = d.argsort()[:k]
    return top, d[top]
