"""K-means from scratch in vectorized NumPy, with k-means++ initialization
and empty-cluster repair.

Shape suffix convention:
  N = points, D = feature dim, K = clusters
"""

import numpy as np


def pairwise_sq_dists(x_ND, c_KD):
    """(N, K) squared euclidean distances, no python loops.

    ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 — one matmul instead of
    broadcasting an (N, K, D) intermediate.
    """
    x2_N1 = (x_ND ** 2).sum(1, keepdims=True)
    c2_K = (c_KD ** 2).sum(1)
    d_NK = x2_N1 - 2.0 * x_ND @ c_KD.T + c2_K
    return np.maximum(d_NK, 0.0)  # clamp tiny negatives from cancellation


def kmeans_pp_init(x_ND, k, rng):
    """k-means++: each new center is sampled with probability proportional to
    squared distance from the nearest already-chosen center. Gives an
    O(log k)-competitive expected initialization vs. optimal (Arthur &
    Vassilvitskii, 2007) — plain random init has no such guarantee.
    """
    n = x_ND.shape[0]
    centers = [x_ND[rng.integers(n)]]
    for _ in range(k - 1):
        d2_N = pairwise_sq_dists(x_ND, np.stack(centers)).min(axis=1)
        if d2_N.sum() == 0:                      # all points already covered
            centers.append(x_ND[rng.integers(n)])
            continue
        centers.append(x_ND[rng.choice(n, p=d2_N / d2_N.sum())])
    return np.stack(centers)


def kmeans(x_ND, k, rng, init="++", max_iters=100, tol=1e-9):
    """Lloyd's algorithm. Returns (centers_KD, labels_N, inertia_history).

    Inertia (sum of squared distances to assigned center) is monotonically
    non-increasing across iterations — both the assignment step and the
    update step can only lower it. If your history ever goes up, the
    implementation is wrong.
    """
    if init == "++":
        c_KD = kmeans_pp_init(x_ND, k, rng)
    else:
        c_KD = x_ND[rng.choice(x_ND.shape[0], size=k, replace=False)]

    history = []
    labels_N = None
    for _ in range(max_iters):
        d_NK = pairwise_sq_dists(x_ND, c_KD)
        labels_N = d_NK.argmin(axis=1)
        history.append(float(d_NK[np.arange(len(labels_N)), labels_N].sum()))

        new_c_KD = c_KD.copy()
        for j in range(k):
            members = labels_N == j
            if members.any():
                new_c_KD[j] = x_ND[members].mean(axis=0)
            else:
                # Empty cluster: reseed at the point farthest from its
                # current center — splits the worst-served region instead
                # of leaving a dead centroid.
                new_c_KD[j] = x_ND[d_NK.min(axis=1).argmax()]

        if np.linalg.norm(new_c_KD - c_KD) < tol:
            c_KD = new_c_KD
            break
        c_KD = new_c_KD

    return c_KD, labels_N, history
