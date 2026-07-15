"""JAX port of k-means (see kmeans.py for the NumPy original and full
commentary). Randomness threads explicit PRNG keys."""

import jax
import jax.numpy as jnp


def pairwise_sq_dists(x_ND, c_KD):
    x2_N1 = (x_ND ** 2).sum(1, keepdims=True)
    c2_K = (c_KD ** 2).sum(1)
    return jnp.maximum(x2_N1 - 2.0 * x_ND @ c_KD.T + c2_K, 0.0)


def kmeans_pp_init(x_ND, k, key):
    n = x_ND.shape[0]
    key, sub = jax.random.split(key)
    centers = [x_ND[int(jax.random.randint(sub, (), 0, n))]]
    for _ in range(k - 1):
        d2_N = pairwise_sq_dists(x_ND, jnp.stack(centers)).min(axis=1)
        key, sub = jax.random.split(key)
        if float(d2_N.sum()) == 0:
            centers.append(x_ND[int(jax.random.randint(sub, (), 0, n))])
            continue
        idx = int(jax.random.choice(sub, n, p=d2_N / d2_N.sum()))
        centers.append(x_ND[idx])
    return jnp.stack(centers)


def kmeans(x_ND, k, key, init="++", max_iters=100, tol=1e-9):
    key, sub = jax.random.split(key)
    if init == "++":
        c_KD = kmeans_pp_init(x_ND, k, sub)
    else:
        idx = jax.random.choice(sub, x_ND.shape[0], (k,), replace=False)
        c_KD = x_ND[idx]

    history, labels_N = [], None
    for _ in range(max_iters):
        d_NK = pairwise_sq_dists(x_ND, c_KD)
        labels_N = d_NK.argmin(axis=1)
        history.append(float(d_NK[jnp.arange(len(labels_N)), labels_N].sum()))

        new_c = []
        for j in range(k):
            members = labels_N == j
            if bool(members.any()):
                # Masked mean (jnp boolean indexing has dynamic shape).
                w = members.astype(x_ND.dtype)[:, None]
                new_c.append((x_ND * w).sum(0) / w.sum())
            else:
                new_c.append(x_ND[int(d_NK.min(axis=1).argmax())])
        new_c_KD = jnp.stack(new_c)

        if float(jnp.linalg.norm(new_c_KD - c_KD)) < tol:
            c_KD = new_c_KD
            break
        c_KD = new_c_KD

    return c_KD, labels_N, history
