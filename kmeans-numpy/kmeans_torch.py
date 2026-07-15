"""PyTorch port of k-means (see kmeans.py for the NumPy original and full
commentary)."""

import torch


def pairwise_sq_dists(x_ND, c_KD):
    x2_N1 = (x_ND ** 2).sum(1, keepdim=True)
    c2_K = (c_KD ** 2).sum(1)
    return (x2_N1 - 2.0 * x_ND @ c_KD.T + c2_K).clamp(min=0.0)


def kmeans_pp_init(x_ND, k, generator):
    n = x_ND.shape[0]
    centers = [x_ND[int(torch.randint(n, (1,), generator=generator))]]
    for _ in range(k - 1):
        d2_N = pairwise_sq_dists(x_ND, torch.stack(centers)).min(dim=1).values
        if float(d2_N.sum()) == 0:
            centers.append(x_ND[int(torch.randint(n, (1,), generator=generator))])
            continue
        idx = int(torch.multinomial(d2_N / d2_N.sum(), 1, generator=generator))
        centers.append(x_ND[idx])
    return torch.stack(centers)


def kmeans(x_ND, k, generator, init="++", max_iters=100, tol=1e-9):
    if init == "++":
        c_KD = kmeans_pp_init(x_ND, k, generator)
    else:
        idx = torch.randperm(x_ND.shape[0], generator=generator)[:k]
        c_KD = x_ND[idx].clone()

    history, labels_N = [], None
    for _ in range(max_iters):
        d_NK = pairwise_sq_dists(x_ND, c_KD)
        labels_N = d_NK.argmin(dim=1)
        history.append(float(d_NK[torch.arange(len(labels_N)), labels_N].sum()))

        new_c_KD = c_KD.clone()
        for j in range(k):
            members = labels_N == j
            if bool(members.any()):
                new_c_KD[j] = x_ND[members].mean(dim=0)
            else:
                new_c_KD[j] = x_ND[int(d_NK.min(dim=1).values.argmax())]

        if float(torch.linalg.norm(new_c_KD - c_KD)) < tol:
            c_KD = new_c_KD
            break
        c_KD = new_c_KD

    return c_KD, labels_N, history
