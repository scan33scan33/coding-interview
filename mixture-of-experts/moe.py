"""Mixture-of-Experts layer: top-k gating, load-balancing auxiliary loss,
and capacity-limited dispatch with token dropping.

The architecture behind Mixtral, DeepSeek-V3, Switch Transformer, etc.

Shape suffix convention:
  N = tokens (batch*seq flattened), D = model dim, E = experts, K = top-k
"""

import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x_ND):
        return self.net(x_ND)


def load_balancing_loss(gates_NE, expert_index_NK):
    """Switch-Transformer auxiliary loss:  E * sum_e f_e * P_e

    f_e = fraction of tokens whose TOP-1 choice is expert e (hard counts)
    P_e = mean gate probability for expert e (soft, differentiable)

    Minimized (value 1.0) when routing is perfectly uniform. The hard f_e
    has no gradient; the loss pushes the soft P_e toward uniform wherever
    the hard counts are skewed.
    """
    n, e = gates_NE.shape
    f_E = torch.zeros(e, device=gates_NE.device)
    top1_N = expert_index_NK[:, 0]
    f_E.scatter_add_(0, top1_N, torch.ones_like(top1_N, dtype=f_E.dtype))
    f_E = f_E / n
    p_E = gates_NE.mean(dim=0)
    return e * (f_E * p_E).sum()


class MoELayer(nn.Module):
    def __init__(self, dim, n_experts, hidden=None, k=2, capacity_factor=None):
        super().__init__()
        self.n_experts, self.k = n_experts, k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(dim, hidden or 4 * dim) for _ in range(n_experts))

    def forward(self, x_ND):
        """Returns (out_ND, aux_loss, n_dropped).

        Router picks top-k experts per token; their gate weights are
        RENORMALIZED over the chosen k so each token's mixture sums to 1.
        With a capacity factor, each expert processes at most
          cap = ceil(capacity_factor * N * k / E)
        assignments; overflow assignments are dropped — the token loses that
        expert's contribution (a residual connection outside this layer is
        what keeps dropped tokens from being zeroed entirely).
        """
        n, d = x_ND.shape
        gates_NE = F.softmax(self.router(x_ND), dim=-1)
        topv_NK, topi_NK = gates_NE.topk(self.k, dim=-1)
        weight_NK = topv_NK / topv_NK.sum(dim=-1, keepdim=True)

        cap = None
        if self.capacity_factor is not None:
            cap = int(torch.ceil(torch.tensor(
                self.capacity_factor * n * self.k / self.n_experts)))

        out_ND = torch.zeros_like(x_ND)
        n_dropped = 0
        for e in range(self.n_experts):
            tok_idx, slot_idx = (topi_NK == e).nonzero(as_tuple=True)
            if cap is not None and tok_idx.numel() > cap:
                # Keep the first `cap` assignments in token order; drop rest.
                n_dropped += tok_idx.numel() - cap
                tok_idx, slot_idx = tok_idx[:cap], slot_idx[:cap]
            if tok_idx.numel() == 0:
                continue
            expert_out = self.experts[e](x_ND[tok_idx])
            out_ND.index_add_(
                0, tok_idx, expert_out * weight_NK[tok_idx, slot_idx, None])

        aux = load_balancing_loss(gates_NE, topi_NK)
        return out_ND, aux, n_dropped
