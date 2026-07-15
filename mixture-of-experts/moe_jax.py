"""JAX port of the MoE layer (see moe.py for the PyTorch original).

Dense-dispatch formulation: instead of gathering each expert's tokens
(dynamic shapes, which jit can't trace), every expert runs over ALL tokens
and a mask zeroes non-assigned contributions. That's O(E) more FLOPs — fine
for a teaching implementation, and it makes the whole layer jittable; real
JAX MoEs recover sparsity with fixed-capacity gather/scatter buffers.
"""

import jax
import jax.numpy as jnp


def init_expert(key, dim, hidden):
    k1, k2 = jax.random.split(key)
    return {"w1": jax.random.normal(k1, (dim, hidden)) * (2.0 / dim) ** 0.5,
            "b1": jnp.zeros(hidden),
            "w2": jax.random.normal(k2, (hidden, dim)) * (2.0 / hidden) ** 0.5,
            "b2": jnp.zeros(dim)}


def expert_apply(p, x_ND):
    return jax.nn.gelu(x_ND @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def init_moe(key, dim, n_experts, hidden=None, k=2):
    keys = jax.random.split(key, n_experts + 1)
    return {
        "router": jax.random.normal(keys[0], (dim, n_experts)) * dim ** -0.5,
        # Stacked expert params (leading axis = expert) so vmap can run them.
        "experts": jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *[init_expert(keys[i + 1], dim, hidden or 4 * dim)
              for i in range(n_experts)]),
        "k": k, "n_experts": n_experts,
    }


def load_balancing_loss(gates_NE, top1_N, n_experts):
    f_E = jnp.zeros(n_experts).at[top1_N].add(1.0) / gates_NE.shape[0]
    p_E = gates_NE.mean(axis=0)
    return n_experts * (f_E * p_E).sum()


def moe_forward(params, x_ND, capacity_factor=None):
    """Returns (out_ND, aux_loss, n_dropped)."""
    n, d = x_ND.shape
    E, K = params["n_experts"], params["k"]
    gates_NE = jax.nn.softmax(x_ND @ params["router"], axis=-1)
    topv_NK, topi_NK = jax.lax.top_k(gates_NE, K)
    weight_NK = topv_NK / topv_NK.sum(axis=-1, keepdims=True)

    # combine_NE[t, e] = renormalized weight of expert e for token t (0 if
    # not chosen). Dense dispatch multiplies expert outputs by this.
    combine_NE = jnp.zeros((n, E))
    for slot in range(K):
        combine_NE = combine_NE.at[jnp.arange(n), topi_NK[:, slot]].add(
            weight_NK[:, slot])

    n_dropped = 0
    if capacity_factor is not None:
        cap = int(-(-capacity_factor * n * K // E))     # ceil
        # Rank each token's assignment within its expert by token order;
        # assignments past the cap are dropped (weight zeroed).
        for e in range(E):
            chosen_N = combine_NE[:, e] > 0
            rank_N = jnp.cumsum(chosen_N) - 1               # 0-based arrival
            drop_N = chosen_N & (rank_N >= cap)
            n_dropped += int(drop_N.sum())
            combine_NE = combine_NE.at[:, e].set(
                jnp.where(drop_N, 0.0, combine_NE[:, e]))

    all_out_END = jax.vmap(expert_apply, in_axes=(0, None))(params["experts"], x_ND)
    out_ND = jnp.einsum("ne,end->nd", combine_NE, all_out_END)

    aux = load_balancing_loss(gates_NE, topi_NK[:, 0], E)
    return out_ND, aux, n_dropped
