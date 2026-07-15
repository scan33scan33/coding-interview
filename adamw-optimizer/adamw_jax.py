"""JAX port of AdamW (see adamw.py for the PyTorch original).

Optax-style functional optimizer: init(params) -> state; update(grads,
state, params) -> (new_params, new_state). Pure functions over pytrees, so
the whole train step jits. The parity bar is the same: match
torch.optim.AdamW step-for-step (cross-checked in the tests).
"""

import jax
import jax.numpy as jnp

# warmup_cosine_lr is pure math — shared verbatim with the torch version.
from adamw import warmup_cosine_lr  # noqa: F401  (re-exported)


def adamw_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params),
            "t": jnp.zeros((), dtype=jnp.int32)}


def adamw_update(grads, state, params, lr=1e-3, b1=0.9, b2=0.999,
                 eps=1e-8, weight_decay=0.01):
    t = state["t"] + 1
    bc1 = 1 - b1 ** t.astype(jnp.float32)
    bc2 = 1 - b2 ** t.astype(jnp.float32)

    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g,
                               state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g,
                               state["v"], grads)

    def step(p, m_, v_):
        p = p * (1 - lr * weight_decay)          # decoupled decay FIRST
        return p - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps)

    new_params = jax.tree_util.tree_map(step, params, m, v)
    return new_params, {"m": m, "v": v, "t": t}


def clip_grad_global_norm(grads, max_norm):
    """Returns (clipped_grads, pre_clip_norm). One global scale factor."""
    leaves = jax.tree_util.tree_leaves(grads)
    total = jnp.sqrt(sum((g ** 2).sum() for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total + 1e-12))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), total
