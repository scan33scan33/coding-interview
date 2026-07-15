"""JAX port of the GRPO/PPO objective (see grpo.py for the PyTorch original).

Group advantages are computed with a segment-mean formulation (scatter-style
ops) instead of a python loop over unique ids, so the whole loss is jittable.
"""

import jax
import jax.numpy as jnp


def masked_mean(x_BT, mask_BT):
    return (x_BT * mask_BT).sum() / jnp.maximum(mask_BT.sum(), 1.0)


def grpo_advantages(rewards_B, group_ids_B, n_groups, eps=1e-4):
    """Standardize rewards within each group, vectorized via segment sums.

    n_groups must be static (a python int) for jit-ability — in real
    training it's just batch_size // group_size.
    """
    count_G = jax.ops.segment_sum(jnp.ones_like(rewards_B), group_ids_B, n_groups)
    sum_G = jax.ops.segment_sum(rewards_B, group_ids_B, n_groups)
    mean_G = sum_G / jnp.maximum(count_G, 1.0)
    centered_B = rewards_B - mean_G[group_ids_B]
    var_G = jax.ops.segment_sum(centered_B ** 2, group_ids_B, n_groups) \
        / jnp.maximum(count_G, 1.0)
    std_G = jnp.sqrt(var_G)                      # biased, matching torch impl
    return centered_B / (std_G[group_ids_B] + eps)


def ppo_clip_loss(logp_BT, logp_old_BT, adv_B, mask_BT, clip_eps=0.2):
    ratio_BT = jnp.exp(logp_BT - logp_old_BT)
    adv_BT = adv_B[:, None] * jnp.ones_like(ratio_BT)
    unclipped_BT = ratio_BT * adv_BT
    clipped_BT = jnp.clip(ratio_BT, 1 - clip_eps, 1 + clip_eps) * adv_BT
    return -masked_mean(jnp.minimum(unclipped_BT, clipped_BT), mask_BT)


def k3_kl(logp_BT, logp_ref_BT):
    logr_BT = logp_ref_BT - logp_BT
    return jnp.exp(logr_BT) - 1 - logr_BT


def grpo_loss(logp_BT, logp_old_BT, logp_ref_BT, rewards_B, group_ids_B,
              n_groups, mask_BT, clip_eps=0.2, kl_coef=0.04):
    adv_B = grpo_advantages(rewards_B, group_ids_B, n_groups)
    pg = ppo_clip_loss(logp_BT, logp_old_BT, adv_B, mask_BT, clip_eps)
    kl = masked_mean(k3_kl(logp_BT, logp_ref_BT), mask_BT)
    return pg + kl_coef * kl, {"pg_loss": pg, "kl": kl}
