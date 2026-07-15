"""GRPO / PPO policy-gradient losses for LLM RL (RLHF / reasoning RL).

GRPO (Shao et al. 2024, DeepSeekMath): sample a GROUP of completions per
prompt, use the group-normalized reward as the advantage — no value network.
The per-token update is the standard PPO clipped surrogate, plus a KL
penalty against a reference policy estimated with the low-variance k3
estimator.

Shape suffix convention:
  B = completions in batch, T = completion length (tokens)
"""

import torch


def masked_mean(x_BT, mask_BT):
    return (x_BT * mask_BT).sum() / mask_BT.sum().clamp(min=1)


def grpo_advantages(rewards_B, group_ids_B, eps=1e-4):
    """Advantage = reward standardized WITHIN its prompt group.

    A group is all completions sampled from the same prompt. Standardizing
    inside the group removes prompt difficulty as a confounder — the same
    role a learned value baseline plays in PPO, with zero extra parameters.
    """
    adv_B = torch.empty_like(rewards_B)
    for g in group_ids_B.unique():
        m = group_ids_B == g
        r = rewards_B[m]
        adv_B[m] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv_B


def ppo_clip_loss(logp_BT, logp_old_BT, adv_B, mask_BT, clip_eps=0.2):
    """Clipped surrogate, token-level, sequence advantage broadcast to tokens.

    ratio = exp(logp_new - logp_old). Clipping removes the incentive to move
    the ratio outside [1-eps, 1+eps]: beyond the clip point in the favored
    direction the objective is constant, so the gradient is exactly zero.
    """
    ratio_BT = (logp_BT - logp_old_BT).exp()
    adv_BT = adv_B[:, None].expand_as(ratio_BT)
    unclipped_BT = ratio_BT * adv_BT
    clipped_BT = ratio_BT.clamp(1 - clip_eps, 1 + clip_eps) * adv_BT
    return -masked_mean(torch.minimum(unclipped_BT, clipped_BT), mask_BT)


def k3_kl(logp_BT, logp_ref_BT):
    """Per-token k3 estimator of KL(pi || pi_ref):  r - 1 - log r,
    with r = pi_ref/pi. Unlike the naive (logp - logp_ref) sample estimate,
    k3 is nonnegative for every sample and has much lower variance
    (Schulman, 2020, "Approximating KL Divergence").
    """
    logr_BT = logp_ref_BT - logp_BT
    return logr_BT.exp() - 1 - logr_BT


def grpo_loss(logp_BT, logp_old_BT, logp_ref_BT, rewards_B, group_ids_B,
              mask_BT, clip_eps=0.2, kl_coef=0.04):
    """Full GRPO objective: clipped policy loss + beta * KL(pi || pi_ref).

    logp_old (behavior policy at sampling time) anchors the ratio; logp_ref
    (frozen SFT/reference model) anchors the KL. They are different models
    after the first optimizer epoch — conflating them is a classic bug.
    """
    adv_B = grpo_advantages(rewards_B, group_ids_B)
    pg = ppo_clip_loss(logp_BT, logp_old_BT, adv_B, mask_BT, clip_eps)
    kl = masked_mean(k3_kl(logp_BT, logp_ref_BT), mask_BT)
    return pg + kl_coef * kl, {"pg_loss": pg.item(), "kl": kl.item()}
