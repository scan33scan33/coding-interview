"""JAX port of the DPO objective (see dpo.py for the PyTorch original).

Pure functions over arrays; the policy/reference models are represented by
their token logits, and dpo_step takes apply-fns + param pytrees, computing
reference scores with stop_gradient (JAX's analogue of torch.no_grad here).
"""

import jax
import jax.numpy as jnp


def sequence_logprob(logits_BLV, labels_BL, mask_BL):
    logp_BLV = jax.nn.log_softmax(logits_BLV, axis=-1)
    tok_logp_BL = jnp.take_along_axis(logp_BLV, labels_BL[..., None], axis=-1)[..., 0]
    return (tok_logp_BL * mask_BL).sum(-1)


def dpo_loss(policy_chosen_B, policy_rejected_B,
             ref_chosen_B, ref_rejected_B,
             beta=0.1, label_smoothing=0.0):
    chosen_logratio_B = policy_chosen_B - ref_chosen_B
    rejected_logratio_B = policy_rejected_B - ref_rejected_B
    margin_B = beta * (chosen_logratio_B - rejected_logratio_B)

    loss_B = (-jax.nn.log_sigmoid(margin_B) * (1 - label_smoothing)
              - jax.nn.log_sigmoid(-margin_B) * label_smoothing)

    chosen_rewards_B = jax.lax.stop_gradient(beta * chosen_logratio_B)
    rejected_rewards_B = jax.lax.stop_gradient(beta * rejected_logratio_B)
    return loss_B.mean(), chosen_rewards_B, rejected_rewards_B


def dpo_step(policy_apply, policy_params, ref_apply, ref_params, batch, beta=0.1):
    def score(apply_fn, params, ids_BL, mask_BL):
        logits_BLV = apply_fn(params, ids_BL)
        return sequence_logprob(logits_BLV[:, :-1], ids_BL[:, 1:], mask_BL[:, 1:])

    policy_chosen_B = score(policy_apply, policy_params,
                            batch["chosen_ids"], batch["chosen_mask"])
    policy_rejected_B = score(policy_apply, policy_params,
                              batch["rejected_ids"], batch["rejected_mask"])
    # Reference model is frozen: block gradients through its scores.
    ref_chosen_B = jax.lax.stop_gradient(
        score(ref_apply, ref_params, batch["chosen_ids"], batch["chosen_mask"]))
    ref_rejected_B = jax.lax.stop_gradient(
        score(ref_apply, ref_params, batch["rejected_ids"], batch["rejected_mask"]))

    return dpo_loss(policy_chosen_B, policy_rejected_B,
                    ref_chosen_B, ref_rejected_B, beta=beta)
