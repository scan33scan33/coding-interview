import torch
import torch.nn.functional as F

# Shape suffix convention:
#   B = batch, L = sequence length, V = vocab size
# A tensor's name tells you its shape. If an op's output name doesn't
# follow from its input names, the op is probably wrong.


def sequence_logprob(logits_BLV, labels_BL, mask_BL):
    """Sum of per-token log-probs over the completion, per sequence.

    logits_BLV : model outputs, already shifted so logits[:, t] predicts
                 labels[:, t] (i.e. pass logits[:, :-1] and labels[:, 1:]).
    labels_BL  : target token ids.
    mask_BL    : 1.0 on completion tokens, 0.0 on prompt and padding tokens.
                 Only completion tokens contribute — DPO compares responses,
                 not prompts.
    """
    logp_BLV = F.log_softmax(logits_BLV, dim=-1)
    tok_logp_BL = logp_BLV.gather(-1, labels_BL.unsqueeze(-1)).squeeze(-1)
    return (tok_logp_BL * mask_BL).sum(-1)


def dpo_loss(policy_chosen_B, policy_rejected_B,
             ref_chosen_B, ref_rejected_B,
             beta=0.1, label_smoothing=0.0):
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    Inputs are summed completion log-probs (from `sequence_logprob`) under the
    trainable policy and the frozen reference model.

    Returns (loss, chosen_rewards_B, rejected_rewards_B); the rewards are the
    implicit reward model beta * log(pi/ref), detached for logging.
    """
    chosen_logratio_B = policy_chosen_B - ref_chosen_B
    rejected_logratio_B = policy_rejected_B - ref_rejected_B
    margin_B = beta * (chosen_logratio_B - rejected_logratio_B)

    # label_smoothing = probability that the preference label is flipped
    # (conservative / cDPO). At 0 this is plain -logsigmoid(margin).
    loss_B = (-F.logsigmoid(margin_B) * (1 - label_smoothing)
              - F.logsigmoid(-margin_B) * label_smoothing)

    chosen_rewards_B = (beta * chosen_logratio_B).detach()
    rejected_rewards_B = (beta * rejected_logratio_B).detach()
    return loss_B.mean(), chosen_rewards_B, rejected_rewards_B


def dpo_step(policy, ref, batch, beta=0.1):
    """One DPO forward: score chosen and rejected under both models.

    batch: dict with input ids, and completion masks for the
    chosen and rejected sequences. The reference model runs under no_grad —
    it is frozen and only anchors the KL constraint.
    """
    def score(model, ids_BL, mask_BL):
        logits_BLV = model(ids_BL)
        return sequence_logprob(logits_BLV[:, :-1], ids_BL[:, 1:], mask_BL[:, 1:])

    policy_chosen_B = score(policy, batch["chosen_ids"], batch["chosen_mask"])
    policy_rejected_B = score(policy, batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():
        ref_chosen_B = score(ref, batch["chosen_ids"], batch["chosen_mask"])
        ref_rejected_B = score(ref, batch["rejected_ids"], batch["rejected_mask"])

    return dpo_loss(policy_chosen_B, policy_rejected_B,
                    ref_chosen_B, ref_rejected_B, beta=beta)
