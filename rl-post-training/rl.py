"""Stage 2: RL post-training with a verifiable reward — GRPO or PPO.

Both algorithms share the rollout/reward/clip machinery and differ only in
where the baseline comes from:
  GRPO: group-relative — sample G completions per prompt, standardize the
        rewards within the group. No value network.
  PPO:  a learned critic (value head on the '=' position) predicts the
        expected reward of the prompt; advantage = reward - V(prompt).

Shape suffixes: B rollouts (P prompts x G samples), T answer tokens (4).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from task import ANSWER_LEN, PROMPT_LEN, evaluate, prompts_tensor, reward_fn


@torch.no_grad()
def sample_rollouts(model, pairs, temperature=1.0, generator=None):
    """Sample answers; returns (full_ids_BL, gen_ids_BT, logp_old_BT)."""
    ids = prompts_tensor(pairs)
    logps = []
    for _ in range(ANSWER_LEN):
        logits_V = model(ids)[:, -1] / temperature
        probs = F.softmax(logits_V, dim=-1)
        tok_B1 = torch.multinomial(probs, 1, generator=generator)
        logps.append(F.log_softmax(logits_V, dim=-1).gather(-1, tok_B1))
        ids = torch.cat([ids, tok_B1], dim=1)
    return ids, ids[:, PROMPT_LEN:], torch.cat(logps, dim=1)


def answer_logprobs(model, full_ids_BL, return_value=False):
    """Per-token logp of the 4 generated tokens under the current model.
    Position t's logits predict token t+1, so answer tokens (positions 6..9)
    are scored by logits at positions 5..8."""
    if return_value:
        logits_BLV, value_BL = model(full_ids_BL, return_value=True)
    else:
        logits_BLV = model(full_ids_BL)
    logp = F.log_softmax(logits_BLV[:, PROMPT_LEN - 1:-1], dim=-1)
    tok = full_ids_BL[:, PROMPT_LEN:]
    logp_BT = logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    if return_value:
        # V of the prompt state = value head at the '=' token (position 5).
        return logp_BT, value_BL[:, PROMPT_LEN - 1]
    return logp_BT


def grpo_advantages(rewards_B, group_size, eps=1e-4):
    r_PG = rewards_B.view(-1, group_size)
    mean_P1 = r_PG.mean(dim=1, keepdim=True)
    std_P1 = r_PG.std(dim=1, unbiased=False, keepdim=True)
    return ((r_PG - mean_P1) / (std_P1 + eps)).view(-1)


def clipped_pg_loss(logp_BT, logp_old_BT, adv_B, clip_eps=0.2):
    ratio_BT = (logp_BT - logp_old_BT).exp()
    adv_BT = adv_B[:, None].expand_as(ratio_BT)
    return -torch.minimum(ratio_BT * adv_BT,
                          ratio_BT.clamp(1 - clip_eps, 1 + clip_eps) * adv_BT
                          ).mean()


def k3_kl(logp_BT, logp_ref_BT):
    logr = logp_ref_BT - logp_BT
    return (logr.exp() - 1 - logr).mean()


def train_rl(model, train_pairs, eval_pairs, algo="grpo", steps=200,
             prompts_per_step=16, group_size=8, lr=1e-4, clip_eps=0.2,
             kl_coef=0.02, value_coef=0.5, eval_every=25, seed=0):
    """Returns (model, history). history rows:
    {step, eval_acc, mean_reward, kl} — the curve you watch."""
    ref = copy.deepcopy(model)            # frozen SFT reference for the KL
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    history = [{"step": 0, "eval_acc": evaluate(model, eval_pairs),
                "mean_reward": float("nan"), "kl": 0.0}]

    for step in range(1, steps + 1):
        idx = rng.choice(len(train_pairs), size=prompts_per_step)
        pairs = [train_pairs[i] for i in idx for _ in range(group_size)]

        full_ids, gen_ids, logp_old = sample_rollouts(model, pairs, generator=gen)
        rewards_B = reward_fn(pairs, gen_ids)

        if algo == "grpo":
            adv_B = grpo_advantages(rewards_B, group_size)
            logp_BT = answer_logprobs(model, full_ids)
            value_loss = torch.tensor(0.0)
        elif algo == "ppo":
            logp_BT, value_B = answer_logprobs(model, full_ids, return_value=True)
            adv_B = rewards_B - value_B.detach()
            value_loss = F.mse_loss(value_B, rewards_B)
        else:
            raise ValueError(algo)

        with torch.no_grad():
            logp_ref_BT = answer_logprobs(ref, full_ids)
        kl = k3_kl(logp_BT, logp_ref_BT)

        loss = (clipped_pg_loss(logp_BT, logp_old, adv_B, clip_eps)
                + kl_coef * kl + value_coef * value_loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps:
            history.append({"step": step,
                            "eval_acc": evaluate(model, eval_pairs),
                            "mean_reward": float(rewards_B.mean()),
                            "kl": float(kl.detach())})
    return model, history
