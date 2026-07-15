# GRPO / PPO Loss for LLM Reinforcement Learning

**Problem:** Implement the policy-gradient objective used to RL-train
reasoning LLMs: **GRPO** (DeepSeekMath / R1) — group-normalized advantages
with **no value network** — on top of the standard **PPO clipped surrogate**,
plus a KL penalty against a frozen reference model using the **k3 estimator**.

The hottest post-training topic in current frontier-lab interviews.

---

## Core Requirements

1. **Group-relative advantages (`grpo_advantages`)**
   - A group = all completions sampled from the same prompt.
   - `A = (r − mean_group) / (std_group + ε)` — standardize *within* the
     group only. This replaces PPO's learned value baseline: prompt
     difficulty cancels out with zero extra parameters.

2. **Clipped surrogate (`ppo_clip_loss`)**
   - `ratio = exp(logp_new − logp_old)`, token-level, sequence advantage
     broadcast to tokens, padding masked out of the mean.
   - `-mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))`.
   - Know *why* it works: past the clip point in the favored direction the
     objective is constant ⇒ the gradient is exactly zero (tested).

3. **KL penalty (`k3_kl`)**
   - `k3 = r − 1 − log r` with `r = π_ref/π`, per token.
   - Unlike the naive `logp − logp_ref` sample estimate, k3 is **nonnegative
     for every sample** and much lower variance (Schulman 2020).

4. **Assembly (`grpo_loss`)** — `pg_loss + β·KL`, with the crucial
   distinction: `logp_old` (behavior policy at sampling time, anchors the
   ratio) ≠ `logp_ref` (frozen SFT model, anchors the KL).

---

## Interface

```python
def grpo_advantages(rewards_B, group_ids_B, eps=1e-4) -> adv_B
def ppo_clip_loss(logp_BT, logp_old_BT, adv_B, mask_BT, clip_eps=0.2) -> loss
def k3_kl(logp_BT, logp_ref_BT) -> kl_BT
def grpo_loss(logp_BT, logp_old_BT, logp_ref_BT, rewards_B, group_ids_B,
              mask_BT, clip_eps=0.2, kl_coef=0.04) -> (loss, stats)
```

Shape-suffix naming convention: `B` = completions, `T` = tokens.

---

## Behavior Notes / Gotchas

- **Clipping is one-sided per advantage sign.** With A > 0 and ratio above
  1+ε the gradient is zero, but with A < 0 the same ratio is *unclipped*
  (the `min` picks the branch that hurts you). Getting this asymmetry wrong
  silently removes the trust region on one side — there's a test for both.
- **`logp_old` vs `logp_ref` are different models.** After the first
  gradient epoch, the sampling-time policy is stale but the reference model
  is *frozen forever*. Using one tensor for both is the classic bug.
- **Group standardization needs the biased std** (`unbiased=False`) and an ε:
  a group where all rewards are equal (all completions correct!) must yield
  advantage 0, not NaN.
- **k3, not `logp − logp_ref`.** The naive estimator is unbiased but can be
  negative per-sample and has huge variance; k3 is nonnegative and its
  expectation still converges to the true KL (verified empirically in the
  tests against a known categorical pair).
- **Mask everything.** Rewards are per-sequence but the loss is per-token;
  padding tokens must not dilute the mean (`masked_mean`, denominator
  clamped).

---

## Running the Smoke Test

```bash
pip install torch pytest
python -m pytest test_grpo.py -v
```

| Test | Validates |
|------|-----------|
| `test_advantages_standardized_per_group` | Mean 0 / std 1 within group, order preserved |
| `test_group_isolation` | Rewards in one group can't leak into another |
| `test_gradient_pushes_toward_good_completions` | Sign of the policy gradient |
| `test_clipping_zeroes_gradient_outside_trust_region` | Zero grad past clip; one-sidedness |
| `test_k3_kl_nonnegative_and_zero_at_equality` | k3 sample-level properties |
| `test_k3_estimates_true_kl` | k3 expectation → true KL (200k samples) |
| `test_masked_tokens_do_not_contribute` | Padding hygiene |
| `test_full_grpo_loss_runs_and_kl_anchors` | End-to-end assembly |

---

## Discussion Questions (interview follow-ups)

- **Why does GRPO drop the value network?** What does a learned critic buy
  PPO, when is it worth the extra model, and why does group sampling
  substitute well for verifiable-reward tasks (math/code)?
- **Token-level vs sequence-level ratios** — GRPO broadcasts a sequence
  advantage over tokens. What changes in DAPO / token-level variants and why
  does length bias creep in?
- **KL in the reward vs KL in the loss** — early RLHF subtracted KL from the
  reward; GRPO adds it to the loss. How do the gradients differ?
- **Reward hacking** — with a learned reward model instead of a verifier,
  which component here is the main defense, and what failure does
  over-optimizing look like on the KL/reward curves?
- **Dr. GRPO / length normalization** — recent work argues the `1/std`
  scaling and per-length averaging bias optimization. What are the fixes?
