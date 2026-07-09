# Direct Preference Optimization (DPO) Loss

**Problem:** Implement the **DPO** objective (Rafailov et al., 2023) — the
RLHF-without-RL alignment method. Given preference pairs (a *chosen* and a
*rejected* completion for the same prompt), train the policy to prefer the
chosen response while staying close to a frozen reference model, using a
single classification-style loss instead of PPO.

```
L = -E[ log σ( β·(log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)) ) ]
```

---

## Core Requirements

1. **Per-sequence log-probs (`sequence_logprob`)**
   - Gather the log-prob of each target token from the logits
     (`log_softmax` + `gather`).
   - **Completion masking:** only completion tokens contribute. Prompt tokens
     and padding must be zeroed out — DPO compares *responses*, not prompts.
   - Caller passes logits/labels already shifted by one
     (`logits[:, :-1]` vs `ids[:, 1:]`).

2. **The loss (`dpo_loss`)**
   - Compute the two log-ratios `π/π_ref` (chosen and rejected), take the
     β-scaled margin, and apply `-logsigmoid`.
   - Return the **implicit rewards** `β·log(π/π_ref)` (detached) for logging —
     the reward-margin curve is the main training health signal.
   - Optional **label smoothing** (cDPO): treat preference labels as flipped
     with probability ε, which keeps gradient pressure on saturated pairs.

3. **The step (`dpo_step`)**
   - Score chosen and rejected under the policy *and* the reference model.
   - The reference forward runs under `torch.no_grad()` — it is frozen and
     only anchors the KL constraint.

---

## Interface

```python
def sequence_logprob(logits_BLV, labels_BL, mask_BL) -> Tensor      # (B,)
def dpo_loss(policy_chosen_B, policy_rejected_B,
             ref_chosen_B, ref_rejected_B,
             beta=0.1, label_smoothing=0.0) -> (loss, chosen_rewards_B, rejected_rewards_B)
def dpo_step(policy, ref, batch, beta=0.1) -> (loss, chosen_rewards_B, rejected_rewards_B)
```

Shape-suffix naming convention:

```
B = batch      L = sequence length      V = vocab size
```

---

## Behavior Notes / Gotchas

- **Loss at init is exactly `log 2`.** When policy == reference every
  log-ratio cancels, the margin is 0, and `-logsigmoid(0) = log 2 ≈ 0.693`.
  If your first logged loss isn't ~0.693, the implementation is wrong —
  the single best smoke test in this problem.
- **Sum, don't average, token log-probs.** DPO's derivation uses sequence
  log-probs; per-token averaging changes the objective (that variant exists —
  it's essentially what SimPO builds on — but it's a different method).
- **Masking bugs are silent.** Include prompt tokens and the model gets
  rewarded for re-ranking the prompt; include padding and batch composition
  leaks into the loss. Both train "fine" and evaluate badly.
- **β is the inverse KL budget.** Small β lets the policy drift far from the
  reference (reward hacking risk); large β pins it. Typical range 0.05–0.5.
- **Both implicit rewards often go *down* during training.** Only the margin
  is optimized. Watch chosen-vs-rejected margin, not absolute rewards.

---

## Running the Smoke Test

```bash
pip install torch pytest
python -m pytest test_dpo.py -v
```

| Test | Validates |
|------|-----------|
| `test_sequence_logprob_masks_prompt_and_padding` | Only completion tokens are scored |
| `test_loss_is_log2_when_policy_equals_ref` | The `log 2` init invariant |
| `test_gradient_widens_the_margin` | Chosen pushed up, rejected pushed down |
| `test_loss_decreases_as_margin_grows` | Loss is monotone in the margin |
| `test_beta_scales_the_implicit_reward` | Reward = β·log-ratio |
| `test_label_smoothing_softens_confident_margins` | cDPO keeps pressure on saturated pairs |
| `test_dpo_training_learns_the_preference` | End-to-end: tiny LM learns the pair |

---

## Discussion Questions (interview follow-ups)

- **Derivation** — walk from the KL-constrained RLHF objective to the DPO
  loss. Where does the partition function go, and why does it cancel?
- **DPO vs PPO** — what does DPO give up by being offline? When do frontier
  labs still reach for on-policy RL (or online DPO)?
- **Reference-free variants** — SimPO drops π_ref and length-normalizes.
  What breaks, what improves?
- **Length bias** — chosen responses tend to be longer. How does that leak
  into the sequence-sum objective, and how would you mitigate it?
- **IPO / KTO** — how do these alternatives change the link function, and
  what failure mode of DPO (overfitting confident pairs) does IPO target?
