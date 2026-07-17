# RL Post-Training v2: Actually Train a Model, Watch the Eval Move

**Problem:** The v1 problem ([`grpo-ppo-loss`](../grpo-ppo-loss/)) implements
the GRPO/PPO *losses* against synthetic tensors. This v2 closes the loop:
**post-train a real (tiny) language model with RL against a verifiable
reward and demonstrate a held-out eval improvement** — the full
SFT → rollout → reward → advantage → update → eval pipeline that RLVR
(RL with verifiable rewards, à la DeepSeek-R1) runs at frontier scale,
shrunk to run on a CPU in about a minute.

```
Stage 1  SFT (undertrained on purpose)     held-out accuracy ~0.30
Stage 2  RL with GRPO or PPO               held-out accuracy ~0.60-0.65
```

## The task

Character-level 2-digit addition with fixed-width encoding:

```
prompt  "07+25="    (6 tokens, numbers zero-padded)
answer  "032" EOS   (4 tokens, sum zero-padded)
```

Fixed widths mean every sequence is exactly 10 tokens and the answer always
starts at position 6 — no padding masks, so the RL plumbing stays readable.
The reward is **exact match on all 4 answer tokens**: programmatically
verifiable, no reward model. Train and eval prompt sets are disjoint, so the
eval measures *generalization*, not memorization of rolled-out prompts.

## The pipeline (what you implement)

1. **`sft.py`** — supervised stage, cross-entropy on answer tokens only,
   stopped early (~30% accuracy) so RL has headroom to demonstrate a gain.
2. **`rl.py`** — the core:
   - `sample_rollouts` — batched temperature sampling, recording per-token
     `logp_old` at sampling time.
   - `grpo_advantages` — G samples per prompt, rewards standardized within
     the group (no critic).
   - PPO mode — a **value head** on the `'='` position predicts expected
     reward; `advantage = reward − V(prompt)`, critic trained with MSE.
   - Shared: PPO clip loss, k3 KL against the frozen SFT reference.
   - `train_rl` returns a **history of held-out eval accuracy** — the curve.
3. **`run_experiment.py`** — SFT → GRPO and SFT → PPO, printing the curve.

## Run it

```bash
pip install torch numpy pytest
python run_experiment.py     # ~1 minute on CPU
python -m pytest -q          # ~25 s
```

Actual output (seeds pinned):

```
SFT held-out accuracy: 0.303

[GRPO]  step 0    eval 0.303
        step 100  eval 0.490
        step 200  eval 0.580
        step 400  eval 0.650   reward 0.44  kl 0.153

[PPO]   step 400  eval 0.617   reward 0.40  kl 0.123

Summary:  SFT 0.303 -> GRPO 0.650 (+0.347) | PPO 0.617 (+0.313)
```

## Behavior Notes / Gotchas

- **The rollout/score position off-by-one.** Tokens generated at positions
  6..9 are scored by logits at positions 5..8. Get this wrong and `logp_old`
  disagrees with recomputed log-probs and the ratio is garbage from step 1.
  `test_sampled_logprobs_match_recompute` pins it: log-probs recorded during
  sampling must exactly equal a from-scratch recompute on the same sequences.
- **Exploration must carry signal.** GRPO learns nothing from a group whose
  rewards are all equal (advantage ≡ 0 — and note the ε in the denominator
  keeps a solved prompt at 0 rather than NaN). The undertrained ~30% SFT
  model at temperature 1 yields mixed groups, which is exactly why the
  baseline is left mediocre. From a 0% or 100% model, GRPO is inert.
- **Eval ≠ reward.** Training reward is computed on *sampled* rollouts of
  *train* prompts; the eval is *greedy* decoding on *held-out* prompts. The
  gap between the two curves is your generalization/variance signal — watch
  both in the experiment output.
- **The KL anchor earns its keep.** With `kl_coef=0`, the policy can win
  reward while drifting far from the reference (and in bigger models,
  wrecking everything else it knows). Here KL stays ~0.1–0.25; the tests
  assert it stays bounded.
- **GRPO vs PPO on this task.** GRPO's group baseline is computed from 8
  fresh samples per prompt — cheap here, and slightly better than PPO's
  learned critic, which must itself be trained and starts out wrong. That's
  the actual argument for GRPO in verifiable-reward settings.

## Test map

| Test | Validates |
|------|-----------|
| `test_encoding_hand_case`, `test_reward_is_exact_match` | Task plumbing, EOS included in the match |
| `test_grpo_advantages_group_properties` | Mean-0 groups; all-equal group → 0, not NaN |
| `test_train_eval_split_is_disjoint` | Eval measures generalization |
| `test_sft_baseline_is_mediocre` | Headroom exists |
| `test_sampled_logprobs_match_recompute` | The off-by-one killer |
| `test_rollouts_have_reward_signal` | Mixed-reward groups at temp 1 |
| `test_grpo_improves_heldout_eval` | ≥ +12 points, KL bounded |
| `test_ppo_improves_heldout_eval` | Critic variant also works |
| `test_history_records_the_curve` | The eval curve is recorded correctly |

## Discussion Questions (interview follow-ups)

- Why is the *held-out greedy* accuracy the right eval rather than mean
  training reward? What failure would reward-only monitoring hide?
- The reward here is binary exact-match. What changes with partial credit
  (per-digit reward)? When does reward shaping help vs distort?
- Scale this to a 7B model: which parts of `rl.py` survive (the loss math)
  and which get replaced (batched inference engine for rollouts, distributed
  logp recompute, reference-model sharding)?
- The SFT stage was stopped early on purpose. In real RLVR, what plays the
  role of "headroom" — and why does RL on a task the SFT model *never*
  solves at temperature fail (zero gradient — the exploration problem)?
- Off-policy drift: we do one gradient step per rollout batch. What breaks
  when you take many inner epochs, and how do the clip ratio + KL interact
  to contain it?

*This problem is PyTorch-only by design — the training loop is the point,
and a JAX port (pure-functional rollout buffer, `lax.scan` decode loop) is
itself a good follow-up exercise.*
