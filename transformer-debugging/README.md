# ML Debugging: The Broken Transformer

**Problem:** `buggy_transformer.py` trains a tiny decoder-only transformer on
a copy task (`[prefix, SEP, prefix]` — reproduce the prefix after the SEP).
The script **runs without errors and the loss goes down**, yet the model
can't copy anything. Find and fix every bug.

This is the signature OpenAI-style "ML debugging" round (reported repeatedly
on Blind and 1point3acres): no stack trace, no crash — just a model that
silently doesn't work. Grade yourself with `test_transformer.py`, whose tests
are named for the bug classes they catch.

**Rules of the game**

1. Read `buggy_transformer.py` and list every bug you can find *before*
   running anything.
2. Fix them in a copy of the file, then point the tests at your fix
   (or just compare with `fixed_transformer.py` afterwards).
3. For each bug, be able to say **what symptom it produces** — that mapping
   is what the interview is actually testing.

---

## What a healthy run looks like

- Final training loss ≈ **0.96, not 0** — the first four predicted positions
  are unpredictable random prefix tokens, an irreducible `4·ln(11)/10`.
  Knowing *what loss value to expect* is part of debugging: a broken model
  sits near `ln(11) ≈ 2.4`; a "too good" loss (≈ 0) means the objective is
  leaking the answer.
- Greedy generation reproduces the prefix after SEP essentially perfectly.

---

## The bugs (spoilers!)

<details>
<summary>Click to reveal the planted bugs and their symptoms</summary>

| # | Bug | Symptom |
|---|-----|---------|
| 1 | **No causal mask** in `attention` | Future tokens influence past positions; train loss looks great (the answer is visible!), generation is garbage. Caught by `test_causal_no_future_leak`. |
| 2 | **Softmax over the query axis** (`dim=-2` instead of `-1`) | Attention "weights" don't sum to 1 over keys; each key distributes over queries instead. Caught by `test_attention_rows_sum_to_one_over_keys`. |
| 3 | **Missing `1/√head_dim` scaling** | Score variance grows with head dim; softmax saturates, gradients through attention vanish, training is slow/unstable. Caught by `test_attention_scores_are_scaled`. |
| 4 | **Dropped residual around the MLP** (`x = mlp(ln(x))` instead of `x = x + mlp(ln(x))`) | The block overwrites its input; signal (and gradient highway) destroyed, deep stacks stop training. Caught by `test_blocks_have_residual_path`. |
| 5 | **Unshifted labels in `loss_fn`** (predicting the *current* token) | Loss collapses toward 0 — the model just echoes its input embedding — but generation repeats the last token forever. Caught by `test_loss_uses_shifted_labels` / `test_buggy_loss_rewards_copying_the_input`. |
| 6 | **Missing `opt.zero_grad()`** | Gradients accumulate across steps; effective step direction is a runaway momentum sum. Loss curve is noisy/plateaued. Caught (indirectly) by the end-to-end convergence bar in `test_training_overfits_and_generation_copies`. |

</details>

---

## Debugging heuristics this exercise drills

- **"Loss goes down" proves nothing.** Bugs 1 and 5 both *lower* the loss.
  Always pair the loss with a behavioral eval (here: actual generation).
- **Know your entropy floors.** Compute what loss a perfect model would get
  and what a trivial one would get; those two numbers bracket every diagnosis.
- **Test invariants, not implementations:** causality (perturb a future
  token), attention row-normalization (one-hot values trick), residual
  identity (zero the output projections and demand the identity map).
- **Shape-correct ≠ correct.** Every one of these bugs produces tensors of
  exactly the right shape. `dim=-2` softmax is the canonical example.

---

## Running

```bash
pip install torch pytest
python -m pytest test_transformer.py -v   # 8 tests; all pass against fixed_transformer.py
```

---

## Discussion Questions (interview follow-ups)

- What *training-curve* signatures distinguish bug 1 (causal leak) from
  bug 5 (label leak)? Both reach suspiciously low loss.
- Why does missing attention scaling get *worse* as head_dim grows? Relate
  to the variance of a dot product of independent unit-variance vectors.
- Pre-norm vs post-norm: this model is pre-norm. What breaks differently if
  you drop a residual in a post-norm block?
- The buggy script calls `loss.backward()` without `zero_grad()`. Why does
  Adam partially mask this bug compared to SGD?
- What would you add to a real training run to catch each of these six bugs
  within the first 100 steps? (grad-norm logging, attention-entropy stats,
  a held-out generation probe, activation norms per block...)
