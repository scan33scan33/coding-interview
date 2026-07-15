# Speculative Decoding

**Problem:** Implement **speculative decoding** (Leviathan et al. 2023;
Chen et al. 2023): a small **draft** model proposes γ tokens, the large
**target** model verifies them in one batched pass, and a rejection-sampling
rule decides what to keep.

The remarkable guarantee you must implement correctly: **the output
distribution is exactly the target model's**, regardless of how bad the draft
is. Draft quality affects *speed only*, never correctness — and the tests
verify this empirically against an adversarial draft.

---

## Core Requirements

1. **One round (`speculative_step`)**
   - Draft proposes γ tokens autoregressively.
   - For each draft token `x` with target prob `p(x)` and draft prob `q(x)`:
     **accept with probability `min(1, p(x)/q(x))`**.
   - On the first rejection at position `i`, sample the replacement from the
     **residual distribution `r(x) ∝ max(0, pᵢ(x) − qᵢ(x))`** and stop.
   - If all γ are accepted, emit a **bonus token** from the target's
     distribution at the position after the last draft token (its logits came
     free from the same verification pass).
   - Every round therefore emits ≥ 1 token — no livelock.

2. **The loop (`generate`)** — repeat rounds until `n` tokens; report the
   empirical acceptance rate.

3. **Speedup model (`expected_speedup`)** — with per-token acceptance `a`,
   one round emits `E = (1−a^{γ+1})/(1−a)` tokens for 1 target call +
   γ draft calls; return `E / (1 + γ·cost_ratio)`.

---

## Why the math works (the part to be able to derive on a whiteboard)

For each position, the emitted token's law is:

```
P(emit x) = q(x)·min(1, p(x)/q(x))  +  P(reject)·r(x)
          = min(p(x), q(x))         +  (Σ_y max(0, p(y)−q(y))) · max(0, p(x)−q(x)) / Σ_y max(0, p(y)−q(y))
          = min(p(x), q(x)) + max(0, p(x) − q(x))  =  p(x)   ∎
```

The accepted mass `min(p, q)` plus the residual mass reconstructs `p`
exactly — the same argument as rejection sampling with proposal `q`.

---

## Behavior Notes / Gotchas

- **The residual must be `max(0, p−q)` renormalized** — not `p`, not `p−q`
  with negatives. Resampling from plain `p` after a rejection is the classic
  wrong implementation: it double-counts mass where `q > p` and the output
  is biased (subtly — you need a distribution test to catch it, so the tests
  here run 40k trials).
- **Stop at the first rejection.** Positions after a rejected token were
  drafted under a prefix that no longer exists.
- **The bonus token is not optional garnish** — without it, a perfect draft
  (p == q, everything accepted) would emit γ tokens for one target pass but
  never use the target's final-position logits, wasting guaranteed progress.
- **Acceptance rate is the observability metric.** `E[accept] = Σ min(p, q)`
  = 1 − TV-distance between draft and target. Identical models accept 100%;
  the adversarial draft test shows correctness is preserved even near 0%.
- **Why this speeds anything up at all:** decode is memory-bandwidth-bound —
  scoring γ+1 positions in one target pass costs nearly the same as scoring
  1. Speculation converts sequential target calls into parallel verification.

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_speculative.py -v
```

| Test | Validates |
|------|-----------|
| `test_exactness_with_bad_draft` | Output dist == target dist, unrelated draft |
| `test_exactness_with_adversarial_draft` | ...even with a near-deterministic wrong draft |
| `test_perfect_draft_accepts_everything` | p/q = 1 ⇒ acceptance rate 1 |
| `test_acceptance_tracks_draft_quality` | Rate degrades with draft-target distance |
| `test_every_round_emits_at_least_one_token` | Rejection resample / bonus token bookkeeping |
| `test_generated_sequence_matches_target_markov_stats` | Long-run bigram stats match ancestral sampling |
| `test_speedup_formula_sanity` | The (1−a^{γ+1})/(1−a) accounting |

---

## Discussion Questions (interview follow-ups)

- **Prove exactness** — walk through the `min(p,q) + residual = p` argument
  above without notes.
- **Choosing γ** — what happens to speedup as γ grows with acceptance 0.8?
  Where's the optimum, and how does the draft/target cost ratio move it?
- **Temperature and top-p** — the theorem holds per *modified* distribution.
  What must you be careful about when target sampling uses top-p?
- **Self-speculation** — Medusa heads, EAGLE, and draft-free lookahead:
  what replaces the draft model, and what breaks in the acceptance math when
  proposals aren't sampled from a proper q?
- **Tree speculation** — verifying a *tree* of drafts in one pass
  (SpecInfer): how does the acceptance rule generalize?
- **Batch serving** — why does speculative decoding help less at high batch
  sizes on a saturated server?
