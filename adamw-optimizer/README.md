# AdamW From Scratch (+ Grad Clipping + Warmup-Cosine Schedule)

**Problem:** Implement **AdamW** with bias correction and *decoupled* weight
decay, **global-norm gradient clipping**, and the standard LLM
**warmup-cosine** learning-rate schedule.

The correctness bar is unforgiving: your optimizer must match
`torch.optim.AdamW` **step-for-step to float precision** over 25 steps on
random gradients. "Roughly converges" doesn't cut it — every constant and
every operation-order choice must be right.

---

## Core Requirements

1. **AdamW (`AdamW`)**
   ```
   p     *= 1 − lr·wd                      # decoupled decay, FIRST
   m      = β₁·m + (1−β₁)·g
   v      = β₂·v + (1−β₂)·g²
   m̂, v̂  = m/(1−β₁ᵗ), v/(1−β₂ᵗ)          # bias correction
   p     -= lr · m̂ / (√v̂ + ε)
   ```
   - Decay is applied to the **parameter**, never through `m`/`v` — that's
     the entire difference between AdamW and Adam+L2.
   - `t` starts at 1 on the first step (β⁰ = 1 would divide by zero).

2. **Global-norm clipping (`clip_grad_global_norm`)** — one scale factor for
   *all* gradients so the global L2 norm ≤ max_norm; preserves gradient
   direction (per-tensor clipping doesn't); returns the pre-clip norm — the
   number you actually log to detect spikes.

3. **Schedule (`warmup_cosine_lr`)** — linear warmup 0 → base_lr over
   `warmup_steps`, then cosine to `min_lr`; clamps after `total_steps`.

---

## Behavior Notes / Gotchas

- **Order matters vs torch.** `torch.optim.AdamW` decays the parameter
  *before* the Adam step (`p.mul_(1 − lr·wd)`), not after. Decay-after passes
  every "does it converge" test and still fails the exact-match test — this
  exact off-by-one-operation bug was found while writing this problem.
- **Why bias correction exists.** `m` and `v` start at zero, so early EMAs
  are shrunk toward zero by a factor `(1−βᵗ)`. Without correction the first
  step is ~30× smaller than `lr` for default betas (there's a test measuring
  it). With correction, step 1 ≈ `lr·sign(g)`.
- **Adam is scale-invariant in the gradient.** Multiplying `g` by 100
  changes the step by ~nothing (`m̂/√v̂` carries the scale in both numerator
  and denominator). This is why lr transfers across losses — and why
  weight decay must be decoupled: an L2 term folded into `g` gets divided
  by `√v̂` and stops behaving like decay for parameters with large gradients.
- **Clip globally, not per-tensor.** Per-tensor clipping rotates the update
  direction toward the small-gradient tensors. One global factor preserves
  the direction exactly (tested with a 3-4-5 triangle).
- **Warmup is an Adam story.** Early `v̂` is estimated from a handful of
  samples; the effective preconditioner is noise. Low lr rides it out.

---

## Running the Smoke Test

```bash
pip install torch pytest
python -m pytest test_adamw.py -v
```

| Test | Validates |
|------|-----------|
| `test_matches_torch_adamw_exactly` | Step-for-step parity with torch.optim over 25 steps |
| `test_bias_correction_matters_at_step_one` | First step ≈ lr, not lr/30 |
| `test_decoupled_decay_bypasses_moments` | Zero grad still decays; moments untouched |
| `test_adam_step_size_is_scale_invariant` | 100× gradient ⇒ same step |
| `test_global_norm_clipping_preserves_direction` | Single scale factor, correct norm |
| `test_clipping_is_noop_under_threshold` | No spurious scaling |
| `test_schedule_shape` | Warmup peak, monotone decay, floor, cosine midpoint |
| `test_optimizes_a_quadratic` | End-to-end sanity |

---

## Discussion Questions (interview follow-ups)

- **Memory:** Adam keeps 2 extra floats per parameter. For a 70B model in
  mixed precision, walk through the full training memory budget (params,
  grads, moments, master weights) — where do 8-bit optimizers and ZeRO help?
- **β₂ and spikes:** why do LLM runs often lower β₂ (0.999 → 0.95), and how
  does that interact with loss spikes and grad clipping?
- **ε placement:** `√(v̂) + ε` vs `√(v̂ + ε)` — do they differ meaningfully?
  Which does torch use?
- **Weight decay exclusions:** why are LayerNorm gains and biases
  conventionally excluded from decay?
- **Schedules:** cosine vs linear vs WSD (warmup-stable-decay). Why did WSD
  become popular for continued pretraining?
