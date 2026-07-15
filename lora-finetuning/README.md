# LoRA: Low-Rank Adaptation From Scratch

**Problem:** Implement **LoRA** (Hu et al., 2021) — the standard
parameter-efficient fine-tuning method. Wrap a frozen `nn.Linear` with a
trainable low-rank update, support **merge/unmerge** for zero-overhead
inference and adapter swapping, and inject adapters into a model by module
name.

```
y = W₀x + (α/r) · B A x        A: r×I (random init), B: O×r (ZERO init)
```

Reported in ML-coding rounds at frontier labs ("implement LoRA, KV cache,
beam search" per recent interview writeups).

---

## Core Requirements

1. **`LoRALinear`**
   - Freeze the wrapped base layer (`requires_grad_(False)`).
   - `A` Kaiming-initialized, `B` **zeros** → at step 0 the wrapped layer
     computes exactly the base function. Fine-tuning starts from the
     pretrained model, not a random perturbation of it.
   - Scale the update by `α/r` so changing the rank doesn't require retuning
     the learning rate.
   - Optional dropout on the adapter input path only.

2. **`merge()` / `unmerge()`**
   - Merge folds `(α/r)·BA` into `W₀` in place → inference cost identical to
     the base model.
   - Both must be **idempotent**, and `unmerge` must be the exact inverse —
     that's what lets one base model serve many adapters.

3. **Injection & plumbing**
   - `add_lora(model, target_names)` — replace matching `nn.Linear`
     submodules in place (classic recipe: attention `q_proj`/`v_proj` only).
     Modules can't replace themselves; swap via the parent.
   - `lora_state_dict(model)` — adapter weights only (the artifact you ship:
     MBs, not GBs).
   - `mark_only_lora_trainable(model)`.

---

## Behavior Notes / Gotchas

- **Why is B zero and not A?** Either being zero makes the update zero at
  init, but if *both* were zero there would be no gradient signal at all
  (`∂L/∂A ∝ B = 0` and vice versa). One random + one zero gives identity
  init *and* nonzero gradients. Initializing both randomly perturbs the
  pretrained model before training starts.
- **Merge idempotence is a real bug class.** Calling `merge()` twice must not
  add the update twice — track state. Same for unmerge.
- **Gradient hygiene.** `test_finetuning_moves_adapter_not_base` checks the
  base weight is **bit-exact** after 100 optimizer steps. Passing only
  "close" weights means the base leaked into the optimizer.
- **Where the memory savings come from.** Not the forward pass — the frozen
  weights are still resident. The win is optimizer state: Adam keeps two
  moments per *trainable* parameter, so 0.1% trainable params cuts optimizer
  memory ~1000×, plus you skip gradient storage for frozen weights.
- **`(x @ Aᵀ) @ Bᵀ`, not `x @ (BA)ᵀ`.** Materializing `BA` is an O×I matmul —
  the whole point of low rank is doing two skinny matmuls instead.

---

## Running the Smoke Test

```bash
pip install torch pytest
python -m pytest test_lora.py -v
```

| Test | Validates |
|------|-----------|
| `test_init_is_identity_wrt_base` | Zero-init B → exact base function at step 0 |
| `test_only_lora_params_get_gradients` | Base frozen, adapter trainable |
| `test_merge_matches_unmerged_forward` | Merge exact + idempotent |
| `test_unmerge_restores_base_weights` | Unmerge is the exact inverse |
| `test_add_lora_targets_only_named_modules` | q/v wrapped, k/out untouched |
| `test_lora_state_dict_is_small` | Shipping artifact is adapters only |
| `test_finetuning_moves_adapter_not_base` | Learns; base bit-exact after training |
| `test_adapter_swap_via_unmerge` | One base, multiple adapters |

---

## Discussion Questions (interview follow-ups)

- **Rank choice** — what does the intrinsic-dimension argument say about why
  r=8 works on billion-parameter models? When does it fail (hard domain
  shift, new languages)?
- **QLoRA** — what changes when the base is NF4-quantized? Why do adapters
  stay in bf16, and what is double quantization?
- **Serving many adapters** — merged weights can serve only one adapter at a
  time. How do multi-tenant systems (e.g. S-LoRA, punica) batch requests for
  *different* adapters in one forward pass?
- **LoRA vs full fine-tuning** — where does LoRA measurably lag (long
  training, reasoning-heavy tasks), and why might that be a regularization
  story rather than a capacity story?
- **DoRA / rsLoRA** — what instability in vanilla LoRA do the
  magnitude-direction decomposition and the `α/√r` rescaling address?
