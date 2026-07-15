# INT8 Post-Training Quantization

**Problem:** Implement INT8 quantization from scratch in NumPy: **symmetric
(absmax)** and **asymmetric (zero-point)** schemes, **per-tensor vs
per-channel** granularity, and a **W8A8 quantized linear layer** with int32
accumulation and a single float rescale.

The inference-efficiency question in MLE loops: everyone says "we quantized
the model"; this problem checks you know what that actually means at the
integer level.

---

## Core Requirements

1. **Symmetric / absmax** — `scale = absmax/127`, `q = round(x/scale)`;
   real 0.0 maps to integer 0 exactly. `axis` selects per-tensor vs
   per-channel scales.

2. **Asymmetric / zero-point** — `scale = (max−min)/255`,
   `zp = round(−min/scale) + qmin`; the range must be widened to include 0,
   and 0.0 must map to `zp` **exactly** (padding and post-ReLU zeros must
   introduce no error). Uses the full int8 range even on all-positive data.

3. **`QuantLinear` (W8A8)**
   - Weights: per-**output-channel** symmetric, quantized once at load.
   - Activations: per-tensor symmetric, quantized dynamically per call.
   - Matmul in **int32** (`int8 × int8` products summed over D — int16
     overflows at `127·127·D` almost immediately), then one float rescale:
     `y = acc · (scale_x · scale_w[channel])`. Bias stays float.

4. **Error metric** — relative L2 per layer; the number that decides which
   layers stay in high precision.

---

## Behavior Notes / Gotchas

- **Why zero must be exact.** Zero is the most common value flowing through
  a network (ReLU, padding, masks). If 0.0 dequantizes to ±ε, every padded
  position injects noise. Symmetric gets this for free; asymmetric gets it
  only because `zp` is *rounded to an integer* — that rounding is load-
  bearing, not cosmetic.
- **Per-channel exists because of outlier channels.** One weight row 100×
  larger forces a per-tensor scale that crushes every other row to a couple
  of integer levels. The test measures *per-row* relative error — global L2
  hides the damage because the outlier dominates the norm (a metric-choice
  lesson in itself).
- **Symmetric for weights, asymmetric for activations** is the standard
  pairing: weights are roughly zero-centered (symmetric wastes nothing);
  post-ReLU activations are one-sided (symmetric wastes half the range —
  tested: asymmetric reconstructs strictly better on positive data).
- **The rescale factorizes only because scales are per-tensor × per-channel.**
  Per-channel *activation* scales would put a different factor on every
  element of the accumulator — that's why activation quantization is
  per-tensor (or per-token) in practice.
- **Clamp before cast.** `np.round(x/scale)` can land on 128; casting
  silently wraps to −128 without an explicit clip.

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_quantize.py -v
```

| Test | Validates |
|------|-----------|
| `test_symmetric_roundtrip_error_bound` | Max error ≤ scale/2 |
| `test_symmetric_zero_is_exact` | 0.0 → integer 0 → 0.0 |
| `test_int8_range_respected` | Clip-then-cast |
| `test_asymmetric_uses_full_range_on_positive_data` | Zero-point range utilization, beats symmetric |
| `test_asymmetric_real_zero_maps_exactly` | Integer zero-point exactness |
| `test_per_channel_beats_per_tensor_with_outlier_channel` | ≥5× per-row error reduction |
| `test_quant_linear_close_to_float` | W8A8 end-to-end < 2% error |
| `test_quant_linear_accumulates_in_int32` | No overflow at D=512 |

---

## Discussion Questions (interview follow-ups)

- **LLM.int8() outliers** — transformer activations grow rare huge-magnitude
  feature dimensions. Why does naive W8A8 fall apart on >6B models, and how
  does mixed-precision decomposition fix it?
- **SmoothQuant** — migrate activation outliers into the weights via a
  per-channel rescale. Derive why `(x/s)(s·W) = xW` makes both sides easier
  to quantize.
- **Static vs dynamic activation quantization** — calibration sets,
  percentile clipping vs absmax, and what breaks with distribution shift.
- **GPTQ / AWQ (weight-only 4-bit)** — why is weight-only the right trade
  for memory-bound LLM decoding, and where does W8A8 still win (prefill,
  batch serving)?
- **QAT vs PTQ** — when is post-training quantization not enough, and what
  does the straight-through estimator do in QAT?
