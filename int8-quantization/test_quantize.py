import numpy as np

from quantize import (QMAX, QMIN, QuantLinear, dequantize_asymmetric,
                      dequantize_symmetric, quantization_error,
                      quantize_asymmetric, quantize_symmetric)

rng = np.random.default_rng(0)


def test_symmetric_roundtrip_error_bound():
    x = rng.normal(size=(64, 64)).astype(np.float32)
    q, scale = quantize_symmetric(x)
    x_hat = dequantize_symmetric(q, scale)
    # Rounding error is at most scale/2 per element.
    assert np.abs(x - x_hat).max() <= scale / 2 + 1e-7


def test_symmetric_zero_is_exact():
    x = np.array([[-3.0, 0.0, 5.0]], dtype=np.float32)
    q, scale = quantize_symmetric(x)
    x_hat = dequantize_symmetric(q, scale)
    assert x_hat[0, 1] == 0.0


def test_int8_range_respected():
    x = rng.normal(size=(100,)).astype(np.float32) * 1000
    q, _ = quantize_symmetric(x)
    assert q.dtype == np.int8
    assert q.min() >= QMIN and q.max() <= QMAX


def test_asymmetric_uses_full_range_on_positive_data():
    x = rng.uniform(1.0, 9.0, size=(1000,)).astype(np.float32)
    q_sym, _ = quantize_symmetric(x)
    q_asym, scale, zp = quantize_asymmetric(x)
    # Symmetric wastes the negative half on all-positive data...
    assert q_sym.min() >= 0
    # ...asymmetric spreads over (nearly) the whole int8 range.
    assert q_asym.min() <= QMIN + 40
    assert q_asym.max() >= QMAX - 5
    # And reconstructs strictly better.
    err_sym = quantization_error(x, dequantize_symmetric(q_sym, _))
    err_asym = quantization_error(x, dequantize_asymmetric(q_asym, scale, zp))
    assert err_asym < err_sym


def test_asymmetric_real_zero_maps_exactly():
    x = np.array([0.0, 0.5, 4.0], dtype=np.float32)
    q, scale, zp = quantize_asymmetric(x)
    assert dequantize_asymmetric(q, scale, zp)[0] == 0.0


def test_per_channel_beats_per_tensor_with_outlier_channel():
    # One channel 100x larger: a per-tensor scale crushes the small
    # channels to a few int levels; per-channel keeps them all precise.
    w = rng.normal(size=(8, 32)).astype(np.float32)
    w[3] *= 100.0
    q_t, s_t = quantize_symmetric(w, axis=None)
    q_c, s_c = quantize_symmetric(w, axis=0)

    # Global L2 is dominated by the outlier channel either way; the damage
    # shows up in the per-channel relative error of the SMALL channels.
    def per_row_err(w_hat):
        return np.mean([quantization_error(w[i], w_hat[i]) for i in range(8)])

    err_t = per_row_err(dequantize_symmetric(q_t, s_t))
    err_c = per_row_err(dequantize_symmetric(q_c, s_c))
    assert err_c < err_t / 5


def test_quant_linear_close_to_float():
    w = rng.normal(size=(16, 32)).astype(np.float32)
    b = rng.normal(size=16).astype(np.float32)
    x = rng.normal(size=(4, 32)).astype(np.float32)
    got = QuantLinear(w, b)(x)
    want = x @ w.T + b
    assert quantization_error(want, got) < 0.02


def test_quant_linear_accumulates_in_int32():
    # Large D with adversarial signs: int16 accumulation would overflow
    # (127*127*512 >> 32767); int32 must survive and stay accurate.
    D = 512
    w = np.ones((2, D), dtype=np.float32)
    x = np.ones((1, D), dtype=np.float32)
    got = QuantLinear(w)(x)
    assert quantization_error(np.array([[D, D]], dtype=np.float32), got) < 1e-3


def test_error_metric_sanity():
    x = rng.normal(size=100)
    assert quantization_error(x, x) == 0.0
    assert abs(quantization_error(x, np.zeros_like(x)) - 1.0) < 1e-9
