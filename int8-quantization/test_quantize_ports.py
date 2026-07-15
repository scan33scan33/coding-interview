"""Parametrized suite for the PyTorch and JAX INT8 quantization ports."""

import numpy as np
import pytest

rng = np.random.default_rng(0)


@pytest.fixture(params=["torch", "jax"])
def impl(request):
    if request.param == "torch":
        import torch
        import quantize_torch as m
        return m, lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    import quantize_jax as m
    import jax.numpy as jnp
    return m, lambda a: jnp.asarray(a, dtype=jnp.float32)


def test_symmetric_roundtrip_error_bound(impl):
    m, to = impl
    x = rng.normal(size=(64, 64)).astype(np.float32)
    q, scale = m.quantize_symmetric(to(x))
    x_hat = np.asarray(m.dequantize_symmetric(q, scale))
    assert np.abs(x - x_hat).max() <= float(np.asarray(scale)) / 2 + 1e-6


def test_zero_maps_exactly_both_schemes(impl):
    m, to = impl
    x = to([-3.0, 0.0, 5.0])
    q, s = m.quantize_symmetric(x)
    assert float(np.asarray(m.dequantize_symmetric(q, s))[1]) == 0.0
    q, s, zp = m.quantize_asymmetric(x)
    assert float(np.asarray(m.dequantize_asymmetric(q, s, zp))[1]) == 0.0


def test_asymmetric_beats_symmetric_on_positive_data(impl):
    m, to = impl
    x = rng.uniform(1.0, 9.0, size=1000).astype(np.float32)
    q_s, s_s = m.quantize_symmetric(to(x))
    q_a, s_a, zp = m.quantize_asymmetric(to(x))
    err_s = m.quantization_error(to(x), m.dequantize_symmetric(q_s, s_s))
    err_a = m.quantization_error(to(x), m.dequantize_asymmetric(q_a, s_a, zp))
    assert err_a < err_s


def test_per_channel_beats_per_tensor_with_outlier(impl):
    m, to = impl
    w = rng.normal(size=(8, 32)).astype(np.float32)
    w[3] *= 100.0
    q_t, s_t = m.quantize_symmetric(to(w), axis=None)
    q_c, s_c = m.quantize_symmetric(to(w), axis=0)
    wt = np.asarray(m.dequantize_symmetric(q_t, s_t))
    wc = np.asarray(m.dequantize_symmetric(q_c, s_c))

    def per_row(w_hat):
        return np.mean([np.linalg.norm(w[i] - w_hat[i]) / np.linalg.norm(w[i])
                        for i in range(8)])

    assert per_row(wc) < per_row(wt) / 5


def test_quant_linear_close_to_float(impl):
    m, to = impl
    w = rng.normal(size=(16, 32)).astype(np.float32)
    b = rng.normal(size=16).astype(np.float32)
    x = rng.normal(size=(4, 32)).astype(np.float32)
    got = np.asarray(m.QuantLinear(to(w), to(b))(to(x)))
    want = x @ w.T + b
    assert np.linalg.norm(want - got) / np.linalg.norm(want) < 0.02


def test_int32_accumulation_survives_large_d(impl):
    m, to = impl
    D = 512
    got = np.asarray(m.QuantLinear(to(np.ones((2, D))))(to(np.ones((1, D)))))
    np.testing.assert_allclose(got, [[D, D]], rtol=1e-3)
