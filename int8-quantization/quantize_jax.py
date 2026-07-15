"""JAX port of INT8 quantization (see quantize.py for the NumPy original
and full commentary)."""

import jax.numpy as jnp

QMIN, QMAX = -128, 127


def quantize_symmetric(x, axis=None):
    if axis is None:
        absmax = jnp.abs(x).max()
    else:
        absmax = jnp.abs(x).max(axis=1 - axis, keepdims=True)   # 2D only
    scale = jnp.maximum(absmax, 1e-12) / QMAX
    q = jnp.clip(jnp.round(x / scale), QMIN, QMAX).astype(jnp.int8)
    return q, scale


def dequantize_symmetric(q, scale):
    return q.astype(jnp.float32) * scale


def quantize_asymmetric(x):
    lo = min(float(x.min()), 0.0)
    hi = max(float(x.max()), 0.0)
    scale = max(hi - lo, 1e-12) / (QMAX - QMIN)
    zp = int(round(-lo / scale)) + QMIN
    q = jnp.clip(jnp.round(x / scale) + zp, QMIN, QMAX).astype(jnp.int8)
    return q, scale, zp


def dequantize_asymmetric(q, scale, zp):
    return (q.astype(jnp.float32) - zp) * scale


class QuantLinear:
    def __init__(self, w_OD, b_O=None):
        self.q_w_OD, self.w_scale_O1 = quantize_symmetric(w_OD, axis=0)
        self.b_O = b_O

    def __call__(self, x_ND):
        q_x_ND, x_scale = quantize_symmetric(x_ND, axis=None)
        acc_NO = q_x_ND.astype(jnp.int32) @ self.q_w_OD.astype(jnp.int32).T
        y_NO = acc_NO.astype(jnp.float32) * (x_scale * self.w_scale_O1.T)
        if self.b_O is not None:
            y_NO = y_NO + self.b_O
        return y_NO


def quantization_error(x, x_hat):
    return float(jnp.linalg.norm(x - x_hat) / (jnp.linalg.norm(x) + 1e-12))
