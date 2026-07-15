"""Post-training INT8 quantization: symmetric (absmax) and asymmetric
(zero-point) schemes, per-tensor and per-channel granularity, and a W8A8
quantized linear layer with int32 accumulation.

Shape suffix convention:
  N = rows / batch, D = in features, O = out features
"""

import numpy as np

QMIN, QMAX = -128, 127  # int8


def quantize_symmetric(x, axis=None):
    """absmax scheme: scale = absmax / 127, q = round(x / scale).

    axis=None -> per-tensor (one scale); axis=k -> per-channel (a scale per
    slice along axis k, e.g. per output row of a weight matrix).
    Returns (q_int8, scale). Zero always maps to integer 0 exactly.
    """
    if axis is None:
        absmax = np.abs(x).max()
    else:
        absmax = np.abs(x).max(axis=1 - axis, keepdims=True)  # 2D only
    scale = np.maximum(absmax, 1e-12) / QMAX
    q = np.clip(np.round(x / scale), QMIN, QMAX).astype(np.int8)
    return q, scale


def dequantize_symmetric(q, scale):
    return q.astype(np.float32) * scale


def quantize_asymmetric(x):
    """zero-point scheme for skewed ranges (e.g. post-ReLU activations):

        scale = (max - min) / 255
        zp    = round(-min / scale) + QMIN      (an int8 value)
        q     = round(x / scale) + zp

    The full int8 range is used even when the data is all-positive, and the
    real value 0.0 maps to the integer zp EXACTLY (required so that zero
    padding / ReLU zeros introduce no quantization error).
    """
    lo, hi = float(x.min()), float(x.max())
    lo, hi = min(lo, 0.0), max(hi, 0.0)          # range must include 0
    scale = max(hi - lo, 1e-12) / (QMAX - QMIN)
    zp = int(round(-lo / scale)) + QMIN
    q = np.clip(np.round(x / scale) + zp, QMIN, QMAX).astype(np.int8)
    return q, scale, zp


def dequantize_asymmetric(q, scale, zp):
    return (q.astype(np.float32) - zp) * scale


class QuantLinear:
    """W8A8 linear: weights quantized per-OUTPUT-channel symmetric at load
    time; activations quantized per-tensor symmetric dynamically per call.

    The matmul runs in int32 (int8 x int8 accumulates safely: 127*127*D
    stays in int32 for any sane D), then ONE float rescale at the end:

        y = (q_x @ q_w.T) * (scale_x * scale_w_per_channel)
    """

    def __init__(self, w_OD, b_O=None):
        self.q_w_OD, self.w_scale_O1 = quantize_symmetric(w_OD, axis=0)
        self.b_O = b_O

    def __call__(self, x_ND):
        q_x_ND, x_scale = quantize_symmetric(x_ND, axis=None)
        acc_NO = q_x_ND.astype(np.int32) @ self.q_w_OD.astype(np.int32).T
        y_NO = acc_NO.astype(np.float32) * (x_scale * self.w_scale_O1.T)
        if self.b_O is not None:
            y_NO = y_NO + self.b_O                # bias stays in float
        return y_NO


def quantization_error(x, x_hat):
    """Relative L2 error — the metric to report per layer when deciding
    which layers to keep in higher precision."""
    return float(np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-12))
