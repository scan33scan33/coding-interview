"""Tiled attention with online softmax — the core FlashAttention recurrence.

Computes softmax(Q K^T / sqrt(d)) V without ever materializing the L x T
attention matrix: keys/values stream in blocks, and running statistics
(row max m, normalizer l, unnormalized accumulator o) are updated per block.

Shape suffix convention:
  L = query length, T = key length, K = head dim, C = key block (chunk) size
"""

import numpy as np


def attention_naive(q_LK, k_TK, v_TK, causal=False):
    """Reference: materializes the full L x T score matrix."""
    scale = 1.0 / np.sqrt(q_LK.shape[-1])
    s_LT = q_LK @ k_TK.T * scale
    if causal:
        L, T = s_LT.shape
        s_LT = np.where(np.arange(T)[None, :] > np.arange(L)[:, None],
                        -np.inf, s_LT)
    s_LT = s_LT - s_LT.max(axis=-1, keepdims=True)  # safe softmax
    p_LT = np.exp(s_LT)
    p_LT /= p_LT.sum(axis=-1, keepdims=True)
    return p_LT @ v_TK


def attention_tiled(q_LK, k_TK, v_TK, block=16, causal=False):
    """FlashAttention-style streaming attention. Peak extra memory is O(L*C),
    independent of T — never the full L x T matrix.

    Per-row state, updated as each key block arrives:
      m : running max of scores seen so far      (for numerical stability)
      l : running sum of exp(score - m)          (softmax denominator)
      o : running sum of exp(score - m) @ V      (UNnormalized output)

    When a new block raises the max from m to m', previously accumulated
    l and o were scaled by exp(-m) instead of exp(-m'), so both are
    rescaled by alpha = exp(m - m'). Division by l happens once, at the end.
    """
    L, K = q_LK.shape
    T = k_TK.shape[0]
    scale = 1.0 / np.sqrt(K)

    m_L = np.full(L, -np.inf)
    l_L = np.zeros(L)
    o_LK = np.zeros((L, K))

    for start in range(0, T, block):
        kb_CK = k_TK[start:start + block]
        vb_CK = v_TK[start:start + block]
        s_LC = q_LK @ kb_CK.T * scale
        if causal:
            qpos_L1 = np.arange(L)[:, None]
            kpos_1C = (start + np.arange(kb_CK.shape[0]))[None, :]
            s_LC = np.where(kpos_1C > qpos_L1, -np.inf, s_LC)

        m_new_L = np.maximum(m_L, s_LC.max(axis=-1))
        # exp(-inf - -inf) would be nan; a row that has seen nothing and
        # gets a fully masked block must keep alpha = 1 (nothing to rescale).
        alpha_L = np.where(np.isneginf(m_new_L), 1.0, np.exp(m_L - m_new_L))
        p_LC = np.exp(s_LC - np.where(np.isneginf(m_new_L), 0.0, m_new_L)[:, None])

        l_L = alpha_L * l_L + p_LC.sum(axis=-1)
        o_LK = alpha_L[:, None] * o_LK + p_LC @ vb_CK
        m_L = m_new_L

    return o_LK / l_L[:, None]
