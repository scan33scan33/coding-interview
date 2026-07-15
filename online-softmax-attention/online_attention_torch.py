"""PyTorch port of the tiled/online-softmax attention (see
online_attention.py for the NumPy original and the full commentary)."""

import math

import torch


def attention_naive(q_LK, k_TK, v_TK, causal=False):
    scale = 1.0 / math.sqrt(q_LK.shape[-1])
    s_LT = q_LK @ k_TK.T * scale
    if causal:
        L, T = s_LT.shape
        mask = torch.arange(T)[None, :] > torch.arange(L)[:, None]
        s_LT = s_LT.masked_fill(mask, float("-inf"))
    return torch.softmax(s_LT, dim=-1) @ v_TK


def attention_tiled(q_LK, k_TK, v_TK, block=16, causal=False):
    L, K = q_LK.shape
    T = k_TK.shape[0]
    scale = 1.0 / math.sqrt(K)

    m_L = torch.full((L,), float("-inf"), dtype=q_LK.dtype)
    l_L = torch.zeros(L, dtype=q_LK.dtype)
    o_LK = torch.zeros(L, K, dtype=q_LK.dtype)

    for start in range(0, T, block):
        kb_CK = k_TK[start:start + block]
        vb_CK = v_TK[start:start + block]
        s_LC = q_LK @ kb_CK.T * scale
        if causal:
            qpos_L1 = torch.arange(L)[:, None]
            kpos_1C = start + torch.arange(kb_CK.shape[0])[None, :]
            s_LC = s_LC.masked_fill(kpos_1C > qpos_L1, float("-inf"))

        m_new_L = torch.maximum(m_L, s_LC.max(dim=-1).values)
        neg_inf_L = torch.isinf(m_new_L) & (m_new_L < 0)
        alpha_L = torch.where(neg_inf_L, torch.ones_like(m_L),
                              torch.exp(m_L - m_new_L))
        shift_L = torch.where(neg_inf_L, torch.zeros_like(m_new_L), m_new_L)
        p_LC = torch.exp(s_LC - shift_L[:, None])

        l_L = alpha_L * l_L + p_LC.sum(dim=-1)
        o_LK = alpha_L[:, None] * o_LK + p_LC @ vb_CK
        m_L = m_new_L

    return o_LK / l_L[:, None]
