"""JAX port of the tiled/online-softmax attention (see online_attention.py
for the NumPy original and the full commentary)."""

import jax
import jax.numpy as jnp


def attention_naive(q_LK, k_TK, v_TK, causal=False):
    scale = 1.0 / jnp.sqrt(q_LK.shape[-1]).astype(q_LK.dtype)
    s_LT = q_LK @ k_TK.T * scale
    if causal:
        L, T = s_LT.shape
        mask = jnp.arange(T)[None, :] > jnp.arange(L)[:, None]
        s_LT = jnp.where(mask, -jnp.inf, s_LT)
    return jax.nn.softmax(s_LT, axis=-1) @ v_TK


def attention_tiled(q_LK, k_TK, v_TK, block=16, causal=False):
    L, K = q_LK.shape
    T = k_TK.shape[0]
    scale = 1.0 / jnp.sqrt(K).astype(q_LK.dtype)

    m_L = jnp.full((L,), -jnp.inf, dtype=q_LK.dtype)
    l_L = jnp.zeros(L, dtype=q_LK.dtype)
    o_LK = jnp.zeros((L, K), dtype=q_LK.dtype)

    for start in range(0, T, block):
        kb_CK = k_TK[start:start + block]
        vb_CK = v_TK[start:start + block]
        s_LC = q_LK @ kb_CK.T * scale
        if causal:
            qpos_L1 = jnp.arange(L)[:, None]
            kpos_1C = start + jnp.arange(kb_CK.shape[0])[None, :]
            s_LC = jnp.where(kpos_1C > qpos_L1, -jnp.inf, s_LC)

        m_new_L = jnp.maximum(m_L, s_LC.max(axis=-1))
        neg_inf_L = jnp.isneginf(m_new_L)
        alpha_L = jnp.where(neg_inf_L, 1.0, jnp.exp(m_L - m_new_L))
        shift_L = jnp.where(neg_inf_L, 0.0, m_new_L)
        p_LC = jnp.exp(s_LC - shift_L[:, None])

        l_L = alpha_L * l_L + p_LC.sum(axis=-1)
        o_LK = alpha_L[:, None] * o_LK + p_LC @ vb_CK
        m_L = m_new_L

    return o_LK / l_L[:, None]
