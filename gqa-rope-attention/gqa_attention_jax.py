"""JAX port of the GQA + RoPE + KV-cache attention block.

Functional style: parameters live in a dict (pytree), the forward is a pure
function `gqa_forward(params, x, cache)` — the JAX-native counterpart of the
PyTorch nn.Module in gqa_attention.py. Same shape-suffix convention:
  B batch, L new length, T total length, H query heads, G kv heads, K head dim
"""

import jax
import jax.numpy as jnp


def rope_tables(head_dim, max_len, base=10000.0):
    inv_freq_J = base ** (-jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    ang_TJ = jnp.outer(jnp.arange(max_len, dtype=jnp.float32), inv_freq_J)
    return jnp.cos(ang_TJ), jnp.sin(ang_TJ)


def apply_rope(x_BhLK, cos_TJ, sin_TJ, offset=0):
    L = x_BhLK.shape[2]
    cos_LJ = cos_TJ[offset:offset + L]
    sin_LJ = sin_TJ[offset:offset + L]
    x1_BhLJ, x2_BhLJ = x_BhLK[..., 0::2], x_BhLK[..., 1::2]
    even = x1_BhLJ * cos_LJ - x2_BhLJ * sin_LJ
    odd = x1_BhLJ * sin_LJ + x2_BhLJ * cos_LJ
    # Interleave back: stack pairs on a new trailing axis, then flatten.
    return jnp.stack([even, odd], axis=-1).reshape(x_BhLK.shape)


def init_gqa(key, dim, n_heads, n_kv_heads, max_len=2048):
    assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
    head_dim = dim // n_heads
    ks = jax.random.split(key, 4)
    scale = dim ** -0.5
    cos_TJ, sin_TJ = rope_tables(head_dim, max_len)
    return {
        "wq": jax.random.normal(ks[0], (dim, n_heads * head_dim)) * scale,
        "wk": jax.random.normal(ks[1], (dim, n_kv_heads * head_dim)) * scale,
        "wv": jax.random.normal(ks[2], (dim, n_kv_heads * head_dim)) * scale,
        "wo": jax.random.normal(ks[3], (n_heads * head_dim, dim)) * scale,
        "cos": cos_TJ, "sin": sin_TJ,
        "n_heads": n_heads, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
    }


def gqa_forward(params, x_BLD, cache=None):
    B, L, _ = x_BLD.shape
    H, G, K = params["n_heads"], params["n_kv_heads"], params["head_dim"]
    offset = 0 if cache is None else cache[0].shape[2]

    q_BHLK = (x_BLD @ params["wq"]).reshape(B, L, H, K).swapaxes(1, 2)
    k_BGLK = (x_BLD @ params["wk"]).reshape(B, L, G, K).swapaxes(1, 2)
    v_BGLK = (x_BLD @ params["wv"]).reshape(B, L, G, K).swapaxes(1, 2)

    q_BHLK = apply_rope(q_BHLK, params["cos"], params["sin"], offset)
    k_BGLK = apply_rope(k_BGLK, params["cos"], params["sin"], offset)

    if cache is not None:
        k_BGTK = jnp.concatenate([cache[0], k_BGLK], axis=2)
        v_BGTK = jnp.concatenate([cache[1], v_BGLK], axis=2)
    else:
        k_BGTK, v_BGTK = k_BGLK, v_BGLK
    T = k_BGTK.shape[2]

    rep = H // G
    k_BHTK = jnp.repeat(k_BGTK, rep, axis=1)
    v_BHTK = jnp.repeat(v_BGTK, rep, axis=1)

    att_BHLT = q_BHLK @ k_BHTK.swapaxes(-1, -2) / jnp.sqrt(K).astype(x_BLD.dtype)
    qpos_L1 = jnp.arange(offset, offset + L)[:, None]
    kpos_1T = jnp.arange(T)[None, :]
    att_BHLT = jnp.where(kpos_1T > qpos_L1, -jnp.inf, att_BHLT)

    out_BHLK = jax.nn.softmax(att_BHLT, axis=-1) @ v_BHTK
    out_BLD = out_BHLK.swapaxes(1, 2).reshape(B, L, H * K)
    return out_BLD @ params["wo"], (k_BGTK, v_BGTK)
