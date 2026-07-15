"""JAX port of the bi-encoder InfoNCE components (see bi_encoder.py for the
PyTorch original). The backbone is abstracted as hidden states + mask, so
the port covers the interview-relevant math: masked mean pooling, L2
normalization, temperature in log-space, and false-negative masking.

Shape suffixes: B batch, L seq len, D model dim.
"""

import jax
import jax.numpy as jnp


def masked_mean_pool(h_BLD, mask_BL):
    """sum(h * mask) / sum(mask), denominator clamped >= 1."""
    mask_BL1 = mask_BL[..., None].astype(h_BLD.dtype)
    return (h_BLD * mask_BL1).sum(1) / jnp.maximum(mask_BL1.sum(1), 1.0)


def encode(h_BLD, mask_BL):
    """Pool + L2-normalize: dot product == cosine similarity afterwards."""
    pooled_BD = masked_mean_pool(h_BLD, mask_BL)
    return pooled_BD / jnp.maximum(
        jnp.linalg.norm(pooled_BD, axis=-1, keepdims=True), 1e-12)


def info_nce(q_BD, d_BD, doc_id_B, log_temp):
    B = q_BD.shape[0]
    sim_BB = q_BD @ d_BD.T / jnp.exp(log_temp)
    dup_BB = (doc_id_B[:, None] == doc_id_B[None, :]) & ~jnp.eye(B, dtype=bool)
    sim_BB = jnp.where(dup_BB, -jnp.inf, sim_BB)
    logp_BB = jax.nn.log_softmax(sim_BB, axis=-1)
    return -jnp.diagonal(logp_BB).mean()
