import math

import jax
import jax.numpy as jnp
import numpy as np

from bi_encoder_jax import encode, info_nce, masked_mean_pool


def test_pooling_ignores_padding():
    h = jax.random.normal(jax.random.key(0), (2, 6, 8))
    mask = jnp.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    got = masked_mean_pool(h, mask)
    np.testing.assert_allclose(got[0], h[0, :3].mean(0), atol=1e-6)
    # Garbage in the padded region must not move the pooled vector.
    h2 = h.at[0, 3:].add(100.0)
    np.testing.assert_allclose(masked_mean_pool(h2, mask)[0], got[0], atol=1e-5)


def test_output_is_normalized():
    h = jax.random.normal(jax.random.key(0), (4, 5, 16))
    mask = jnp.ones((4, 5))
    norms = jnp.linalg.norm(encode(h, mask), axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_perfect_match_low_loss():
    q = encode(jax.random.normal(jax.random.key(0), (8, 3, 16)), jnp.ones((8, 3)))
    doc_ids = jnp.arange(8)
    log_temp = jnp.log(jnp.array(0.05))
    loss_aligned = info_nce(q, q, doc_ids, log_temp)       # d == q: perfect
    assert float(loss_aligned) < 0.01


def test_misaligned_high_loss():
    q = encode(jax.random.normal(jax.random.key(0), (8, 3, 16)), jnp.ones((8, 3)))
    d = jnp.roll(q, 1, axis=0)                             # shuffled positives
    log_temp = jnp.log(jnp.array(0.05))
    assert float(info_nce(q, d, jnp.arange(8), log_temp)) > \
        float(info_nce(q, q, jnp.arange(8), log_temp))


def test_false_negative_masking():
    # Rows 0 and 1 share a document. Without masking, row 0's loss treats
    # column 1 (an identical doc) as a negative and can never go to zero.
    q = encode(jax.random.normal(jax.random.key(0), (4, 3, 16)), jnp.ones((4, 3)))
    d = q
    dup_ids = jnp.array([7, 7, 2, 3])
    log_temp = jnp.log(jnp.array(0.05))
    loss_masked = info_nce(q, d, dup_ids, log_temp)
    assert float(loss_masked) < 0.01                       # dup not penalized

    unique_ids = jnp.array([0, 1, 2, 3])
    q_dup = q.at[1].set(q[0])                              # duplicate WITHOUT masking
    loss_unmasked = info_nce(q_dup, q_dup, unique_ids, log_temp)
    assert float(loss_unmasked) > float(loss_masked)


def test_loss_gradient_flows_to_temperature():
    q = encode(jax.random.normal(jax.random.key(0), (8, 3, 16)), jnp.ones((8, 3)))
    d = jnp.roll(q, 1, axis=0)
    g = jax.grad(lambda lt: info_nce(q, d, jnp.arange(8), lt))(
        jnp.log(jnp.array(0.05)))
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_uniform_random_loss_is_log_B():
    # With random unrelated embeddings and high temperature, the softmax is
    # near-uniform: loss ~ log(B).
    B = 16
    q = encode(jax.random.normal(jax.random.key(1), (B, 3, 32)), jnp.ones((B, 3)))
    d = encode(jax.random.normal(jax.random.key(2), (B, 3, 32)), jnp.ones((B, 3)))
    loss = info_nce(q, d, jnp.arange(B), jnp.log(jnp.array(100.0)))
    assert abs(float(loss) - math.log(B)) < 0.05
