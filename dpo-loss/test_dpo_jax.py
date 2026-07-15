import math

import jax
import jax.numpy as jnp
import numpy as np

from dpo_jax import dpo_loss, dpo_step, sequence_logprob


def test_sequence_logprob_masks_prompt_and_padding():
    key = jax.random.key(0)
    logits = jax.random.normal(key, (2, 5, 7))
    labels = jax.random.randint(jax.random.key(1), (2, 5), 0, 7)
    mask = jnp.array([[0., 0., 1., 1., 0.], [0., 1., 1., 1., 1.]])
    got = sequence_logprob(logits, labels, mask)

    logp = jax.nn.log_softmax(logits, axis=-1)
    manual0 = logp[0, 2, labels[0, 2]] + logp[0, 3, labels[0, 3]]
    assert abs(float(got[0] - manual0)) < 1e-5

    logits2 = logits.at[0, 0].add(100.0)      # masked position: no effect
    np.testing.assert_allclose(got, sequence_logprob(logits2, labels, mask),
                               atol=1e-5)


def test_loss_is_log2_when_policy_equals_ref():
    lp = jax.random.normal(jax.random.key(0), (4,))
    loss, cr, rr = dpo_loss(lp, lp - 1.0, lp, lp - 1.0)
    assert abs(float(loss) - math.log(2.0)) < 1e-6
    np.testing.assert_allclose(cr, jnp.zeros(4), atol=1e-7)
    np.testing.assert_allclose(rr, jnp.zeros(4), atol=1e-7)


def test_gradient_widens_the_margin():
    ref = jnp.zeros(1)

    def loss_of(pc, pr):
        return dpo_loss(pc, pr, ref, ref)[0]

    g_pc, g_pr = jax.grad(loss_of, argnums=(0, 1))(jnp.zeros(1), jnp.zeros(1))
    assert float(g_pc[0]) < 0        # descent raises chosen logprob
    assert float(g_pr[0]) > 0        # ...and lowers rejected


def test_rewards_carry_no_gradient():
    # stop_gradient: differentiating through the returned rewards gives 0.
    ref = jnp.zeros(1)

    def reward_sum(pc):
        _, cr, _ = dpo_loss(pc, jnp.zeros(1), ref, ref)
        return cr.sum()

    assert float(jax.grad(reward_sum)(jnp.array([2.0]))[0]) == 0.0


def test_dpo_training_learns_the_preference():
    V, D = 11, 16
    k1, k2 = jax.random.split(jax.random.key(42))
    params = {"emb": jax.random.normal(k1, (V, D)) * 0.1,
              "head": jax.random.normal(k2, (D, V)) * 0.1}
    ref_params = jax.tree_util.tree_map(jnp.copy, params)

    def apply_fn(p, ids):
        return p["emb"][ids] @ p["head"]

    batch = {
        "chosen_ids": jnp.array([[1, 2, 3, 4]]),
        "chosen_mask": jnp.array([[0., 0., 1., 1.]]),
        "rejected_ids": jnp.array([[1, 2, 5, 6]]),
        "rejected_mask": jnp.array([[0., 0., 1., 1.]]),
    }

    def loss_fn(p):
        return dpo_step(apply_fn, p, apply_fn, ref_params, batch, beta=0.1)[0]

    first = float(loss_fn(params))
    assert abs(first - math.log(2.0)) < 1e-5      # policy == ref at init

    for _ in range(100):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree_util.tree_map(lambda p, g: p - 0.5 * g, params, grads)
    final = float(loss_fn(params))
    assert final < first

    _, cr, rr = dpo_step(apply_fn, params, apply_fn, ref_params, batch, beta=0.1)
    assert (cr > rr).all()
