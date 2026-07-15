import jax
import jax.numpy as jnp
import numpy as np

from grpo_jax import grpo_advantages, grpo_loss, k3_kl, ppo_clip_loss


def test_advantages_standardized_per_group():
    rewards = jnp.array([1.0, 2.0, 3.0, 10.0, 20.0])
    groups = jnp.array([0, 0, 0, 1, 1])
    adv = grpo_advantages(rewards, groups, n_groups=2)
    for g in [0, 1]:
        vals = np.asarray(adv)[np.asarray(groups) == g]
        assert abs(vals.mean()) < 1e-5
        assert abs(vals.std() - 1.0) < 1e-2
    assert adv[2] > adv[1] > adv[0]


def test_matches_torch_implementation():
    # Cross-framework parity with the reference grpo.py on the same inputs.
    import torch
    from grpo import grpo_advantages as torch_adv
    rewards = np.array([0.3, -1.2, 2.0, 0.9, 0.0, -0.5], dtype=np.float32)
    groups = np.array([0, 0, 0, 1, 1, 1])
    got = grpo_advantages(jnp.array(rewards), jnp.array(groups), n_groups=2)
    want = torch_adv(torch.tensor(rewards), torch.tensor(groups))
    np.testing.assert_allclose(np.asarray(got), want.numpy(), atol=1e-5)


def test_gradient_pushes_toward_good_completions():
    logp_old = jnp.zeros((2, 3))
    adv = jnp.array([1.0, -1.0])
    mask = jnp.ones((2, 3))
    g = jax.grad(lambda lp: ppo_clip_loss(lp, logp_old, adv, mask))(jnp.zeros((2, 3)))
    assert (np.asarray(g[0]) < 0).all()
    assert (np.asarray(g[1]) > 0).all()


def test_clipping_zeroes_gradient_outside_trust_region():
    logp_old = jnp.zeros((1, 4))
    mask = jnp.ones((1, 4))
    at = jnp.full((1, 4), 0.5)                      # ratio ~ 1.65 > 1.2
    g_pos = jax.grad(lambda lp: ppo_clip_loss(
        lp, logp_old, jnp.array([1.0]), mask))(at)
    np.testing.assert_allclose(np.asarray(g_pos), 0.0, atol=1e-9)
    g_neg = jax.grad(lambda lp: ppo_clip_loss(
        lp, logp_old, jnp.array([-1.0]), mask))(at)
    assert np.abs(np.asarray(g_neg)).sum() > 0      # one-sided clipping


def test_k3_kl_properties():
    logp = jax.random.normal(jax.random.key(0), (4, 6))
    np.testing.assert_allclose(np.asarray(k3_kl(logp, logp)), 0.0, atol=1e-6)
    logp_ref = logp + jax.random.normal(jax.random.key(1), (4, 6)) * 0.5
    assert (np.asarray(k3_kl(logp, logp_ref)) >= -1e-6).all()


def test_full_loss_jits_and_differentiates():
    B, T, G = 6, 8, 2
    key = jax.random.key(0)
    logp = jax.random.normal(key, (B, T)) * 0.1
    logp_old = logp + jax.random.normal(jax.random.key(1), (B, T)) * 0.05
    logp_ref = logp + jax.random.normal(jax.random.key(2), (B, T)) * 0.05
    rewards = jax.random.normal(jax.random.key(3), (B,))
    groups = jnp.array([0, 0, 0, 1, 1, 1])
    mask = jnp.ones((B, T))

    @jax.jit
    def loss_fn(lp):
        return grpo_loss(lp, logp_old, logp_ref, rewards, groups, G, mask)[0]

    loss = loss_fn(logp)
    g = jax.grad(loss_fn)(logp)
    assert np.isfinite(float(loss))
    assert np.isfinite(np.asarray(g)).all()
