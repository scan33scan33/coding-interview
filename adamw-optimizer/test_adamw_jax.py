import jax
import jax.numpy as jnp
import numpy as np
import torch

from adamw_jax import adamw_init, adamw_update, clip_grad_global_norm


def test_matches_torch_adamw_exactly():
    np.random.seed(0)
    base = [np.random.randn(4, 3).astype(np.float32),
            np.random.randn(7).astype(np.float32)]
    jax_params = {"a": jnp.array(base[0]), "b": jnp.array(base[1])}
    torch_params = [torch.tensor(base[0], requires_grad=True),
                    torch.tensor(base[1], requires_grad=True)]
    ref = torch.optim.AdamW(torch_params, lr=3e-3, betas=(0.9, 0.99),
                            eps=1e-8, weight_decay=0.05)
    state = adamw_init(jax_params)

    for step in range(25):
        np.random.seed(100 + step)
        grads_np = [np.random.randn(*p.shape).astype(np.float32) for p in base]
        jax_grads = {"a": jnp.array(grads_np[0]), "b": jnp.array(grads_np[1])}
        for p, g in zip(torch_params, grads_np):
            p.grad = torch.tensor(g)
        jax_params, state = adamw_update(jax_grads, state, jax_params,
                                         lr=3e-3, b1=0.9, b2=0.99,
                                         eps=1e-8, weight_decay=0.05)
        ref.step()

    np.testing.assert_allclose(np.asarray(jax_params["a"]),
                               torch_params[0].detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(jax_params["b"]),
                               torch_params[1].detach().numpy(), atol=1e-5)


def test_update_is_jittable():
    params = {"w": jnp.ones((3, 3))}
    state = adamw_init(params)
    jit_update = jax.jit(adamw_update)
    params2, state2 = jit_update({"w": jnp.ones((3, 3))}, state, params)
    assert params2["w"].shape == (3, 3)
    assert int(state2["t"]) == 1


def test_global_norm_clipping_preserves_direction():
    grads = {"a": jnp.array([3.0, 0.0, 0.0]), "b": jnp.array([4.0])}
    clipped, pre = clip_grad_global_norm(grads, max_norm=1.0)
    assert abs(float(pre) - 5.0) < 1e-5
    np.testing.assert_allclose(clipped["a"], [0.6, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(clipped["b"], [0.8], atol=1e-6)


def test_clipping_is_noop_under_threshold():
    grads = {"a": jnp.array([0.3, 0.4])}
    clipped, _ = clip_grad_global_norm(grads, max_norm=1.0)
    np.testing.assert_allclose(clipped["a"], [0.3, 0.4], atol=1e-7)


def test_optimizes_a_quadratic():
    params = {"p": jnp.array([5.0, -3.0])}
    target = jnp.array([1.0, 2.0])
    state = adamw_init(params)

    @jax.jit
    def train_step(params, state):
        loss, grads = jax.value_and_grad(
            lambda pr: ((pr["p"] - target) ** 2).sum())(params)
        params, state = adamw_update(grads, state, params, lr=0.1,
                                     weight_decay=0.0)
        return params, state, loss

    for _ in range(500):
        params, state, loss = train_step(params, state)
    np.testing.assert_allclose(np.asarray(params["p"]), [1.0, 2.0], atol=1e-3)
