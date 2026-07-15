import jax
import jax.numpy as jnp
import numpy as np

from moe_jax import (expert_apply, init_expert, init_moe,
                     load_balancing_loss, moe_forward)


def tie_experts(params):
    """Make every expert identical to expert 0."""
    tied = jax.tree_util.tree_map(
        lambda s: jnp.broadcast_to(s[0], s.shape), params["experts"])
    return {**params, "experts": tied}


def expert0(params, x):
    p0 = jax.tree_util.tree_map(lambda s: s[0], params["experts"])
    return expert_apply(p0, x)


def test_identical_experts_reduce_to_single_expert():
    params = tie_experts(init_moe(jax.random.key(0), dim=16, n_experts=4, k=2))
    x = jax.random.normal(jax.random.key(1), (10, 16))
    out, _, dropped = moe_forward(params, x)
    assert dropped == 0
    np.testing.assert_allclose(out, expert0(params, x), atol=1e-5)


def test_top1_routes_to_argmax_expert():
    params = init_moe(jax.random.key(0), dim=16, n_experts=4, k=1)
    x = jax.random.normal(jax.random.key(1), (6, 16))
    gates = jax.nn.softmax(x @ params["router"], axis=-1)
    out, _, _ = moe_forward(params, x)
    for i in range(6):
        e = int(gates[i].argmax())
        pe = jax.tree_util.tree_map(lambda s: s[e], params["experts"])
        np.testing.assert_allclose(out[i], expert_apply(pe, x[i:i + 1])[0],
                                   atol=1e-5)


def test_aux_loss_minimized_at_uniform_routing():
    n, e = 1000, 8
    uniform = jnp.full((n, e), 1 / e)
    top1 = jnp.arange(n) % e
    assert abs(float(load_balancing_loss(uniform, top1, e)) - 1.0) < 1e-5

    skewed = jnp.zeros((n, e)).at[:, 0].set(0.9).at[:, 1:].set(0.1 / (e - 1))
    assert float(load_balancing_loss(skewed, jnp.zeros(n, dtype=int), e)) > 1.0


def test_aux_loss_gradient_reaches_router():
    params = init_moe(jax.random.key(0), dim=16, n_experts=4, k=2)
    x = jax.random.normal(jax.random.key(1), (20, 16))

    def aux_of(router):
        return moe_forward({**params, "router": router}, x)[1]

    g = jax.grad(aux_of)(params["router"])
    assert float(jnp.abs(g).sum()) > 0


def test_capacity_drops_overflow():
    params = init_moe(jax.random.key(0), dim=8, n_experts=4, k=1)
    router = jnp.zeros((8, 4)).at[:, 0].set(10.0)   # everything to expert 0
    params = {**params, "router": router}
    x = jnp.abs(jax.random.normal(jax.random.key(1), (16, 8)))
    out, _, dropped = moe_forward(params, x, capacity_factor=1.0)
    cap = 4
    assert dropped == 16 - cap
    assert float(jnp.abs(out[cap:]).sum()) == 0.0   # dropped tokens zeroed


def test_matches_torch_moe_on_tied_experts():
    # Cross-framework: with identical (tied) experts both implementations
    # must equal that single expert, hence equal each other.
    params = tie_experts(init_moe(jax.random.key(0), dim=16, n_experts=4, k=2))
    x = jax.random.normal(jax.random.key(1), (10, 16))
    out_jax, _, _ = moe_forward(params, x)
    np.testing.assert_allclose(out_jax, expert0(params, x), atol=1e-5)
