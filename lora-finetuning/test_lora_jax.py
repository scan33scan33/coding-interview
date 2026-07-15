import jax
import jax.numpy as jnp
import numpy as np

from lora_jax import base_apply, init_lora, lora_apply, merge, unmerge


def make_base(key, in_f=16, out_f=8):
    kw, kb = jax.random.split(key)
    return {"w": jax.random.normal(kw, (out_f, in_f)) * 0.3,
            "b": jax.random.normal(kb, (out_f,)) * 0.1}


def test_init_is_identity_wrt_base():
    base = make_base(jax.random.key(0))
    lora = init_lora(jax.random.key(1), 16, 8, r=4)
    x = jax.random.normal(jax.random.key(2), (5, 16))
    np.testing.assert_allclose(lora_apply(base, lora, x),
                               base_apply(base, x), atol=1e-6)


def test_gradients_flow_only_where_differentiated():
    # Freezing by construction: grad wrt the lora pytree only.
    base = make_base(jax.random.key(0))
    lora = init_lora(jax.random.key(1), 16, 8, r=4)
    x = jax.random.normal(jax.random.key(2), (3, 16))

    g = jax.grad(lambda lo: lora_apply(base, lo, x).sum())(lora)
    # At init B == 0, so dL/dA (which flows THROUGH B) is exactly zero while
    # dL/dB is not — precisely why only-one-factor-zero init still trains.
    assert float(jnp.abs(g["A"]).sum()) == 0
    assert float(jnp.abs(g["B"]).sum()) > 0

    lora2 = {**lora, "B": jax.random.normal(jax.random.key(9), (8, 4))}
    g2 = jax.grad(lambda lo: lora_apply(base, lo, x).sum())(lora2)
    assert float(jnp.abs(g2["A"]).sum()) > 0    # nonzero B unblocks A's grad


def test_merge_matches_and_unmerge_restores():
    base = make_base(jax.random.key(0))
    lora = init_lora(jax.random.key(1), 16, 8, r=4)
    lora = {**lora, "B": jax.random.normal(jax.random.key(3), (8, 4))}
    x = jax.random.normal(jax.random.key(2), (5, 16))

    want = lora_apply(base, lora, x)
    merged = merge(base, lora)
    np.testing.assert_allclose(base_apply(merged, x), want, atol=1e-5)

    restored = unmerge(merged, lora)
    np.testing.assert_allclose(restored["w"], base["w"], atol=1e-6)


def test_finetuning_moves_adapter_not_base():
    base = make_base(jax.random.key(0), 32, 1)
    lora = init_lora(jax.random.key(1), 32, 1, r=4)
    x = jax.random.normal(jax.random.key(2), (64, 32))
    y = x.sum(axis=-1, keepdims=True) / 6.0     # scaled: keeps plain SGD stable
    base_w0 = base["w"].copy()

    def loss_fn(lo):
        return ((lora_apply(base, lo, x) - y) ** 2).mean()

    first = float(loss_fn(lora))
    for _ in range(500):
        g = jax.grad(loss_fn)(lora)
        lora = {**lora, "A": lora["A"] - 0.02 * g["A"],
                "B": lora["B"] - 0.02 * g["B"]}
    assert float(loss_fn(lora)) < first * 0.1
    np.testing.assert_array_equal(base["w"], base_w0)   # base untouched
