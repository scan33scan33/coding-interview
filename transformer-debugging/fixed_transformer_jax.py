"""JAX port of the fixed reference transformer (see fixed_transformer.py).

Functional: params are a pytree, forward/loss are pure functions, training
uses an inline Adam and jax.jit. Same copy task, same architecture, same
irreducible-loss floor (~0.96).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

VOCAB, SEP = 12, 11
PREFIX_LEN = 5
SEQ_LEN = 2 * PREFIX_LEN + 1


def attention(q_BHLK, k_BHLK, v_BHLK):
    K = q_BHLK.shape[-1]
    scores_BHLL = q_BHLK @ k_BHLK.swapaxes(-1, -2) / jnp.sqrt(K).astype(q_BHLK.dtype)
    L = q_BHLK.shape[2]
    causal_LL = jnp.triu(jnp.ones((L, L), dtype=bool), k=1)
    scores_BHLL = jnp.where(causal_LL, -jnp.inf, scores_BHLL)
    return jax.nn.softmax(scores_BHLL, axis=-1) @ v_BHLK


def _linear(key, n_in, n_out):
    return {"w": jax.random.normal(key, (n_in, n_out)) * (2.0 / n_in) ** 0.5,
            "b": jnp.zeros(n_out)}


def init_params(key, dim=32, n_heads=2, n_layers=2):
    keys = jax.random.split(key, 4 * n_layers + 3)
    blocks = []
    for i in range(n_layers):
        base = 3 + 4 * i
        blocks.append({
            "ln1_g": jnp.ones(dim), "ln1_b": jnp.zeros(dim),
            "ln2_g": jnp.ones(dim), "ln2_b": jnp.zeros(dim),
            "qkv": _linear(keys[base], dim, 3 * dim),
            "proj": _linear(keys[base + 1], dim, dim),
            "up": _linear(keys[base + 2], dim, 4 * dim),
            "down": _linear(keys[base + 3], 4 * dim, dim),
        })
    return {
        "tok_emb": jax.random.normal(keys[0], (VOCAB, dim)) * 0.02,
        "pos_emb": jax.random.normal(keys[1], (SEQ_LEN, dim)) * 0.02,
        "head": _linear(keys[2], dim, VOCAB),
        "lnf_g": jnp.ones(dim), "lnf_b": jnp.zeros(dim),
        "blocks": blocks, "n_heads": n_heads,
    }


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * g + b


def block_forward(blk, x_BLD, n_heads):
    B, L, D = x_BLD.shape
    K = D // n_heads
    h = layer_norm(x_BLD, blk["ln1_g"], blk["ln1_b"])
    qkv = (h @ blk["qkv"]["w"] + blk["qkv"]["b"]).reshape(B, L, 3, n_heads, K)
    q, k, v = (qkv[:, :, i].swapaxes(1, 2) for i in range(3))
    att = attention(q, k, v).swapaxes(1, 2).reshape(B, L, D)
    x_BLD = x_BLD + att @ blk["proj"]["w"] + blk["proj"]["b"]

    h = layer_norm(x_BLD, blk["ln2_g"], blk["ln2_b"])
    mlp = jax.nn.gelu(h @ blk["up"]["w"] + blk["up"]["b"])
    mlp = mlp @ blk["down"]["w"] + blk["down"]["b"]
    return x_BLD + mlp                              # residual around MLP


def forward(params, ids_BL):
    x_BLD = params["tok_emb"][ids_BL] + params["pos_emb"][:ids_BL.shape[1]]
    for blk in params["blocks"]:
        x_BLD = block_forward(blk, x_BLD, params["n_heads"])
    x_BLD = layer_norm(x_BLD, params["lnf_g"], params["lnf_b"])
    return x_BLD @ params["head"]["w"] + params["head"]["b"]


def loss_fn(params, ids_BL):
    logits_BLV = forward(params, ids_BL)
    logp = jax.nn.log_softmax(logits_BLV[:, :-1], axis=-1)
    tgt = ids_BL[:, 1:]                             # SHIFTED labels
    picked = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    return -picked.mean()


def make_copy_batch(batch_size, rng):
    prefix = rng.integers(0, SEP, (batch_size, PREFIX_LEN))
    sep = np.full((batch_size, 1), SEP)
    return jnp.array(np.concatenate([prefix, sep, prefix], axis=1))


def train_copy_task(steps=300, batch_size=64, lr=3e-3, seed=0):
    params = init_params(jax.random.key(seed))
    n_heads = params.pop("n_heads")                 # static for jit
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed)

    @partial(jax.jit, static_argnums=())
    def step_fn(params, m, v, t, ids):
        loss, grads = jax.value_and_grad(
            lambda p: loss_fn({**p, "n_heads": n_heads}, ids))(params)
        m = jax.tree_util.tree_map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
        v = jax.tree_util.tree_map(lambda a, g: 0.999 * a + 0.001 * g * g, v, grads)
        bc1, bc2 = 1 - 0.9 ** t, 1 - 0.999 ** t
        params = jax.tree_util.tree_map(
            lambda p, m_, v_: p - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + 1e-8),
            params, m, v)
        return params, m, v, loss

    loss = None
    for t in range(1, steps + 1):
        ids = make_copy_batch(batch_size, rng)
        params, m, v, loss = step_fn(params, m, v, jnp.float32(t), ids)
    return {**params, "n_heads": n_heads}, float(loss)


def greedy_generate(params, prompt_BL, n_tokens):
    ids = np.asarray(prompt_BL)
    for _ in range(n_tokens):
        next_B = np.asarray(forward(params, jnp.array(ids))[:, -1].argmax(-1))
        ids = np.concatenate([ids, next_B[:, None]], axis=1)
    return ids
