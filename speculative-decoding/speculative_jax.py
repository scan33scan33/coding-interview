"""JAX port of speculative decoding (see speculative.py for the NumPy
original and the full commentary/derivation). Randomness threads explicit
PRNG keys; the caller passes one key per step/generation."""

import jax
import jax.numpy as jnp

from speculative import expected_speedup  # noqa: F401  (pure math, shared)


def speculative_step(target_fn, draft_fn, prefix, gamma, key):
    ctx = list(prefix)
    draft_tokens, q_dists = [], []
    for _ in range(gamma):
        key, sub = jax.random.split(key)
        q_V = draft_fn(ctx)
        tok = int(jax.random.choice(sub, q_V.shape[0], p=q_V))
        draft_tokens.append(tok)
        q_dists.append(q_V)
        ctx.append(tok)

    p_dists = []
    ctx = list(prefix)
    for tok in draft_tokens:
        p_dists.append(target_fn(ctx))
        ctx.append(tok)
    p_final_V = target_fn(ctx)

    accepted = []
    for i, tok in enumerate(draft_tokens):
        p, q = float(p_dists[i][tok]), float(q_dists[i][tok])
        key, sub = jax.random.split(key)
        if float(jax.random.uniform(sub)) < min(1.0, p / q):
            accepted.append(tok)
            continue
        res_V = jnp.clip(p_dists[i] - q_dists[i], min=0.0)
        res_V = res_V / res_V.sum()
        key, sub = jax.random.split(key)
        accepted.append(int(jax.random.choice(sub, res_V.shape[0], p=res_V)))
        return accepted, {"drafted": gamma, "accepted": i}

    key, sub = jax.random.split(key)
    accepted.append(int(jax.random.choice(sub, p_final_V.shape[0], p=p_final_V)))
    return accepted, {"drafted": gamma, "accepted": gamma}


def generate(target_fn, draft_fn, prefix, n_tokens, gamma, key):
    out = list(prefix)
    drafted = accepted = 0
    while len(out) - len(prefix) < n_tokens:
        key, sub = jax.random.split(key)
        new, stats = speculative_step(target_fn, draft_fn, out, gamma, sub)
        out.extend(new)
        drafted += stats["drafted"]
        accepted += stats["accepted"]
    return out[len(prefix):len(prefix) + n_tokens], accepted / max(drafted, 1)
