"""PyTorch port of speculative decoding (see speculative.py for the NumPy
original and the full commentary/derivation)."""

import torch

from speculative import expected_speedup  # noqa: F401  (pure math, shared)


def speculative_step(target_fn, draft_fn, prefix, gamma, generator):
    ctx = list(prefix)
    draft_tokens, q_dists = [], []
    for _ in range(gamma):
        q_V = draft_fn(ctx)
        tok = int(torch.multinomial(q_V, 1, generator=generator))
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
        u = float(torch.rand((), generator=generator))
        if u < min(1.0, p / q):
            accepted.append(tok)
            continue
        res_V = (p_dists[i] - q_dists[i]).clamp(min=0.0)
        res_V = res_V / res_V.sum()
        accepted.append(int(torch.multinomial(res_V, 1, generator=generator)))
        return accepted, {"drafted": gamma, "accepted": i}

    accepted.append(int(torch.multinomial(p_final_V, 1, generator=generator)))
    return accepted, {"drafted": gamma, "accepted": gamma}


def generate(target_fn, draft_fn, prefix, n_tokens, gamma, generator):
    out = list(prefix)
    drafted = accepted = 0
    while len(out) - len(prefix) < n_tokens:
        new, stats = speculative_step(target_fn, draft_fn, out, gamma, generator)
        out.extend(new)
        drafted += stats["drafted"]
        accepted += stats["accepted"]
    return out[len(prefix):len(prefix) + n_tokens], accepted / max(drafted, 1)
