"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A small DRAFT model proposes gamma tokens autoregressively; the large TARGET
model scores all of them in ONE forward pass; a rejection-sampling rule keeps
a prefix of the proposals. The guarantee — and the whole point — is that the
output distribution is EXACTLY the target model's, no matter how bad the
draft is. The draft only affects speed, never correctness.

Models here are abstracted as functions prefix -> next-token probabilities,
so the sampling math (not transformer plumbing) is what's implemented.

Shape suffix convention:  V = vocab size
"""

import numpy as np


def speculative_step(target_fn, draft_fn, prefix, gamma, rng):
    """One draft-then-verify round.

    Returns (accepted_tokens, stats) where accepted_tokens contains the kept
    draft tokens plus exactly one token from the target distribution (the
    correction/bonus token), so every round emits >= 1 token.

    Acceptance rule for draft token x with target prob p(x), draft prob q(x):
        accept with probability min(1, p(x) / q(x))
    On rejection at position i, resample from the RESIDUAL distribution
        r(x) ∝ max(0, p_i(x) - q_i(x))
    and stop. If all gamma drafts are accepted, sample one bonus token from
    the target's distribution at the final position (it was computed in the
    same batched target pass — free).
    """
    # Draft proposes gamma tokens autoregressively (cheap model, cheap loop).
    ctx = list(prefix)
    draft_tokens, q_dists = [], []
    for _ in range(gamma):
        q_V = draft_fn(ctx)
        tok = int(rng.choice(len(q_V), p=q_V))
        draft_tokens.append(tok)
        q_dists.append(q_V)
        ctx.append(tok)

    # Target scores every position in ONE call in the real system; here we
    # just collect its distributions at each prefix.
    p_dists = []
    ctx = list(prefix)
    for tok in draft_tokens:
        p_dists.append(target_fn(ctx))
        ctx.append(tok)
    p_final_V = target_fn(ctx)  # position after the last draft token

    accepted = []
    for i, tok in enumerate(draft_tokens):
        p, q = p_dists[i][tok], q_dists[i][tok]
        if rng.uniform() < min(1.0, p / q):
            accepted.append(tok)
            continue
        # Rejected: sample from the residual max(0, p - q), normalized.
        # This is what makes the combined procedure EXACT: the accepted mass
        # min(p, q) plus the residual mass reconstructs p precisely.
        res_V = np.maximum(p_dists[i] - q_dists[i], 0.0)
        res_V /= res_V.sum()
        accepted.append(int(rng.choice(len(res_V), p=res_V)))
        return accepted, {"drafted": gamma, "accepted": i}

    # All gamma accepted: bonus token from the target's next distribution.
    accepted.append(int(rng.choice(len(p_final_V), p=p_final_V)))
    return accepted, {"drafted": gamma, "accepted": gamma}


def generate(target_fn, draft_fn, prefix, n_tokens, gamma, rng):
    """Generate n_tokens, returning (tokens, acceptance_rate)."""
    out = list(prefix)
    drafted = accepted = 0
    while len(out) - len(prefix) < n_tokens:
        new, stats = speculative_step(target_fn, draft_fn, out, gamma, rng)
        out.extend(new)
        drafted += stats["drafted"]
        accepted += stats["accepted"]
    return out[len(prefix):len(prefix) + n_tokens], accepted / max(drafted, 1)


def expected_speedup(acceptance_rate, gamma, cost_ratio):
    """Expected tokens per target-model call vs. 1 for vanilla decoding.

    With per-token acceptance a, one round emits E = (1 - a^(gamma+1))/(1 - a)
    tokens for 1 target call + gamma draft calls; cost_ratio = draft/target
    per-call cost. Returns E / (1 + gamma * cost_ratio).
    """
    a = min(acceptance_rate, 1.0 - 1e-12)
    e_tokens = (1 - a ** (gamma + 1)) / (1 - a)
    return e_tokens / (1 + gamma * cost_ratio)
