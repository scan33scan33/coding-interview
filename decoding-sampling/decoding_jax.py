"""JAX port of the decoding toolbox (see decoding.py for the NumPy original
and full commentary). Sampling threads explicit PRNG keys, JAX-style."""

import jax
import jax.numpy as jnp


def apply_temperature(logits_V, temperature):
    if temperature <= 0:
        raise ValueError("use greedy decoding for temperature 0")
    return logits_V / temperature


def top_k_filter(logits_V, k):
    if k >= logits_V.shape[0]:
        return logits_V
    kth = jax.lax.top_k(logits_V, k)[0][-1]
    return jnp.where(logits_V < kth, -jnp.inf, logits_V)


def top_p_filter(logits_V, p):
    probs_V = jax.nn.softmax(logits_V)
    order = jnp.argsort(-probs_V)
    csum = jnp.cumsum(probs_V[order])
    cutoff = int(jnp.searchsorted(csum, p)) + 1
    keep = order[:max(cutoff, 1)]
    return jnp.full_like(logits_V, -jnp.inf).at[keep].set(logits_V[keep])


def apply_repetition_penalty(logits_V, prev_ids, penalty):
    out_V = logits_V
    for t in set(int(i) for i in prev_ids):
        out_V = out_V.at[t].set(
            jnp.where(out_V[t] > 0, out_V[t] / penalty, out_V[t] * penalty))
    return out_V


def sample_token(logits_V, key, temperature=1.0, top_k=None, top_p=None,
                 prev_ids=(), repetition_penalty=1.0):
    logits_V = logits_V.astype(jnp.float32)
    if repetition_penalty != 1.0:
        logits_V = apply_repetition_penalty(logits_V, prev_ids, repetition_penalty)
    logits_V = apply_temperature(logits_V, temperature)
    if top_k is not None:
        logits_V = top_k_filter(logits_V, top_k)
    if top_p is not None:
        logits_V = top_p_filter(logits_V, top_p)
    return int(jax.random.categorical(key, logits_V))


def greedy_decode(step_fn, bos, eos, max_len):
    seq = [bos]
    for _ in range(max_len):
        seq.append(int(step_fn(seq).argmax()))
        if seq[-1] == eos:
            break
    return seq


def beam_search(step_fn, bos, eos, beam_width, max_len, length_alpha=0.0):
    beams = [([bos], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logp_V = jax.nn.log_softmax(step_fn(seq).astype(jnp.float64))
            vals, idxs = jax.lax.top_k(logp_V, beam_width)
            for tok, lp in zip(idxs.tolist(), vals.tolist()):
                candidates.append((seq + [tok], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)

        beams = []
        for seq, score in candidates:
            if seq[-1] == eos:
                norm = (len(seq) - 1) ** length_alpha if length_alpha else 1.0
                finished.append((seq, score / norm))
            else:
                beams.append((seq, score))
            if len(beams) == beam_width:
                break
        if not beams:
            break
    for seq, score in beams:
        norm = (len(seq) - 1) ** length_alpha if length_alpha else 1.0
        finished.append((seq, score / norm))
    return max(finished, key=lambda c: c[1])
