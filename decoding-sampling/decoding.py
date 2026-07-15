"""LLM decoding strategies: temperature, top-k, top-p, repetition penalty,
and beam search. Pure NumPy — the model is abstracted as a function that
maps a prefix to next-token logits.

Shape suffix convention:
  V = vocab size, W = beam width
"""

import numpy as np


def softmax(logits_V):
    z_V = logits_V - logits_V.max()
    e_V = np.exp(z_V)
    return e_V / e_V.sum()


def log_softmax(logits_V):
    z_V = logits_V - logits_V.max()
    return z_V - np.log(np.exp(z_V).sum())


def apply_temperature(logits_V, temperature):
    """temperature -> 0 approaches argmax; > 1 flattens. Must not change argmax."""
    if temperature <= 0:
        raise ValueError("use greedy decoding for temperature 0")
    return logits_V / temperature


def top_k_filter(logits_V, k):
    """Keep the k highest logits, set the rest to -inf."""
    if k >= logits_V.size:
        return logits_V.copy()
    kth = np.partition(logits_V, -k)[-k]
    out_V = np.where(logits_V < kth, -np.inf, logits_V)
    return out_V


def top_p_filter(logits_V, p):
    """Nucleus sampling: keep the SMALLEST set of tokens whose cumulative
    probability reaches p; everything else to -inf.

    The token that crosses the threshold is INCLUDED (otherwise kept mass
    could be < p), and the top-1 token is always kept even if its prob > p.
    """
    probs_V = softmax(logits_V)
    order = np.argsort(-probs_V)
    csum = np.cumsum(probs_V[order])
    # First position where cumulative mass reaches p; keep through it.
    cutoff = int(np.searchsorted(csum, p)) + 1
    keep = order[:max(cutoff, 1)]
    out_V = np.full_like(logits_V, -np.inf)
    out_V[keep] = logits_V[keep]
    return out_V


def apply_repetition_penalty(logits_V, prev_ids, penalty):
    """CTRL-style: for already-generated tokens, divide positive logits by
    `penalty` and multiply negative ones — both moves REDUCE the token's
    probability. A plain subtraction would flip sign behavior for negatives.
    """
    out_V = logits_V.copy()
    for t in set(prev_ids):
        out_V[t] = out_V[t] / penalty if out_V[t] > 0 else out_V[t] * penalty
    return out_V


def sample_token(logits_V, rng, temperature=1.0, top_k=None, top_p=None,
                 prev_ids=(), repetition_penalty=1.0):
    """Full sampling pipeline. Filter order matters and mirrors HF:
    repetition penalty -> temperature -> top-k -> top-p -> sample.
    """
    logits_V = np.asarray(logits_V, dtype=np.float64)
    if repetition_penalty != 1.0:
        logits_V = apply_repetition_penalty(logits_V, prev_ids, repetition_penalty)
    logits_V = apply_temperature(logits_V, temperature)
    if top_k is not None:
        logits_V = top_k_filter(logits_V, top_k)
    if top_p is not None:
        logits_V = top_p_filter(logits_V, top_p)
    return int(rng.choice(logits_V.size, p=softmax(logits_V)))


def greedy_decode(step_fn, bos, eos, max_len):
    seq = [bos]
    for _ in range(max_len):
        seq.append(int(np.argmax(step_fn(seq))))
        if seq[-1] == eos:
            break
    return seq


def beam_search(step_fn, bos, eos, beam_width, max_len, length_alpha=0.0):
    """Standard beam search over log-probs.

    step_fn(prefix) -> next-token logits (V,).
    Scores are sums of token log-probs; finished beams are ranked by
    score / len(tokens)^length_alpha (GNMT length normalization, alpha=0
    disables it). Without it, beam search systematically prefers short
    sequences because every extra token adds a negative log-prob.

    Returns the best (sequence, normalized_score).
    """
    beams = [([bos], 0.0)]           # (sequence, cumulative logprob)
    finished = []

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logp_V = log_softmax(np.asarray(step_fn(seq), dtype=np.float64))
            # Expanding only the top beam_width tokens per beam is safe:
            # a candidate outside a beam's own top-W can never be in the
            # global top-W of this step.
            for tok in np.argsort(-logp_V)[:beam_width]:
                candidates.append((seq + [int(tok)], score + logp_V[tok]))
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
        if not beams:                 # every surviving beam has finished
            break

    for seq, score in beams:          # ran out of length: rank as-is
        norm = (len(seq) - 1) ** length_alpha if length_alpha else 1.0
        finished.append((seq, score / norm))
    return max(finished, key=lambda c: c[1])
