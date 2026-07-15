"""PyTorch port of the decoding toolbox (see decoding.py for the NumPy
original and full commentary)."""

import torch


def apply_temperature(logits_V, temperature):
    if temperature <= 0:
        raise ValueError("use greedy decoding for temperature 0")
    return logits_V / temperature


def top_k_filter(logits_V, k):
    if k >= logits_V.numel():
        return logits_V.clone()
    kth = logits_V.topk(k).values[-1]
    return logits_V.masked_fill(logits_V < kth, float("-inf"))


def top_p_filter(logits_V, p):
    probs_V = torch.softmax(logits_V, dim=-1)
    sorted_p, order = probs_V.sort(descending=True)
    csum = sorted_p.cumsum(-1)
    cutoff = int(torch.searchsorted(csum, torch.tensor(p))) + 1
    keep = order[:max(cutoff, 1)]
    out_V = torch.full_like(logits_V, float("-inf"))
    out_V[keep] = logits_V[keep]
    return out_V


def apply_repetition_penalty(logits_V, prev_ids, penalty):
    out_V = logits_V.clone()
    for t in set(int(i) for i in prev_ids):
        out_V[t] = out_V[t] / penalty if out_V[t] > 0 else out_V[t] * penalty
    return out_V


def sample_token(logits_V, generator, temperature=1.0, top_k=None, top_p=None,
                 prev_ids=(), repetition_penalty=1.0):
    logits_V = logits_V.double()
    if repetition_penalty != 1.0:
        logits_V = apply_repetition_penalty(logits_V, prev_ids, repetition_penalty)
    logits_V = apply_temperature(logits_V, temperature)
    if top_k is not None:
        logits_V = top_k_filter(logits_V, top_k)
    if top_p is not None:
        logits_V = top_p_filter(logits_V, top_p)
    probs_V = torch.softmax(logits_V, dim=-1)
    return int(torch.multinomial(probs_V, 1, generator=generator))


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
            logp_V = torch.log_softmax(step_fn(seq).double(), dim=-1)
            top = logp_V.topk(beam_width)
            for tok, lp in zip(top.indices.tolist(), top.values.tolist()):
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
