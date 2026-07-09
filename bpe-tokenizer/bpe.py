"""Byte-level BPE tokenizer (train / encode / decode), GPT-2 style but minimal.

Pure Python, no dependencies. Base vocabulary is the 256 byte values, so ANY
string round-trips exactly — there is no <unk> token and no unicode edge case:
text is UTF-8 encoded first and merges operate on byte sequences.
"""

import re
from collections import Counter

# Pre-tokenization: split into word chunks and whitespace chunks. Merges never
# cross a chunk boundary, and because every byte of the input lands in some
# chunk, decode is an exact inverse of encode.
_PRETOK = re.compile(rb"\S+|\s+")


def _chunks(text):
    return [tuple(m) for m in _PRETOK.findall(text.encode("utf-8"))]


def _pair_counts(chunk_freqs):
    """Count adjacent-pair frequencies across the corpus, weighted by chunk freq."""
    counts = Counter()
    for chunk, freq in chunk_freqs.items():
        for pair in zip(chunk, chunk[1:]):
            counts[pair] += freq
    return counts


def _merge_chunk(chunk, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `chunk` with `new_id`."""
    out, i = [], 0
    while i < len(chunk):
        if i + 1 < len(chunk) and (chunk[i], chunk[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(chunk[i])
            i += 1
    return tuple(out)


def train(texts, num_merges):
    """Learn `num_merges` BPE merges from an iterable of strings.

    Returns (merges, vocab):
      merges : list of ((id, id), new_id) in the order they were learned —
               encode() MUST replay them in this order.
      vocab  : dict id -> bytes; ids 0..255 are raw bytes, merged ids follow.

    Each round merges the most frequent adjacent pair. Ties break on the
    smaller pair ids — any rule works, but it must be DETERMINISTIC or the
    same corpus trains different tokenizers on different runs.
    """
    chunk_freqs = Counter()
    for text in texts:
        chunk_freqs.update(_chunks(text))

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for _ in range(num_merges):
        counts = _pair_counts(chunk_freqs)
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        new_id = 256 + len(merges)
        merges.append((best, new_id))
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        chunk_freqs = Counter({_merge_chunk(c, best, new_id): f
                               for c, f in chunk_freqs.items()})
    return merges, vocab


def _apply_merges(chunk, ranks, n_merges):
    while len(chunk) > 1:
        # Merge the pair that was learned EARLIEST — not left-to-right scan.
        best = min(zip(chunk, chunk[1:]),
                   key=lambda p: ranks.get(p, (n_merges, None))[0])
        if best not in ranks:
            break
        chunk = _merge_chunk(chunk, best, ranks[best][1])
    return chunk


def encode(text, merges):
    """Apply learned merges within each chunk, always the lowest-rank pair first."""
    ranks = {pair: (rank, new_id) for rank, (pair, new_id) in enumerate(merges)}
    ids = []
    for chunk in _chunks(text):
        ids.extend(_apply_merges(chunk, ranks, len(merges)))
    return ids


def decode(ids, vocab):
    """Concatenate token byte-strings, UTF-8 decode once at the end.

    Decoding per-token would break: a multi-byte UTF-8 character can be split
    across two tokens, and each half alone is invalid UTF-8.
    """
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
