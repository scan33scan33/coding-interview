# Byte-Level BPE Tokenizer From Scratch

**Problem:** Implement a **byte-level byte-pair-encoding (BPE)** tokenizer —
train, encode, decode — in pure Python. This is the tokenization scheme behind
GPT-2/3/4, Llama, and essentially every frontier LLM.

The must-pass correctness bar: `decode(encode(text)) == text` for **any**
string, including text containing bytes never seen during training.

---

## Core Requirements

1. **Byte-level base vocabulary**
   - IDs `0..255` are the raw byte values; merged tokens get ids `256, 257, ...`.
   - UTF-8 encode the text first, run BPE on bytes. No `<unk>` token, no
     unicode special-casing.

2. **Training (`train`)**
   - Pre-tokenize into chunks (words / whitespace runs); merges never cross a
     chunk boundary.
   - Each round: count adjacent-pair frequencies weighted by chunk frequency,
     merge the most frequent pair everywhere, record `((id, id), new_id)`.
   - **Deterministic tie-breaking** — otherwise the same corpus produces
     different tokenizers on different runs.

3. **Encoding (`encode`)**
   - Replay merges by **training rank**: at each step merge the pair that was
     learned *earliest*, not the leftmost matching pair. This is what makes
     encoding consistent with how the vocab was built.

4. **Decoding (`decode`)**
   - Concatenate token byte-strings, then UTF-8 decode **once at the end** —
     a multi-byte character can be split across tokens, and each half alone
     is invalid UTF-8.

---

## Interface

```python
def train(texts, num_merges) -> (merges, vocab)
    # merges: [((id, id), new_id), ...] in learned order
    # vocab:  {id: bytes}, ids 0..255 = raw bytes

def encode(text, merges) -> list[int]
def decode(ids, vocab) -> str
```

---

## Behavior Notes / Gotchas

- **Encode by rank, not left-to-right.** Greedy left-to-right merging gives
  different (worse) tokenizations than training produced. The invariant:
  among all currently-adjacent pairs, always merge the one with the lowest
  training rank. `test_encode_replays_training_order` catches the scan bug.
- **Per-token decode is broken by design.** `"✨"` is 3 bytes; after merges it
  may end up as 2 tokens. Decode must join bytes first, decode UTF-8 last.
- **Whitespace is data.** Pre-tokenizing with `str.split()` silently destroys
  spacing and round-trip fails on `"a  b"`. Chunk with a pattern that covers
  every byte (here `\S+|\s+`; GPT-2 uses a fancier regex that glues a leading
  space onto each word so `" the"` becomes one token).
- **Ties must break deterministically.** `Counter.most_common` order depends
  on insertion order — fine within one process, but make the rule explicit.
- **Complexity.** This implementation recounts all pairs every merge:
  O(merges × corpus). Production trainers (SentencePiece, tiktoken's trainer)
  incrementally update counts with a heap — a classic follow-up.

---

## Running the Smoke Test

```bash
pip install pytest
python -m pytest test_bpe.py -v
```

| Test | Validates |
|------|-----------|
| `test_roundtrip_ascii` | Basic encode/decode inverse |
| `test_roundtrip_unseen_text_and_unicode` | No `<unk>`: unseen bytes, emoji, CJK, whitespace runs |
| `test_no_merges_yields_raw_bytes` | Base vocab is exactly the 256 bytes |
| `test_most_frequent_pair_merged_first` | Greedy frequency objective |
| `test_merges_never_cross_word_boundaries` | Pre-tokenization respected |
| `test_encode_replays_training_order` | Rank-order (not scan-order) merging |
| `test_compression_on_repetitive_corpus` | Learned merges actually compress |
| `test_training_is_deterministic` | Stable tie-breaking |

---

## Discussion Questions (interview follow-ups)

- **Why bytes, not characters?** What breaks with a character-level base
  vocab on multilingual text, and why did GPT-2 map bytes to printable
  unicode instead of using raw bytes?
- **Vocab size trade-off** — what happens to sequence length, embedding-table
  size, and rare-token undertraining as vocab grows from 32k to 256k?
- **Regex pre-tokenization** — why does GPT-4's splitter treat digits
  specially (max 3 per token)? What goes wrong with arithmetic otherwise?
- **Tokenizer/model mismatch** — why can't you swap tokenizers on a trained
  model, and what does "token healing" fix at inference time?
- **Speeding up training** — sketch the heap + linked-list algorithm that
  avoids recounting all pairs after each merge.
