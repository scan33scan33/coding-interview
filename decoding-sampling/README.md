# LLM Decoding Strategies: Temperature, Top-k, Top-p, Beam Search

**Problem:** Implement the standard LLM decoding toolbox from scratch in NumPy:
temperature scaling, **top-k** and **top-p (nucleus)** filtering, a
sign-correct **repetition penalty**, and **beam search** with GNMT length
normalization. The model is abstracted as `step_fn(prefix) -> logits`, so the
focus is purely on the decoding logic.

One of the most frequently reported ML-coding questions at frontier labs
(OpenAI/Anthropic loops per 1point3acres, Blind, and MLE interview blogs).

---

## Core Requirements

1. **Filters (compose in HF order)**
   `repetition penalty → temperature → top-k → top-p → sample`
   - `apply_temperature` — divides logits; must preserve the argmax.
   - `top_k_filter` — keep the k highest logits, others to `−inf`; `k ≥ V` is
     a no-op.
   - `top_p_filter` — keep the **smallest** set of tokens whose cumulative
     probability reaches `p`. The token that *crosses* the threshold is
     included; the top-1 token is always kept.
   - `apply_repetition_penalty` — CTRL-style: divide positive logits by the
     penalty, **multiply** negative ones. Both moves must reduce probability.

2. **Beam search (`beam_search`)**
   - Track `beam_width` partial hypotheses by cumulative log-prob.
   - Expanding only each beam's own top-`W` tokens is safe (prove it).
   - A beam that emits EOS moves to a `finished` pool and frees its slot.
   - Rank finished hypotheses by `score / len^alpha` (GNMT); `alpha=0`
     disables normalization.

3. **Correct sampling** — after filtering, sample from the renormalized
   softmax; empirical frequencies must match it.

---

## Interface

```python
def sample_token(logits_V, rng, temperature=1.0, top_k=None, top_p=None,
                 prev_ids=(), repetition_penalty=1.0) -> int
def beam_search(step_fn, bos, eos, beam_width, max_len, length_alpha=0.0)
    -> (sequence, normalized_score)
def greedy_decode(step_fn, bos, eos, max_len) -> sequence
```

---

## Behavior Notes / Gotchas

- **Top-p boundary semantics.** "Smallest set with mass ≥ p" means the
  crossing token is *in*. Off-by-one here silently changes the distribution;
  `searchsorted` on the cumulative sum gets it right. Never return an empty
  set — at minimum the top-1 token survives (`p ≈ 0` must still work).
- **Repetition penalty sign bug.** Naively dividing all logits by the penalty
  *increases* the probability of tokens with negative logits. The CTRL rule
  branches on sign so the token always gets less likely. This exact bug
  shipped in real libraries.
- **Beam search without length normalization prefers short outputs** — every
  extra token adds a negative log-prob, so "EOS now" beats any longer
  continuation with slightly lower per-token confidence. The test constructs
  a model where `alpha` flips the winner.
- **Greedy is not optimal.** The trap-model test has a first token with the
  highest immediate probability leading to a low-probability dead end; beam
  width 2 escapes it, and beam width 1 must reduce exactly to greedy.
- **Filter order matters.** Applying top-k after top-p (or temperature after
  filtering) gives different distributions. Match a convention and document
  it — this implementation mirrors HuggingFace's `LogitsProcessor` order.

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_decoding.py -v
```

| Test | Validates |
|------|-----------|
| `test_temperature_preserves_argmax_and_flattens` | Cold sharpens, hot flattens, argmax fixed |
| `test_top_k_keeps_exactly_k` | Exactly k survivors; `k ≥ V` no-op |
| `test_top_p_keeps_smallest_covering_set` | Nucleus = smallest covering set |
| `test_top_p_includes_boundary_token_and_top1` | Crossing token in; never empty |
| `test_sampling_matches_softmax_distribution` | Sampler is unbiased |
| `test_repetition_penalty_reduces_prob_for_any_sign` | The sign bug |
| `test_beam_search_escapes_greedy_trap` | Beam > greedy on trap model |
| `test_beam_width_one_is_greedy` | Degenerate case |
| `test_length_normalization_prefers_longer_completion` | GNMT alpha flips ranking |

---

## Discussion Questions (interview follow-ups)

- **Top-p vs top-k** — why does a fixed k misbehave on both very peaked and
  very flat distributions, and how does nucleus sampling adapt?
- **Min-p sampling** — the newer alternative (keep tokens with
  `p_tok ≥ min_p · p_max`). What failure mode of top-p does it fix?
- **Batched beam search** — how do you vectorize this across a batch on GPU,
  and what bookkeeping does a finished beam need (why do frameworks keep it
  in the batch with a forced pad token)?
- **Sampling temperature vs RL** — why does RLHF training typically sample at
  temperature 1.0 while evaluation uses lower temperatures?
- **Beam search degradation** — why does *increasing* beam width sometimes
  hurt open-ended generation quality (the "beam search curse")?
