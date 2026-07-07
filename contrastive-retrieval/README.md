# Bi-Encoder Dense Retrieval (Contrastive Training)

**Problem:** Implement and train a **bi-encoder** for dense passage retrieval. Given
(query, positive-document) pairs, learn embeddings such that a query lands close to
its matching document and far from every other document in the batch. Training uses
the **in-batch InfoNCE** contrastive objective.

This is the core of modern semantic search / RAG retrievers (DPR, E5, GTE, BGE, ...).

---

## Core Requirements

1. **Encoder (`BiEncoder`)**
   - Wrap a transformer `backbone` (e.g. a HuggingFace encoder) shared between
     queries and documents (a *Siamese* / tied-weights bi-encoder).
   - **Masked mean pooling** over the token dimension using `attention_mask`
     (padding tokens must not contribute).
   - **L2-normalize** the pooled vector so that dot product == cosine similarity.
   - A **learnable temperature**, stored in log-space as a parameter so it stays
     positive under gradient descent.

2. **Loss (`info_nce`)**
   - Build the `B × B` similarity matrix between queries and documents.
   - Scale by the temperature.
   - The diagonal entries are the positives; every off-diagonal document is an
     in-batch negative.
   - **False-negative masking:** if two rows in the batch share the same document
     (same `doc_id`), that off-diagonal pair is not a true negative — mask it to
     `-inf` before the softmax so it doesn't push the model in the wrong direction.
   - Return the mean cross-entropy of the softmax over each query's row.

3. **Data plumbing**
   - `PairDataset` — holds parallel `(queries, docs)` lists.
   - `make_collate(tok)` — tokenizes each side with padding/truncation and emits a
     `doc_id_B` tensor (hash of each doc string) used for false-negative masking.

4. **Training loop (`train`)**
   - AdamW, weight decay, **gradient accumulation** (`accum` micro-batches per step),
     and a step budget.

---

## Interface

```python
class BiEncoder(nn.Module):
    def __init__(self, backbone, temp=0.05): ...
    def forward(self, input_ids, attention_mask) -> Tensor:  # (B, D), L2-normalized
        ...

def info_nce(q_BD, d_BD, doc_id_B, log_temp) -> Tensor:  # scalar loss
    ...
```

Shape-suffix naming convention (a tensor's name encodes its shape):

```
B = batch      L = sequence length      D = model dim
```

---

## Behavior Notes / Gotchas

- **Pooling must respect the mask.** `sum(h * mask) / sum(mask)`, with the
  denominator clamped to `>= 1` to avoid divide-by-zero on an all-padding row.
- **Temperature in log-space.** Store `log_temp = log(temp)` as the parameter and use
  `log_temp.exp()` at call time; this keeps the temperature strictly positive.
  With cosine similarities in `[-1, 1]`, a small temp (~0.05) sharpens the softmax.
- **Why L2-normalize?** After normalization `q · d` is exactly cosine similarity, so
  the InfoNCE logits live on a bounded scale that the temperature controls.
- **False-negative masking is the subtle part.** A naive in-batch loss assumes every
  off-diagonal doc is a negative. If the same passage appears twice in a batch, one of
  those "negatives" is actually a positive — masking it to `-inf` prevents a wrong
  gradient. Note the diagonal itself is explicitly *excluded* from the mask
  (`& ~eye`) so the true positive is always kept.
- **In-batch negatives ⇒ bigger batch = harder task = better retriever.** Gradient
  accumulation increases the effective *optimizer* batch but not the number of
  in-batch negatives per loss call.

---

## Running the Smoke Test

The test uses a tiny random-embedding stub backbone, so it runs on CPU with no
model download and no `transformers` dependency:

```bash
pip install torch
python -m pytest test_bi_encoder.py -v
```

The tests check:

| Test | Validates |
|------|-----------|
| `test_pooling_ignores_padding` | Masked mean pooling excludes padding tokens |
| `test_output_is_normalized`    | Encoder output is unit-norm (cosine-ready) |
| `test_perfect_match_low_loss`  | Aligned q/d embeddings ⇒ near-zero InfoNCE loss |
| `test_misaligned_high_loss`    | Shuffled positives ⇒ larger loss |
| `test_false_negative_masking`  | Duplicate docs are masked, not penalized |
| `test_train_step_runs`         | End-to-end train loop decreases loss |

---

## Discussion Questions (interview follow-ups)

- **Bi-encoder vs. cross-encoder** — why does the bi-encoder scale to millions of docs
  (precompute + ANN index) while the cross-encoder is only used for reranking?
- **Hard negatives** — how would you mine and add hard negatives (BM25 / ANN-mined)
  on top of in-batch negatives?
- **Symmetric loss** — some setups also add the document→query direction. When helps?
- **Distributed negatives** — how do you gather embeddings across GPUs to grow the
  effective negative pool (`all_gather` with gradients)?
