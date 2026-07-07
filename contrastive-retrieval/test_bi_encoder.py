"""Smoke tests for the bi-encoder retriever.

Uses a tiny random-embedding stub backbone so the suite runs on CPU with no
model download and no `transformers` dependency. Only `torch` is required.
"""
import types

import torch
from torch import nn

from bi_encoder import BiEncoder, info_nce, PairDataset


class StubBackbone(nn.Module):
    """Mimics a HF encoder: returns an object with `.last_hidden_state` (B, L, D)."""

    def __init__(self, vocab=50, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, input_ids, attention_mask):
        h_BLD = self.emb(input_ids)
        return types.SimpleNamespace(last_hidden_state=h_BLD)


def test_pooling_ignores_padding():
    """A trailing padding token (mask=0) must not change the pooled vector."""
    torch.manual_seed(0)
    model = BiEncoder(StubBackbone())
    ids = torch.tensor([[1, 2, 3, 0]])           # last token is padding
    mask_pad = torch.tensor([[1, 1, 1, 0]])
    mask_full3 = torch.tensor([[1, 1, 1]])
    out_pad = model(ids, mask_pad)
    out_ref = model(ids[:, :3], mask_full3)
    assert torch.allclose(out_pad, out_ref, atol=1e-6)


def test_output_is_normalized():
    torch.manual_seed(0)
    model = BiEncoder(StubBackbone())
    ids = torch.randint(0, 50, (4, 7))
    mask = torch.ones_like(ids)
    out_BD = model(ids, mask)
    norms = out_BD.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_perfect_match_low_loss():
    """If each query embedding equals its own doc embedding, loss is near zero."""
    torch.manual_seed(0)
    q_BD = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    d_BD = q_BD.clone()
    doc_id_B = torch.arange(8)
    log_temp = torch.tensor(0.05).log()
    loss = info_nce(q_BD, d_BD, doc_id_B, log_temp)
    assert loss.item() < 1e-3


def test_misaligned_high_loss():
    """Shuffling positives should produce a clearly larger loss than aligned."""
    torch.manual_seed(0)
    q_BD = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    doc_id_B = torch.arange(8)
    log_temp = torch.tensor(0.05).log()
    aligned = info_nce(q_BD, q_BD.clone(), doc_id_B, log_temp)
    shuffled = info_nce(q_BD, q_BD[torch.randperm(8)], doc_id_B, log_temp)
    assert shuffled.item() > aligned.item() + 1.0


def test_false_negative_masking():
    """Duplicate docs in a batch must be masked, not counted as negatives.

    Rows 0 and 1 carry the *same* document. Without masking, each acts as a hard
    negative for the other and inflates the loss. Masking should lower it.
    """
    torch.manual_seed(1)
    q_BD = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    d_BD = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    d_BD[1] = d_BD[0]                              # rows 0 and 1 share a doc
    log_temp = torch.tensor(0.05).log()

    masked = info_nce(q_BD, d_BD, torch.tensor([7, 7, 2, 3]), log_temp)
    unmasked = info_nce(q_BD, d_BD, torch.arange(4), log_temp)
    assert masked.item() < unmasked.item()


def test_train_step_runs():
    """End-to-end: a few optimizer steps on a trivial dataset reduce the loss."""
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    model = BiEncoder(StubBackbone())

    # Fixed toy batch: 6 (query, doc) pairs of token-id sequences.
    def fake_collate(batch):
        q = {"input_ids": torch.randint(0, 50, (6, 5)),
             "attention_mask": torch.ones(6, 5, dtype=torch.long)}
        d = {"input_ids": torch.randint(0, 50, (6, 5)),
             "attention_mask": torch.ones(6, 5, dtype=torch.long)}
        return q, d, torch.arange(6)

    ds = PairDataset(list(range(6)), list(range(6)))
    loader = DataLoader(ds, batch_size=6, collate_fn=fake_collate)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    q, d, doc_id_B = next(iter(loader))
    first = info_nce(model(**q), model(**d), doc_id_B, model.log_temp)
    for _ in range(20):
        opt.zero_grad()
        loss = info_nce(model(**q), model(**d), doc_id_B, model.log_temp)
        loss.backward()
        opt.step()
    assert loss.item() < first.item()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
