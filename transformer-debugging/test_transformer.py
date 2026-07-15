"""Grader for the debugging exercise. Point these tests at YOUR fixed copy
of buggy_transformer.py (or at fixed_transformer.py to see them pass).
Each test is named for the bug class it catches.
"""

import math

import torch

import buggy_transformer as buggy
import fixed_transformer as fixed

torch.manual_seed(0)


# ---------- properties the fixed implementation must satisfy ----------

def test_causal_no_future_leak():
    # Catches: missing causal mask (and softmax over the wrong axis).
    model = fixed.TransformerLM()
    ids = torch.randint(0, fixed.VOCAB, (1, fixed.SEQ_LEN))
    out_a = model(ids)
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % fixed.VOCAB   # perturb only the last token
    out_b = model(ids2)
    assert torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-5)


def test_attention_rows_sum_to_one_over_keys():
    # Catches: softmax over the query axis instead of the key axis. With
    # v = one-hot basis, each output row IS that query's attention row.
    L, K = 6, 6
    q = torch.randn(1, 1, L, K)
    k = torch.randn(1, 1, L, K)
    v = torch.eye(L).view(1, 1, L, L)
    rows = fixed.attention(q, k, v)[0, 0]
    assert torch.allclose(rows.sum(-1), torch.ones(L), atol=1e-5)


def test_attention_scores_are_scaled():
    # Catches: missing 1/sqrt(head_dim). With q = k = sqrt(c)*e_i orthogonal
    # rows, the diagonal score is c*K before scaling, c*sqrt(K) after; the
    # resulting softmax weight on "self" differs measurably.
    K = 64
    q = k = torch.eye(K).view(1, 1, K, K) * 4.0
    v = torch.eye(K).view(1, 1, K, K)
    w = fixed.attention(q, k, v)[0, 0]
    # causal row i: self-score 16/sqrt(64)=2 vs 0 for others -> weight
    # exp(2)/(exp(2)+i). Unscaled would be exp(16)/(exp(16)+i) ~ 1.0.
    i = K - 1
    want = math.exp(2) / (math.exp(2) + i)
    assert abs(w[i, i].item() - want) < 1e-4


def test_loss_uses_shifted_labels():
    # Catches: predicting the CURRENT token instead of the next one.
    ids = torch.randint(0, fixed.VOCAB, (2, fixed.SEQ_LEN))
    perfect_next = torch.full((2, fixed.SEQ_LEN, fixed.VOCAB), -30.0)
    for b in range(2):
        for t in range(fixed.SEQ_LEN - 1):
            perfect_next[b, t, ids[b, t + 1]] = 30.0
    assert fixed.loss_fn(perfect_next, ids).item() < 1e-3

    perfect_current = torch.full((2, fixed.SEQ_LEN, fixed.VOCAB), -30.0)
    for b in range(2):
        for t in range(fixed.SEQ_LEN):
            perfect_current[b, t, ids[b, t]] = 30.0
    assert fixed.loss_fn(perfect_current, ids).item() > 5.0


def test_blocks_have_residual_path():
    # Catches: dropped residual around the MLP. With residuals, zeroing every
    # block's OUTPUT projections makes each block the identity map.
    model = fixed.TransformerLM()
    with torch.no_grad():
        for blk in model.blocks:
            blk.proj.weight.zero_(); blk.proj.bias.zero_()
            blk.mlp[-1].weight.zero_(); blk.mlp[-1].bias.zero_()
    ids = torch.randint(0, fixed.VOCAB, (1, fixed.SEQ_LEN))
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(fixed.SEQ_LEN))
    y = x
    for blk in model.blocks:
        y = blk(y)
    assert torch.allclose(y, x, atol=1e-6)


def test_training_overfits_and_generation_copies():
    # The end-to-end bar (also catches the missing zero_grad: accumulated
    # gradients keep training from converging this far this fast).
    # The loss can NOT go to zero: the first 4 predicted positions are
    # unpredictable random prefix tokens, an irreducible 4*ln(11)/10 ~ 0.96.
    # Learned = at the entropy floor; broken = well above it (~2.4).
    model, final_loss = fixed.train_copy_task(steps=300)
    assert final_loss < 1.05

    rng = torch.Generator().manual_seed(123)
    ids = fixed.make_copy_batch(8, rng)
    prompt = ids[:, :fixed.PREFIX_LEN + 1]           # prefix + SEP
    out = fixed.greedy_generate(model, prompt, fixed.PREFIX_LEN)
    copied = out[:, fixed.PREFIX_LEN + 1:]
    assert (copied == ids[:, :fixed.PREFIX_LEN]).float().mean() > 0.95


# ---------- demonstrate the buggy version really is broken ----------

def test_buggy_version_leaks_future_information():
    model = buggy.TransformerLM()
    ids = torch.randint(0, buggy.VOCAB, (1, buggy.SEQ_LEN))
    out_a = model(ids)
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % buggy.VOCAB
    out_b = model(ids2)
    assert not torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-5)


def test_buggy_loss_rewards_copying_the_input():
    # The smoking gun of unshifted labels: logits that just echo the current
    # token get (near-)zero loss, so training "succeeds" at nothing.
    ids = torch.randint(0, buggy.VOCAB, (2, buggy.SEQ_LEN))
    echo = torch.full((2, buggy.SEQ_LEN, buggy.VOCAB), -30.0)
    for b in range(2):
        for t in range(buggy.SEQ_LEN):
            echo[b, t, ids[b, t]] = 30.0
    assert buggy.loss_fn(echo, ids).item() < 1e-3
