import torch

from gqa_attention import GQAttention, apply_rope, rope_tables

torch.manual_seed(0)


def test_rope_preserves_norm():
    cos_TJ, sin_TJ = rope_tables(head_dim=8, max_len=32)
    x_BhLK = torch.randn(2, 3, 16, 8)
    y_BhLK = apply_rope(x_BhLK, cos_TJ, sin_TJ)
    assert torch.allclose(x_BhLK.norm(dim=-1), y_BhLK.norm(dim=-1), atol=1e-5)


def test_rope_is_relative():
    # <rope(q, i), rope(k, j)> must depend only on i - j: shifting both
    # positions by the same amount leaves the dot product unchanged.
    cos_TJ, sin_TJ = rope_tables(head_dim=16, max_len=64)
    q_11LK = torch.randn(1, 1, 1, 16)
    k_11LK = torch.randn(1, 1, 1, 16)
    dots = []
    for shift in [0, 5, 23]:
        qr = apply_rope(q_11LK, cos_TJ, sin_TJ, offset=3 + shift)
        kr = apply_rope(k_11LK, cos_TJ, sin_TJ, offset=7 + shift)
        dots.append((qr * kr).sum())
    assert torch.allclose(dots[0], dots[1], atol=1e-5)
    assert torch.allclose(dots[0], dots[2], atol=1e-5)


def test_gqa_equals_mha_when_groups_equal_heads():
    # With n_kv_heads == n_heads, GQA degenerates to standard MHA; copying
    # the weights across must give identical outputs.
    mha = GQAttention(dim=32, n_heads=4, n_kv_heads=4)
    gqa = GQAttention(dim=32, n_heads=4, n_kv_heads=4)
    gqa.load_state_dict(mha.state_dict())
    x_BLD = torch.randn(2, 10, 32)
    out_a, _ = mha(x_BLD)
    out_b, _ = gqa(x_BLD)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_causal_masking():
    # Perturbing a future token must not change any earlier output position.
    attn = GQAttention(dim=32, n_heads=4, n_kv_heads=2)
    x_BLD = torch.randn(1, 8, 32)
    out_a, _ = attn(x_BLD)
    x2_BLD = x_BLD.clone()
    x2_BLD[0, 5] += 10.0
    out_b, _ = attn(x2_BLD)
    assert torch.allclose(out_a[0, :5], out_b[0, :5], atol=1e-5)
    assert not torch.allclose(out_a[0, 5:], out_b[0, 5:], atol=1e-5)


def test_kv_cache_matches_full_forward():
    # Decoding one token at a time through the cache must reproduce the
    # full-sequence forward exactly (RoPE offsets + mask offsets correct).
    attn = GQAttention(dim=48, n_heads=6, n_kv_heads=2)
    x_BLD = torch.randn(2, 12, 48)
    full_BLD, _ = attn(x_BLD)

    cache = None
    steps = []
    for t in range(x_BLD.size(1)):
        out_B1D, cache = attn(x_BLD[:, t:t + 1], cache)
        steps.append(out_B1D)
    inc_BLD = torch.cat(steps, dim=1)
    assert torch.allclose(full_BLD, inc_BLD, atol=1e-5)


def test_cache_shapes_use_kv_heads():
    # The whole point of GQA: the cache stores G kv heads, not H query heads.
    attn = GQAttention(dim=64, n_heads=8, n_kv_heads=2)
    _, (k, v) = attn(torch.randn(1, 5, 64))
    assert k.shape == (1, 2, 5, 8)
    assert v.shape == (1, 2, 5, 8)
