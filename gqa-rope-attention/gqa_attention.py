import math

import torch
import torch.nn.functional as F
from torch import nn

# Shape suffix convention:
#   B = batch, L = new (query) length, T = total length (cache + new),
#   H = query heads, G = kv heads, K = head dim, D = model dim (H * K)
# A tensor's name tells you its shape. If an op's output name doesn't
# follow from its input names, the op is probably wrong.


def rope_tables(head_dim, max_len, base=10000.0):
    """Precompute cos/sin tables for RoPE. Returns (cos_TJ, sin_TJ), J = K/2."""
    inv_freq_J = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    pos_T = torch.arange(max_len).float()
    ang_TJ = torch.outer(pos_T, inv_freq_J)
    return ang_TJ.cos(), ang_TJ.sin()


def apply_rope(x_BhLK, cos_TJ, sin_TJ, offset=0):
    """Rotate consecutive (even, odd) feature pairs by a position-dependent angle.

    `offset` is the absolute position of the first token in x — required for
    incremental decoding, where the new token is NOT at position 0.
    """
    L = x_BhLK.size(2)
    cos_LJ = cos_TJ[offset:offset + L]
    sin_LJ = sin_TJ[offset:offset + L]
    x1_BhLJ, x2_BhLJ = x_BhLK[..., 0::2], x_BhLK[..., 1::2]
    out_BhLK = torch.empty_like(x_BhLK)
    out_BhLK[..., 0::2] = x1_BhLJ * cos_LJ - x2_BhLJ * sin_LJ
    out_BhLK[..., 1::2] = x1_BhLJ * sin_LJ + x2_BhLJ * cos_LJ
    return out_BhLK


class GQAttention(nn.Module):
    """Causal grouped-query attention with RoPE and an incremental KV cache.

    n_kv_heads == n_heads  -> plain multi-head attention
    n_kv_heads == 1        -> multi-query attention
    """

    def __init__(self, dim, n_heads, n_kv_heads, max_len=2048):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        cos_TJ, sin_TJ = rope_tables(self.head_dim, max_len)
        self.register_buffer("cos_TJ", cos_TJ, persistent=False)
        self.register_buffer("sin_TJ", sin_TJ, persistent=False)

    def forward(self, x_BLD, cache=None):
        """cache is None or (k_BGTK, v_BGTK) from previous steps.

        Returns (out_BLD, new_cache). Positions of x are offset by the
        cache length, so RoPE and the causal mask stay consistent between
        full-sequence and incremental calls.
        """
        B, L, _ = x_BLD.shape
        offset = 0 if cache is None else cache[0].size(2)

        q_BHLK = self.wq(x_BLD).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k_BGLK = self.wk(x_BLD).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_BGLK = self.wv(x_BLD).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE is applied BEFORE caching, so cached keys never need re-rotation.
        q_BHLK = apply_rope(q_BHLK, self.cos_TJ, self.sin_TJ, offset)
        k_BGLK = apply_rope(k_BGLK, self.cos_TJ, self.sin_TJ, offset)

        if cache is not None:
            k_BGTK = torch.cat([cache[0], k_BGLK], dim=2)
            v_BGTK = torch.cat([cache[1], v_BGLK], dim=2)
        else:
            k_BGTK, v_BGTK = k_BGLK, v_BGLK
        new_cache = (k_BGTK, v_BGTK)
        T = k_BGTK.size(2)

        # Each group of H/G query heads shares one kv head.
        rep = self.n_heads // self.n_kv_heads
        k_BHTK = k_BGTK.repeat_interleave(rep, dim=1)
        v_BHTK = v_BGTK.repeat_interleave(rep, dim=1)

        att_BHLT = q_BHLK @ k_BHTK.transpose(-1, -2) / math.sqrt(self.head_dim)

        # Query at absolute position offset+i may attend to keys 0..offset+i.
        qpos_L1 = torch.arange(offset, offset + L, device=x_BLD.device)[:, None]
        kpos_1T = torch.arange(T, device=x_BLD.device)[None, :]
        att_BHLT = att_BHLT.masked_fill(kpos_1T > qpos_L1, float("-inf"))

        out_BHLK = F.softmax(att_BHLT, dim=-1) @ v_BHTK
        out_BLD = out_BHLK.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out_BLD), new_cache
