"""Tiny decoder-only LM with an optional value head — the policy (and PPO
critic) for the RL post-training problem.

Shape suffix convention: B batch, L seq len, D model dim, V vocab.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x_BLD):
        B, L, D = x_BLD.shape
        qkv = self.qkv(self.ln1(x_BLD)).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        att_BHLL = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                       device=x_BLD.device), diagonal=1)
        att_BHLL = att_BHLL.masked_fill(causal, float("-inf"))
        out = (F.softmax(att_BHLL, dim=-1) @ v).transpose(1, 2).reshape(B, L, D)
        x_BLD = x_BLD + self.proj(out)
        return x_BLD + self.mlp(self.ln2(x_BLD))


class TinyLM(nn.Module):
    def __init__(self, vocab, max_len, dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.value_head = nn.Linear(dim, 1)   # PPO critic: V(state) per position

    def forward(self, ids_BL, return_value=False):
        pos = torch.arange(ids_BL.size(1), device=ids_BL.device)
        x_BLD = self.tok_emb(ids_BL) + self.pos_emb(pos)
        for block in self.blocks:
            x_BLD = block(x_BLD)
        h_BLD = self.ln_f(x_BLD)
        logits_BLV = self.head(h_BLD)
        if return_value:
            return logits_BLV, self.value_head(h_BLD).squeeze(-1)
        return logits_BLV
