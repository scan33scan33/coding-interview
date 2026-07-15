"""PyTorch port of the JAX transformer classifier (see transformer.py).

Same architecture: post-norm encoder blocks (MHA -> add+LN -> FFN -> add+LN),
first-token pooling, linear classifier head. Written as an importable module
with a synthetic-data smoke path, so tests don't need the MNIST download the
JAX script performs.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

D_MODEL = 112
NUM_HEADS = 4
NUM_CLASSES = 10


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x_BLD):
        B, L, D = x_BLD.shape
        q = self.q(x_BLD).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x_BLD).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x_BLD).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        # Scale by sqrt(d_head) — the per-head dim (FIX #2 in the JAX file).
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d_head), -1)
        return self.out((att @ v).transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """Post-norm, matching the JAX script: LN(x + sublayer(x))."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x_BLD):
        x_BLD = self.ln1(x_BLD + self.mha(x_BLD))
        return self.ln2(x_BLD + self.ffn(x_BLD))


class TransformerClassifier(nn.Module):
    """Flat input of size L*D_MODEL is reshaped into L tokens of D_MODEL,
    exactly like the JAX version treats flattened MNIST pixels."""

    def __init__(self, d_model=D_MODEL, n_heads=NUM_HEADS, n_layers=3,
                 n_classes=NUM_CLASSES):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(Block(d_model, n_heads)
                                    for _ in range(n_layers))
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x_BF):
        B, F_ = x_BF.shape
        x_BLD = x_BF.view(B, F_ // self.d_model, self.d_model)
        for block in self.blocks:
            x_BLD = block(x_BLD)
        return self.head(x_BLD[:, 0])              # first-token pooling


def train_epochs(model, x_BF, y_B, epochs=30, lr=1e-3, batch_size=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(x_BF))
        for i in range(0, len(x_BF), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(x_BF[idx]), y_B[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def accuracy(model, x_BF, y_B):
    return (model(x_BF).argmax(-1) == y_B).float().mean().item()
