"""THE EXERCISE: this training script runs without errors, the loss goes
down, and the model is still garbage. Find and fix all the bugs.

Don't diff against the reference until you've made your list — in the real
interview there is no reference. See README.md for the task setup.

Shape suffix convention:
  B = batch, L = seq len, D = model dim, H = heads, K = head dim, V = vocab
"""

import torch
import torch.nn.functional as F
from torch import nn

VOCAB, SEP = 12, 11
PREFIX_LEN = 5
SEQ_LEN = 2 * PREFIX_LEN + 1


def attention(q_BHLK, k_BHLK, v_BHLK):
    scores_BHLL = q_BHLK @ k_BHLK.transpose(-1, -2)
    return F.softmax(scores_BHLL, dim=-2) @ v_BHLK


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
        att_BLD = attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        x_BLD = x_BLD + self.proj(att_BLD)
        x_BLD = self.mlp(self.ln2(x_BLD))
        return x_BLD


class TransformerLM(nn.Module):
    def __init__(self, dim=32, n_heads=2, n_layers=2, max_len=SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def forward(self, ids_BL):
        pos_L = torch.arange(ids_BL.size(1), device=ids_BL.device)
        x_BLD = self.tok_emb(ids_BL) + self.pos_emb(pos_L)
        for block in self.blocks:
            x_BLD = block(x_BLD)
        return self.head(self.ln_f(x_BLD))


def loss_fn(logits_BLV, ids_BL):
    return F.cross_entropy(logits_BLV.reshape(-1, VOCAB), ids_BL.reshape(-1))


def make_copy_batch(batch_size, rng):
    prefix = torch.randint(0, SEP, (batch_size, PREFIX_LEN), generator=rng)
    sep = torch.full((batch_size, 1), SEP)
    return torch.cat([prefix, sep, prefix], dim=1)


def train_copy_task(steps=300, batch_size=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = TransformerLM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        ids_BL = make_copy_batch(batch_size, rng)
        loss = loss_fn(model(ids_BL), ids_BL)
        loss.backward()
        opt.step()
    return model, loss.item()


@torch.no_grad()
def greedy_generate(model, prompt_BL, n_tokens):
    ids_BL = prompt_BL.clone()
    for _ in range(n_tokens):
        next_B = model(ids_BL)[:, -1].argmax(-1, keepdim=True)
        ids_BL = torch.cat([ids_BL, next_B], dim=1)
    return ids_BL
