"""Stage 1: supervised fine-tuning, DELIBERATELY undertrained.

The SFT model is stopped while accuracy is mediocre so the RL stage has
headroom to demonstrate a real improvement. (In the frontier-lab analogy:
this is the pretrained/SFT checkpoint before RLHF/RLVR.)
"""

import numpy as np
import torch
import torch.nn.functional as F

from model import TinyLM
from task import PROMPT_LEN, SEQ_LEN, VOCAB, make_batch


def sft_loss(logits_BLV, ids_BL):
    """Next-token CE on ANSWER tokens only (positions >= PROMPT_LEN).
    The prompt is given, not modeled — same masking idea as DPO/RLHF stacks."""
    logp = F.log_softmax(logits_BLV[:, :-1], dim=-1)
    tgt = ids_BL[:, 1:]
    picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(picked)
    mask[:, PROMPT_LEN - 1:] = 1.0        # targets at positions 6..9
    return -(picked * mask).sum() / mask.sum()


def train_sft(train_pairs, steps=250, batch_size=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyLM(VOCAB, SEQ_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        idx = rng.choice(len(train_pairs), size=batch_size)
        ids = make_batch([train_pairs[i] for i in idx])
        loss = sft_loss(model(ids), ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
