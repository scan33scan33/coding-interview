"""The verifiable task: 2-digit addition as character-level sequences.

    prompt  "07+25="   (always 6 tokens, numbers zero-padded)
    answer  "032" EOS  (sum zero-padded to 3 digits, always 4 tokens)

Fixed widths mean every sequence is exactly 10 tokens and the answer always
starts at position 6 — no padding/masking machinery, which keeps the RL
plumbing readable. The reward is EXACT MATCH on all 4 answer tokens: a
programmatically verifiable reward, the setting GRPO was designed for.
"""

import numpy as np
import torch

DIGITS = 10
PLUS, EQ, EOS = 10, 11, 12
VOCAB = 13
PROMPT_LEN, ANSWER_LEN = 6, 4
SEQ_LEN = PROMPT_LEN + ANSWER_LEN


def encode_prompt(a, b):
    s = f"{a:02d}+{b:02d}="
    return [PLUS if c == "+" else EQ if c == "=" else int(c) for c in s]


def encode_answer(a, b):
    return [int(c) for c in f"{a + b:03d}"] + [EOS]


def make_pairs(n, rng, exclude=frozenset()):
    """n distinct (a, b) pairs, disjoint from `exclude` (train/eval split)."""
    pairs, seen = [], set(exclude)
    while len(pairs) < n:
        a, b = int(rng.integers(0, 100)), int(rng.integers(0, 100))
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    return pairs


def make_batch(pairs):
    """Full sequences (B, 10): prompt + gold answer."""
    return torch.tensor([encode_prompt(a, b) + encode_answer(a, b)
                         for a, b in pairs])


def prompts_tensor(pairs):
    return torch.tensor([encode_prompt(a, b) for a, b in pairs])


def reward_fn(pairs, gen_B4):
    """1.0 iff all 4 generated tokens match the gold answer (incl. EOS)."""
    gold_B4 = torch.tensor([encode_answer(a, b) for a, b in pairs])
    return (gen_B4 == gold_B4).all(dim=1).float()


@torch.no_grad()
def greedy_answers(model, pairs, batch_size=256):
    """Greedy-decode 4 answer tokens for each pair."""
    outs = []
    for i in range(0, len(pairs), batch_size):
        ids = prompts_tensor(pairs[i:i + batch_size])
        for _ in range(ANSWER_LEN):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], 1)
        outs.append(ids[:, PROMPT_LEN:])
    return torch.cat(outs)


@torch.no_grad()
def evaluate(model, pairs):
    """Held-out greedy exact-match accuracy — THE eval number."""
    return float(reward_fn(pairs, greedy_answers(model, pairs)).mean())
