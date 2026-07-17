import copy

import numpy as np
import pytest
import torch

from rl import (answer_logprobs, clipped_pg_loss, grpo_advantages, k3_kl,
                sample_rollouts, train_rl)
from sft import train_sft
from task import (ANSWER_LEN, EOS, PROMPT_LEN, encode_answer, encode_prompt,
                  evaluate, make_pairs, reward_fn)


# ---------- unit tests: task plumbing ----------

def test_encoding_hand_case():
    assert encode_prompt(7, 25) == [0, 7, 10, 2, 5, 11]     # "07+25="
    assert encode_answer(7, 25) == [0, 3, 2, EOS]           # "032" EOS


def test_reward_is_exact_match():
    pairs = [(7, 25), (99, 99)]
    gold = torch.tensor([encode_answer(*p) for p in pairs])
    assert reward_fn(pairs, gold).tolist() == [1.0, 1.0]
    wrong = gold.clone()
    wrong[0, 2] = 9                                          # one digit off
    assert reward_fn(pairs, wrong).tolist() == [0.0, 1.0]
    no_eos = gold.clone()
    no_eos[1, 3] = 0                                         # missing EOS
    assert reward_fn(pairs, no_eos).tolist() == [1.0, 0.0]


def test_grpo_advantages_group_properties():
    r = torch.tensor([0.0, 1.0, 1.0, 0.0,   # group 1: mixed
                      1.0, 1.0, 1.0, 1.0])  # group 2: all correct
    adv = grpo_advantages(r, group_size=4)
    assert abs(float(adv[:4].mean())) < 1e-6
    assert float(adv[1]) > 0 > float(adv[0])
    # All-equal rewards (solved prompt): advantage 0, never NaN.
    assert torch.allclose(adv[4:], torch.zeros(4))


def test_train_eval_split_is_disjoint():
    rng = np.random.default_rng(0)
    train = make_pairs(500, rng)
    ev = make_pairs(100, rng, exclude=frozenset(train))
    assert not (set(train) & set(ev))


# ---------- shared expensive fixtures ----------

@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(42)
    train_pairs = make_pairs(2000, rng)
    eval_pairs = make_pairs(300, rng, exclude=frozenset(train_pairs))
    sft = train_sft(train_pairs, steps=250)
    return sft, train_pairs, eval_pairs


def test_sft_baseline_is_mediocre(setup):
    sft, _, eval_pairs = setup
    acc = evaluate(sft, eval_pairs)
    assert 0.10 < acc < 0.60        # headroom for RL, but real signal


def test_sampled_logprobs_match_recompute(setup):
    # logp_old recorded during sampling must equal answer_logprobs on the
    # same sequences (model unchanged) — catches off-by-one position bugs.
    sft, train_pairs, _ = setup
    gen = torch.Generator().manual_seed(0)
    full_ids, _, logp_old = sample_rollouts(sft, train_pairs[:8], generator=gen)
    with torch.no_grad():
        logp_re = answer_logprobs(sft, full_ids)
    assert torch.allclose(logp_old, logp_re, atol=1e-5)


def test_rollouts_have_reward_signal(setup):
    # At temperature 1 from a ~30%-accurate model, a batch of rollouts must
    # contain BOTH correct and incorrect answers — otherwise GRPO advantages
    # are all zero and nothing can learn.
    sft, train_pairs, _ = setup
    gen = torch.Generator().manual_seed(0)
    pairs = [p for p in train_pairs[:16] for _ in range(8)]
    _, gen_ids, _ = sample_rollouts(sft, pairs, generator=gen)
    r = reward_fn(pairs, gen_ids)
    assert 0.0 < float(r.mean()) < 1.0


# ---------- the deliverable: RL improves held-out eval ----------

def test_grpo_improves_heldout_eval(setup):
    sft, train_pairs, eval_pairs = setup
    base = evaluate(sft, eval_pairs)
    _, hist = train_rl(copy.deepcopy(sft), train_pairs, eval_pairs,
                       algo="grpo", steps=200, eval_every=200, seed=0)
    final = hist[-1]["eval_acc"]
    assert final >= base + 0.12          # real, seed-robust improvement
    assert final >= 0.42
    assert hist[-1]["kl"] < 1.0          # policy stayed near the reference


def test_ppo_improves_heldout_eval(setup):
    sft, train_pairs, eval_pairs = setup
    base = evaluate(sft, eval_pairs)
    _, hist = train_rl(copy.deepcopy(sft), train_pairs, eval_pairs,
                       algo="ppo", steps=200, eval_every=200, seed=0)
    final = hist[-1]["eval_acc"]
    assert final >= base + 0.12
    assert hist[-1]["kl"] < 1.0


def test_history_records_the_curve(setup):
    sft, train_pairs, eval_pairs = setup
    _, hist = train_rl(copy.deepcopy(sft), train_pairs, eval_pairs,
                       algo="grpo", steps=50, eval_every=25, seed=0)
    assert [h["step"] for h in hist] == [0, 25, 50]
    assert all(0.0 <= h["eval_acc"] <= 1.0 for h in hist)
    assert not np.isnan(hist[-1]["mean_reward"])
