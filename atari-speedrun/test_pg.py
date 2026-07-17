import numpy as np
import pytest
import torch

from env import MiniBreakout
from pg import (ActorCritic, clipped_surrogate, evaluate_policy, gae,
                group_advantages, train_grpo, train_ppo)


# ---------- shared math ----------

def test_gae_equals_discounted_return_when_lambda_one_and_zero_values():
    # With values=0 and lam=1, GAE reduces to the discounted return-to-go.
    rewards = np.array([1.0, 2.0, 3.0], np.float32)
    values = np.zeros(4, np.float32)                 # includes bootstrap slot
    dones = np.array([0, 0, 1], np.float32)
    adv = gae(rewards, values, dones, gamma=0.5, lam=1.0)
    # r0 + .5 r1 + .25 r2 ; r1 + .5 r2 ; r2
    assert np.allclose(adv, [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3], atol=1e-5)


def test_gae_cuts_bootstrap_at_episode_boundary():
    # A done in the middle must stop advantage from leaking across episodes.
    rewards = np.array([1.0, 1.0, 1.0], np.float32)
    values = np.array([0.0, 0.0, 0.0, 5.0], np.float32)  # big future value
    with_done = gae(rewards, values, np.array([0, 1, 0], np.float32),
                    gamma=0.9, lam=1.0)
    # Step 1 is terminal: its advantage must ignore the value[2]/value[3].
    assert np.isclose(with_done[1], 1.0, atol=1e-5)


def test_group_advantages_standardized():
    adv = group_advantages([0.0, 1.0, 2.0, 3.0, 4.0])
    assert abs(adv.mean()) < 1e-5
    assert abs(adv.std() - 1.0) < 1e-2
    # All-equal returns (group collapsed): advantage ~ 0, never NaN.
    assert np.allclose(group_advantages([2.0, 2.0, 2.0]), 0.0, atol=1e-3)


def test_clipped_surrogate_pushes_toward_positive_advantage():
    old = torch.zeros(4)
    new = torch.zeros(4, requires_grad=True)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    clipped_surrogate(new, old, adv).backward()
    # Descent must raise logp where advantage > 0 and lower it where < 0.
    assert (new.grad[:2] < 0).all()
    assert (new.grad[2:] > 0).all()


# ---------- env support for GRPO grouping ----------

def test_snapshot_restore_replays_identical_trajectory():
    env = MiniBreakout(seed=0)
    env.reset()
    snap = env.snapshot()
    actions = [0, 1, 2, 1, 0, 2, 2, 1]

    def rollout():
        env.restore(snap)
        traj = []
        for a in actions:
            if env.done:
                break
            obs, r, done, _ = env.step(a)
            traj.append((tuple(np.round(obs, 5)), round(r, 5), done))
        return traj

    assert rollout() == rollout()          # deterministic given start + actions


# ---------- greedy eval ----------

def test_evaluate_policy_returns_valid_metrics():
    c, t, w = evaluate_policy(ActorCritic(), n_episodes=100)
    assert 0.0 <= c <= 1.0
    assert 0.0 <= t <= 100.0


# ---------- the deliverable: PPO and GRPO learn to speedrun ----------

@pytest.fixture(scope="module")
def ppo_run():
    return train_ppo(iters=60, eval_every=60, seed=0, log=False)


@pytest.fixture(scope="module")
def grpo_run():
    return train_grpo(iters=50, eval_every=50, seed=0, log=False)


def test_ppo_learns_to_speedrun(ppo_run):
    policy, _ = ppo_run
    c, t, w = evaluate_policy(policy, n_episodes=1000)
    assert c >= 0.9
    assert w <= 46                                   # ~43 optimal
    assert t <= 55                                   # vs ~98 random


def test_ppo_completion_time_drops(ppo_run):
    _, history = ppo_run
    assert history[-1]["mean_time"] <= history[0]["mean_time"] - 20


def test_grpo_learns_to_speedrun(grpo_run):
    policy, _ = grpo_run
    c, t, w = evaluate_policy(policy, n_episodes=1000)
    assert c >= 0.9
    assert w <= 46
    assert t <= 55


def test_grpo_completion_time_drops(grpo_run):
    _, history = grpo_run
    assert history[-1]["mean_time"] <= history[0]["mean_time"] - 20
