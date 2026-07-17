import numpy as np
import pytest
import torch

from dqn import QNet, ReplayBuffer, select_action, td_loss
from env import (LEFT, MAX_STEPS, N_BRICKS, PADDLE_W, RIGHT, STAY, W,
                 MiniBreakout)
from train import evaluate, train


# ---------- environment invariants ----------

def test_reset_and_obs_shape():
    env = MiniBreakout(seed=0)
    obs = env.reset()
    assert obs.shape == (6,)
    assert env.bricks_left == N_BRICKS
    assert (obs >= 0).all() and (obs <= 1).all()      # normalized


def test_paddle_stays_in_bounds():
    env = MiniBreakout(seed=0)
    env.reset()
    for _ in range(50):                                # slam left
        env.step(LEFT)
        if env.done:
            env.reset()
    assert env.px == 0
    env.reset()
    for _ in range(50):
        env.step(RIGHT)
        if env.done:
            env.reset()
    assert env.px == W - PADDLE_W


def test_catch_bounces_ball_up():
    env = MiniBreakout(seed=0)
    env.reset()
    env.bx, env.by, env.vx, env.vy = 2, env_H() - 2, 1, 1   # descending onto paddle
    env.px = 2                                              # paddle under it
    env.step(STAY)
    assert env.vy == -1                                     # bounced up, alive
    assert not env.done


def test_miss_ends_episode():
    env = MiniBreakout(seed=0)
    env.reset()
    H = env_H()
    env.bx, env.by, env.vx, env.vy = 0, H - 2, 1, 1
    env.px = W - PADDLE_W                                   # paddle far away
    _, r, done, info = env.step(STAY)
    assert done and not info["won"]
    assert r < 0                                            # miss penalty


def test_top_contact_chips_a_brick():
    env = MiniBreakout(seed=0)
    env.reset()
    env.bx, env.by, env.vx, env.vy = 2, 1, 1, -1           # ascending into top
    before = env.bricks_left
    _, r, _, info = env.step(STAY)
    assert info["bricks_left"] == before - 1
    assert r > 0                                           # brick reward
    assert env.vy == 1                                     # bounced back down


def test_game_is_completable():
    # A predictive tracker must be able to clear the wall — otherwise the
    # task is unwinnable and no agent could speedrun it.
    env = MiniBreakout(seed=1)
    wins = 0
    for _ in range(50):
        env.reset()
        done, info = False, {"won": False}
        while not done:
            center = env.px + PADDLE_W / 2 - 0.5
            a = RIGHT if center < env.bx else (LEFT if center > env.bx else STAY)
            _, _, done, info = env.step(a)
        wins += info["won"]
    assert wins == 50                                      # tracker always wins


def env_H():
    from env import H
    return H


# ---------- DQN component tests ----------

def test_terminal_target_is_not_bootstrapped():
    # y = r for done transitions; y = r + gamma*max Q' otherwise.
    q, tq = QNet(), QNet()
    s = torch.zeros(1, 6)
    s2 = torch.zeros(1, 6)
    a = torch.tensor([0])
    r = torch.tensor([2.0])
    # With done=1, loss pins Q(s,a) to r regardless of the target net.
    loss_done = td_loss(q, tq, s, a, r, s2, torch.tensor([1.0]), gamma=0.99)
    q_sa = q(s)[0, 0]
    expected = torch.nn.functional.smooth_l1_loss(q_sa, r.squeeze())
    assert torch.allclose(loss_done, expected, atol=1e-6)


def test_replay_buffer_sampling_shapes():
    buf = ReplayBuffer(capacity=100)
    for i in range(50):
        buf.push(np.zeros(6, np.float32), 1, 0.5, np.ones(6, np.float32), False)
    s, a, r, s2, d = buf.sample(16)
    assert s.shape == (16, 6) and a.shape == (16,) and s2.shape == (16, 6)


def test_epsilon_greedy_is_greedy_at_zero():
    q = QNet()
    obs = np.random.default_rng(0).normal(size=6).astype(np.float32)
    rng = np.random.default_rng(0)
    greedy = int(q(torch.tensor(obs).unsqueeze(0)).argmax())
    assert all(select_action(q, obs, 0.0, rng) == greedy for _ in range(10))


def test_td_update_reduces_loss():
    torch.manual_seed(0)
    q, tq = QNet(), QNet()
    opt = torch.optim.Adam(q.parameters(), lr=1e-2)
    buf = ReplayBuffer()
    env = MiniBreakout(seed=0)
    obs = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(600):
        a = select_action(q, obs, 0.3, rng)
        obs2, r, done, _ = env.step(a)
        buf.push(obs, a, r, obs2, done)
        obs = env.reset() if done else obs2
    batch = buf.sample(128)
    first = td_loss(q, tq, *batch).item()
    for _ in range(50):
        loss = td_loss(q, tq, *batch)
        opt.zero_grad(); loss.backward(); opt.step()
    assert td_loss(q, tq, *batch).item() < first


# ---------- the deliverable: DQN learns to speedrun ----------

@pytest.fixture(scope="module")
def trained():
    q, history = train(steps=8000, eval_every=8000, seed=0, log=False)
    return q, history


def test_random_baseline_rarely_completes():
    env = MiniBreakout(seed=7)
    rng = np.random.default_rng(7)
    wins = 0
    for _ in range(1000):
        env.reset()
        done, info = False, {"won": False}
        while not done:
            _, _, done, info = env.step(int(rng.integers(0, 3)))
        wins += info["won"]
    assert wins / 1000 < 0.15                             # random ~4%


def test_dqn_completes_reliably(trained):
    q, _ = trained
    completion, _, _ = evaluate(q, n_episodes=1000)
    assert completion >= 0.9                              # vs ~4% random


def test_dqn_speedruns_near_optimal(trained):
    q, _ = trained
    _, mean_time, win_time = evaluate(q, n_episodes=1000)
    assert win_time <= 46                                 # ~43 is the limit
    assert mean_time <= 55                                # vs ~98 for random


def test_completion_time_drops_over_training(trained):
    _, history = trained
    # Untrained -> trained: time-to-complete must fall substantially.
    assert history[-1]["mean_time"] <= history[0]["mean_time"] - 20
