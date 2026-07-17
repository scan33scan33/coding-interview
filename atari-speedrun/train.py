"""Training + evaluation harness: train a DQN to speedrun MiniBreakout and
watch the completion time drop.

The headline eval is TIME-TO-COMPLETE: steps taken to clear the wall, with a
did-not-finish (miss or timeout) counted as the full step budget. A random
policy almost never finishes, so its expected completion time is ~the cap;
a trained policy speedruns the wall in near-optimal steps.
"""

import copy

import numpy as np
import torch

from dqn import QNet, ReplayBuffer, select_action, td_loss
from env import MAX_STEPS, MiniBreakout


@torch.no_grad()
def evaluate(q, n_episodes=500, seed=999):
    """Greedy rollouts. Returns (completion_rate, mean_time, mean_time_on_win).

    mean_time counts a DNF as MAX_STEPS — this is the "how fast do you
    complete the game" number (lower is better)."""
    env = MiniBreakout(seed=seed)
    rng = np.random.default_rng(seed)
    wins, times, win_times = 0, [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        info = {"won": False, "steps": 0}
        while not done:
            a = select_action(q, obs, epsilon=0.0, rng=rng)
            obs, _, done, info = env.step(a)
        if info["won"]:
            wins += 1
            win_times.append(info["steps"])
            times.append(info["steps"])
        else:
            times.append(MAX_STEPS)          # DNF penalty = full budget
    return (wins / n_episodes, float(np.mean(times)),
            float(np.mean(win_times)) if win_times else float("nan"))


def train(steps=40000, batch_size=128, lr=1e-3, gamma=0.99,
          eps_start=1.0, eps_end=0.05, eps_decay_steps=15000,
          target_sync=500, warmup=1000, eval_every=5000, seed=0, log=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = QNet()
    target_q = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buffer = ReplayBuffer()
    env = MiniBreakout(seed=seed)

    obs = env.reset()
    history = []
    c0, t0, w0 = evaluate(q)
    history.append({"step": 0, "completion": c0, "mean_time": t0, "win_time": w0})
    if log:
        print(f"step {0:>6}  completion {c0:5.1%}  mean_time {t0:5.1f}")

    for step in range(1, steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        a = select_action(q, obs, eps, rng)
        obs2, r, done, _ = env.step(a)
        buffer.push(obs, a, r, obs2, done)
        obs = env.reset() if done else obs2

        if len(buffer) >= warmup:
            batch = buffer.sample(batch_size)
            loss = td_loss(q, target_q, *batch, gamma=gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % target_sync == 0:
                target_q.load_state_dict(q.state_dict())

        if step % eval_every == 0:
            c, t, w = evaluate(q)
            history.append({"step": step, "completion": c,
                            "mean_time": t, "win_time": w})
            if log:
                print(f"step {step:>6}  completion {c:5.1%}  "
                      f"mean_time {t:5.1f}  speedrun {w:5.1f}  eps {eps:.2f}")
    return q, history
