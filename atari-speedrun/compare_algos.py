"""Head-to-head: DQN vs PPO vs GRPO on MiniBreakout, same completion-time
metric. Trains all three and prints a comparison table.

    python compare_algos.py      # ~2-3 minutes on CPU
"""

import time

import numpy as np

from dqn import select_action
from env import MAX_STEPS, MiniBreakout
from pg import evaluate_policy, train_grpo, train_ppo
from train import evaluate as evaluate_dqn
from train import train as train_dqn


def random_baseline(n=2000):
    env = MiniBreakout(seed=7)
    rng = np.random.default_rng(7)
    wins, times = 0, []
    for _ in range(n):
        env.reset()
        done, info = False, {"won": False, "steps": 0}
        while not done:
            _, _, done, info = env.step(int(rng.integers(0, 3)))
        wins += info["won"]
        times.append(info["steps"] if info["won"] else MAX_STEPS)
    return wins / n, float(np.mean(times))


def main():
    rows = []

    wr, mt = random_baseline()
    rows.append(("random policy", wr, mt, float("nan"), 0.0))

    print("Training DQN  (value-based, off-policy) ...")
    t0 = time.time()
    q, _ = train_dqn(steps=12000, eval_every=12000, seed=0, log=False)
    c, t, w = evaluate_dqn(q, n_episodes=1000)
    rows.append(("DQN", c, t, w, time.time() - t0))

    print("Training PPO  (actor-critic, GAE) ...")
    t0 = time.time()
    p, _ = train_ppo(iters=60, eval_every=60, seed=0, log=False)
    c, t, w = evaluate_policy(p, n_episodes=1000)
    rows.append(("PPO", c, t, w, time.time() - t0))

    print("Training GRPO (critic-free, group-relative) ...")
    t0 = time.time()
    g, _ = train_grpo(iters=50, eval_every=50, seed=0, log=False)
    c, t, w = evaluate_policy(g, n_episodes=1000)
    rows.append(("GRPO", c, t, w, time.time() - t0))

    print("\n=== MiniBreakout: how fast does each agent complete the game? ===")
    print(f"{'algorithm':<16}{'completion':>12}{'time-to-finish':>16}"
          f"{'speedrun':>10}{'train (s)':>11}")
    print("-" * 65)
    for name, c, t, w, secs in rows:
        wr_s = f"{w:.0f}" if not np.isnan(w) else "  --"
        secs_s = f"{secs:.0f}" if secs else "  --"
        print(f"{name:<16}{c:>11.1%}{t:>16.1f}{wr_s:>10}{secs_s:>11}")
    print("\n(time-to-finish counts a did-not-finish as the "
          f"{MAX_STEPS}-step budget; ~43 is the physical speed limit.)")


if __name__ == "__main__":
    main()
