"""End-to-end: measure a random baseline, train a DQN to speedrun
MiniBreakout, print the learning curve, and render a greedy rollout.

    python run_experiment.py      # ~1 minute on CPU
"""

import numpy as np
import torch

from dqn import select_action
from env import MAX_STEPS, MiniBreakout
from train import evaluate, train


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


def bar(frac, width=30):
    return "#" * int(frac * width)


def show_rollout(q, seed=123):
    env = MiniBreakout(seed=seed)
    obs = env.reset()
    rng = np.random.default_rng(0)
    frames = [env.render()]
    done, info = False, {"won": False}
    while not done:
        obs, _, done, info = env.step(select_action(q, obs, 0.0, rng))
        frames.append(env.render())
    print(f"Greedy rollout: {'WON' if info['won'] else 'DNF'} "
          f"in {info['steps']} steps. First and last frame:\n")
    print(frames[0])
    print("   ... (ball bounces, wall gets chipped away) ...")
    print(frames[-1])


def main():
    print("=== Baseline: uniform-random policy ===")
    wr, mt = random_baseline()
    print(f"completion rate {wr:.1%}   mean time-to-complete {mt:.1f} "
          f"(DNF counts as {MAX_STEPS})\n")

    print("=== Training DQN (watch it learn to speedrun) ===")
    print(f"{'step':>6}  {'completion':>10}  time-to-complete   speedrun")
    q, history = train(steps=15000, eval_every=1500, seed=0, log=False)
    for h in history:
        w = "  -- " if np.isnan(h["win_time"]) else f"{h['win_time']:4.0f}"
        print(f"{h['step']:>6}  {h['completion']:>9.1%}  "
              f"{h['mean_time']:5.1f} |{bar(1 - h['mean_time'] / MAX_STEPS):<30}| {w}")

    final = history[-1]
    print("\n=== Summary ===")
    print(f"  random policy : {wr:5.1%} completion, {mt:.0f} steps to finish")
    print(f"  trained DQN   : {final['completion']:5.1%} completion, "
          f"{final['win_time']:.0f} steps to finish (optimal speedrun)")
    print(f"  the wall needs 6 bricks x ~1 round-trip each; ~43 steps is the "
          f"physical speed limit.\n")

    show_rollout(q)


if __name__ == "__main__":
    main()
