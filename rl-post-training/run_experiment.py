"""End-to-end experiment: SFT -> {GRPO, PPO} -> eval comparison.

Runs in ~1 minute on CPU. Prints the held-out eval curve for both
algorithms — the "watch RL actually improve the model" deliverable.

    python run_experiment.py
"""

import copy

import numpy as np

from rl import train_rl
from sft import train_sft
from task import evaluate, make_pairs


def bar(x, width=40):
    return "#" * int(x * width)


def main():
    rng = np.random.default_rng(42)
    train_pairs = make_pairs(2000, rng)
    eval_pairs = make_pairs(300, rng, exclude=frozenset(train_pairs))

    print("=== Stage 1: SFT (deliberately undertrained) ===")
    sft = train_sft(train_pairs, steps=250)
    base = evaluate(sft, eval_pairs)
    print(f"SFT held-out accuracy: {base:.3f}\n")

    results = {}
    for algo in ["grpo", "ppo"]:
        print(f"=== Stage 2: RL post-training [{algo.upper()}] ===")
        _, hist = train_rl(copy.deepcopy(sft), train_pairs, eval_pairs,
                           algo=algo, steps=400, eval_every=50)
        for h in hist:
            r = "  --" if h["mean_reward"] != h["mean_reward"] \
                else f"{h['mean_reward']:.2f}"
            print(f"  step {h['step']:>4}  eval {h['eval_acc']:.3f} "
                  f"|{bar(h['eval_acc']):<40}| reward {r}  kl {h['kl']:.3f}")
        results[algo] = hist[-1]["eval_acc"]
        print()

    print("=== Summary (held-out exact-match accuracy) ===")
    print(f"  SFT baseline : {base:.3f}")
    for algo, acc in results.items():
        print(f"  after {algo.upper():<5}: {acc:.3f}  (+{acc - base:.3f})")


if __name__ == "__main__":
    main()
