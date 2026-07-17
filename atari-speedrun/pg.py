"""Policy-gradient agents for MiniBreakout: PPO and GRPO.

Same game, same completion-time metric as the DQN (dqn.py) — the point is to
contrast three algorithm families on identical footing:

  DQN  (dqn.py)  value-based, off-policy, replay buffer, Bellman targets
  PPO  (here)    actor-critic, on-policy, GAE advantages, clipped surrogate
  GRPO (here)    critic-free, on-policy, GROUP-relative trajectory advantages

PPO and GRPO share the actor, the clipped surrogate, and the entropy bonus;
they differ only in where the advantage comes from — a learned value baseline
(PPO/GAE) vs standardizing returns within a group of trajectories sampled from
the same start state (GRPO). That is exactly the LLM GRPO idea (a group of
completions per prompt), with "prompt" = initial game state.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from env import MAX_STEPS, N_ACTIONS, OBS_DIM, MiniBreakout


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def logits(self, x):
        return self.actor(self.trunk(x))


# ---------- greedy evaluation (shared metric with the DQN) ----------

@torch.no_grad()
def evaluate_policy(policy, n_episodes=500, seed=999):
    """Greedy (argmax) rollouts. Returns (completion_rate, mean_time,
    mean_time_on_win); mean_time counts a DNF as MAX_STEPS."""
    env = MiniBreakout(seed=seed)
    wins, times, win_times = 0, [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done, info = False, {"won": False, "steps": 0}
        while not done:
            a = int(policy.logits(torch.tensor(obs).unsqueeze(0)).argmax())
            obs, _, done, info = env.step(a)
        if info["won"]:
            wins += 1
            win_times.append(info["steps"])
            times.append(info["steps"])
        else:
            times.append(MAX_STEPS)
    return (wins / n_episodes, float(np.mean(times)),
            float(np.mean(win_times)) if win_times else float("nan"))


# ---------- shared math ----------

def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation. `dones[t]` marks the last step of an
    episode, where the bootstrap is cut (no future beyond a terminal state)."""
    adv = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv


def clipped_surrogate(new_logp, old_logp, adv, clip_eps=0.2):
    ratio = (new_logp - old_logp).exp()
    return -torch.minimum(ratio * adv,
                          ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()


def group_advantages(returns, eps=1e-4):
    """GRPO's baseline: standardize trajectory returns WITHIN a group sampled
    from the same start state. No critic — the group mean IS the baseline."""
    returns = np.asarray(returns, dtype=np.float32)
    return (returns - returns.mean()) / (returns.std() + eps)


# ---------- PPO ----------

def collect_rollout(policy, env, obs, n_steps, rng):
    """Run the current policy for n_steps, splitting across episodes.
    Returns arrays plus the value at the final (possibly mid-episode) state."""
    O, A, R, D, LP, V = [], [], [], [], [], []
    for _ in range(n_steps):
        obs_t = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, value = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        O.append(obs); A.append(int(a)); LP.append(float(dist.log_prob(a)))
        V.append(float(value))
        obs, r, done, _ = env.step(int(a))
        R.append(r); D.append(float(done))
        obs = env.reset() if done else obs
    with torch.no_grad():
        _, last_v = policy(torch.tensor(obs).unsqueeze(0))
    return (np.array(O, np.float32), np.array(A), np.array(R, np.float32),
            np.array(D, np.float32), np.array(LP, np.float32),
            np.array(V + [float(last_v)], np.float32), obs)


def train_ppo(iters=300, n_steps=512, epochs=4, minibatch=128, lr=3e-3,
              gamma=0.99, lam=0.95, clip_eps=0.2, ent_coef=0.01,
              value_coef=0.5, eval_every=30, seed=0, log=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = ActorCritic()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = MiniBreakout(seed=seed)
    obs = env.reset()

    history = []
    c0, t0, w0 = evaluate_policy(policy)
    history.append({"step": 0, "completion": c0, "mean_time": t0, "win_time": w0})
    if log:
        print(f"iter {0:>4}  completion {c0:5.1%}  mean_time {t0:5.1f}")

    for it in range(1, iters + 1):
        O, A, R, D, LP, Vplus, obs = collect_rollout(policy, env, obs, n_steps, rng)
        adv = gae(R, Vplus, D, gamma, lam)
        ret = adv + Vplus[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        O_t = torch.tensor(O); A_t = torch.tensor(A)
        LP_t = torch.tensor(LP); adv_t = torch.tensor(adv); ret_t = torch.tensor(ret)

        idx = np.arange(n_steps)
        for _ in range(epochs):
            rng.shuffle(idx)
            for s in range(0, n_steps, minibatch):
                b = idx[s:s + minibatch]
                logits, value = policy(O_t[b])
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(A_t[b])
                pg = clipped_surrogate(new_lp, LP_t[b], adv_t[b], clip_eps)
                vloss = F.mse_loss(value, ret_t[b])
                loss = pg + value_coef * vloss - ent_coef * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        if it % eval_every == 0:
            c, t, w = evaluate_policy(policy)
            history.append({"step": it, "completion": c, "mean_time": t, "win_time": w})
            if log:
                print(f"iter {it:>4}  completion {c:5.1%}  mean_time {t:5.1f}  "
                      f"speedrun {w:5.1f}")
    return policy, history


# ---------- GRPO ----------

def sample_trajectory(policy, env, init_state, max_len, rng):
    """One trajectory from a FIXED initial state (restore snapshot)."""
    obs = env.restore(init_state)
    O, A, LP, total_r = [], [], [], 0.0
    done = False
    for _ in range(max_len):
        with torch.no_grad():
            dist = torch.distributions.Categorical(
                logits=policy.logits(torch.tensor(obs).unsqueeze(0)))
        a = dist.sample()
        O.append(obs); A.append(int(a)); LP.append(float(dist.log_prob(a)))
        obs, r, done, _ = env.step(int(a))
        total_r += r
        if done:
            break
    return O, A, LP, total_r


def train_grpo(iters=300, prompts_per_iter=8, group_size=8, epochs=4,
               minibatch=256, lr=2e-3, clip_eps=0.2, ent_coef=0.004,
               eval_every=30, seed=0, log=True):
    """Critic-free: for each start state, sample `group_size` trajectories,
    standardize their TOTAL returns within the group -> one advantage per
    trajectory, broadcast to all its timesteps. No value network, no GAE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = ActorCritic()          # critic head unused; reused for a fair net
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = MiniBreakout(seed=seed)

    history = []
    c0, t0, w0 = evaluate_policy(policy)
    history.append({"step": 0, "completion": c0, "mean_time": t0, "win_time": w0})
    if log:
        print(f"iter {0:>4}  completion {c0:5.1%}  mean_time {t0:5.1f}")

    for it in range(1, iters + 1):
        O, A, LP, ADV = [], [], [], []
        for _ in range(prompts_per_iter):
            env.reset()
            init_state = env.snapshot()
            returns, trajs = [], []
            for _ in range(group_size):
                o, a, lp, tot = sample_trajectory(policy, env, init_state,
                                                  MAX_STEPS, rng)
                trajs.append((o, a, lp)); returns.append(tot)
            # Group-relative advantage: standardize returns WITHIN the group.
            adv = group_advantages(returns)
            for (o, a, lp), ad in zip(trajs, adv):
                O.extend(o); A.extend(a); LP.extend(lp); ADV.extend([ad] * len(o))

        O_t = torch.tensor(np.array(O, np.float32)); A_t = torch.tensor(A)
        LP_t = torch.tensor(np.array(LP, np.float32))
        ADV_t = torch.tensor(np.array(ADV, np.float32))

        idx = np.arange(len(O))
        for _ in range(epochs):
            rng.shuffle(idx)
            for s in range(0, len(O), minibatch):
                b = idx[s:s + minibatch]
                dist = torch.distributions.Categorical(logits=policy.logits(O_t[b]))
                new_lp = dist.log_prob(A_t[b])
                pg = clipped_surrogate(new_lp, LP_t[b], ADV_t[b], clip_eps)
                loss = pg - ent_coef * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        if it % eval_every == 0:
            c, t, w = evaluate_policy(policy)
            history.append({"step": it, "completion": c, "mean_time": t, "win_time": w})
            if log:
                print(f"iter {it:>4}  completion {c:5.1%}  mean_time {t:5.1f}  "
                      f"speedrun {w:5.1f}")
    return policy, history
