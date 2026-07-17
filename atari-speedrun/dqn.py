"""DQN agent for MiniBreakout (Mnih et al. 2015, the algorithm that first
cleared Atari). Small enough to train on CPU in a couple of minutes.

The four ingredients that make DQN stable, each of which the tests probe:
  1. a replay buffer (decorrelate consecutive transitions),
  2. a target network (a periodically-frozen copy for bootstrap targets),
  3. epsilon-greedy exploration with decay,
  4. the Bellman target  y = r + gamma * max_a' Q_target(s', a')  with the
     terminal transition NOT bootstrapped (y = r when done).
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from env import N_ACTIONS, OBS_DIM


class QNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def td_loss(q, target_q, s, a, r, s2, done, gamma=0.99):
    """Bellman error. The `(1 - done)` mask is load-bearing: a terminal
    state has no future, so its target is just the reward — bootstrapping
    past a win/miss teaches the agent a value that doesn't exist."""
    q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_q(s2).max(dim=1).values
        target = r + gamma * next_q * (1 - done)
    return F.smooth_l1_loss(q_sa, target)


@torch.no_grad()
def select_action(q, obs, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return int(q(obs_t).argmax(dim=1))
