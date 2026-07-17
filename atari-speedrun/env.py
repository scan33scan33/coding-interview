"""MiniBreakout: a self-contained, dependency-free Atari-style game where the
agent is judged by HOW FAST it completes the game (a "speedrun").

Breakout is the canonical Atari RL benchmark. This is a shrunk, grid-based
version: bounce the ball off the paddle to chip away a wall of bricks. The
game is "completed" when the wall is cleared. No ROM / gym / ALE dependency,
and it trains a DQN on CPU in a couple of minutes.

Design choices that make SPEED the learnable objective:

  * Any ball contact with the top row removes one brick from the wall
    ("chip the wall"), so the board is always clearable — unlike literal
    Breakout, where a pure-diagonal grid ball can only reach one color of
    the checkerboard and half the bricks are unreachable.
  * A MISS (ball falls past the paddle) is NOT fatal: the ball respawns and
    the run continues, but the wasted round-trip costs time (and a penalty).
    So the only way to finish quickly is to stop missing — survival skill
    translates directly into completion speed.
  * A per-step time cost makes every wasted step hurt the return.

Metric: steps-to-complete (lower is better), plus completion rate within a
step budget. A random policy misses constantly and finishes slowly or times
out; a trained policy tracks the ball and speedruns the wall.

Coordinates: x in [0, W), y in [0, H); row 0 top (bricks), row H-1 paddle.
"""

import numpy as np

W, H = 5, 5          # screen size
PADDLE_W = 2         # paddle spans PADDLE_W columns
N_BRICKS = 6         # wall size (bricks to clear to win)
MAX_STEPS = 100

LEFT, STAY, RIGHT = 0, 1, 2
N_ACTIONS = 3

BRICK_REWARD = 1.0
WIN_BONUS = 5.0
MISS_PENALTY = -1.0
TIME_COST = 0.05

OBS_DIM = 6          # bx, by, vx, vy, px, bricks_left (all normalized)


class MiniBreakout:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.bricks_left = N_BRICKS
        self.px = (W - PADDLE_W) // 2
        self._launch()
        self.steps = 0
        self.done = False
        return self._obs()

    def _launch(self):
        """(Re)spawn the ball just above the paddle, heading up at an angle."""
        self.bx = int(self.rng.integers(0, W))
        self.by = H - 2
        self.vx = int(self.rng.choice([-1, 1]))
        self.vy = -1

    def _obs(self):
        return np.array([self.bx / (W - 1), self.by / (H - 1),
                         (self.vx + 1) / 2, (self.vy + 1) / 2,
                         self.px / (W - PADDLE_W),
                         self.bricks_left / N_BRICKS], dtype=np.float32)

    def _paddle_cols(self):
        return range(self.px, self.px + PADDLE_W)

    def step(self, action):
        assert not self.done, "step() called on a finished episode"
        self.steps += 1

        if action == LEFT:
            self.px = max(0, self.px - 1)
        elif action == RIGHT:
            self.px = min(W - PADDLE_W, self.px + 1)

        reward = -TIME_COST

        # Horizontal move with wall bounce.
        nx = self.bx + self.vx
        if nx < 0 or nx >= W:
            self.vx = -self.vx
            nx = self.bx + self.vx

        ny = self.by + self.vy
        if self.vy == -1 and ny <= 0:
            # Top row: chip one brick off the wall, bounce down.
            ny = 0
            if self.bricks_left > 0:
                self.bricks_left -= 1
                reward += BRICK_REWARD
            self.vy = 1
        elif self.vy == 1 and ny >= H - 1:
            if nx in self._paddle_cols():
                # Caught: bounce up with an angle set by the hit offset.
                offset = nx - self.px
                self.vx = -1 if offset < PADDLE_W / 2 else 1
                self.vy = -1
                ny = H - 1
            else:
                # Missed: game over. Winning therefore requires a whole chain
                # of successful catches, which is the survival skill the agent
                # must learn — and the reason a random policy rarely finishes.
                self.bx, self.by = nx, ny
                self.done = True
                return self._obs(), reward + MISS_PENALTY, True, self._info(won=False)

        self.bx, self.by = nx, ny

        if self.bricks_left == 0:
            self.done = True
            return self._obs(), reward + WIN_BONUS, True, self._info(won=True)

        if self.steps >= MAX_STEPS:
            self.done = True
            return self._obs(), reward, True, self._info(won=False)

        return self._obs(), reward, False, self._info(won=False)

    def _info(self, won):
        return {"won": won, "steps": self.steps, "bricks_left": self.bricks_left}

    def render(self):
        """ASCII 'pixel' frame, Atari-style, for eyeballing a rollout."""
        grid = [[" " for _ in range(W)] for _ in range(H)]
        for c in range(min(self.bricks_left, W)):     # remaining wall (top row)
            grid[0][c] = "="
        if 0 <= self.by < H:
            grid[self.by][self.bx] = "o"
        for c in self._paddle_cols():
            grid[H - 1][c] = "#"
        border = "+" + "-" * W + "+"
        return border + "\n" + "\n".join("|" + "".join(r) + "|" for r in grid) \
            + "\n" + border
