# RL Post-Training v3: Speedrun a Mini-Atari Game with DQN

**Problem:** Write a small self-contained Atari-style game and train a network
to **complete it as fast as possible**, then judge the network by its
completion time. This is the deep-RL-from-pixels lineage (DQN — the algorithm
that first cleared Atari), shrunk to run on CPU in about a minute with no ROM,
no gym, no ALE.

- **v1** [`grpo-ppo-loss`](../grpo-ppo-loss/) — the RL losses on synthetic tensors.
- **v2** [`rl-post-training`](../rl-post-training/) — SFT → GRPO/PPO on a real LM, judged by accuracy.
- **v3 (here)** — a game environment + DQN, judged by **speed of completion**.

```
random policy :   4.2% completion,  ~98 steps to finish
trained DQN   : 100.0% completion,   43 steps to finish  (the physical speed limit)
```

## The game: MiniBreakout (`env.py`)

A 5×5 grid Breakout. Bounce the ball off a 2-wide paddle to chip away a wall
of 6 bricks; the game is **completed** when the wall is cleared.

```
+-----+          +-----+
|=====|   ...    |     |     row 0: the brick wall (depletes)
|     |          |     |
|  o  |          | o   |     o : the ball (moves diagonally)
| ##  |          |   ##|     # : the paddle (you control it)
+-----+          +-----+
 start            cleared -> WIN
```

Three design choices make **speed** the learnable objective:

- **Any top-row contact chips one brick** ("chip the wall"). Literal Breakout
  is unwinnable on a small grid — a pure-diagonal ball lives on one color of
  the checkerboard, so half the bricks are physically unreachable (verified:
  8000 random policies cleared at most 2 of 5 aligned bricks, never winning).
  Chipping the wall sidesteps the resonance and keeps the board always
  clearable.
- **A miss is fatal.** Winning requires a whole *chain* of successful catches
  (the ball must be returned to the top 6 times), which is the survival skill
  a random policy lacks — it completes only ~4% of the time.
- **A per-step time cost** makes the return prefer finishing sooner.

**The metric — time-to-complete** (`train.evaluate`): steps to clear the wall,
with a did-not-finish counted as the full step budget (`MAX_STEPS`). A policy
that never finishes scores ~the cap; a policy that speedruns scores ~43, which
is the physical floor (6 bricks × one ball round-trip each).

## The agent: DQN (`dqn.py`)

Textbook DQN (Mnih et al. 2015) with the four ingredients that make it stable,
each covered by a test:

1. **Replay buffer** — decorrelate consecutive transitions.
2. **Target network** — a periodically-frozen copy for bootstrap targets.
3. **ε-greedy exploration** with linear decay.
4. **Bellman target** `y = r + γ·maxₐ' Q_target(s',a')`, with terminal
   transitions **not bootstrapped** (`y = r` when done — the `(1−done)` mask).

The agent sees a compact 6-dim observation (ball x/y, velocity, paddle x,
bricks left); a CNN over the ASCII pixel frame is the natural follow-up.

## Run it

```bash
pip install torch numpy pytest
python run_experiment.py     # ~1 minute on CPU
python -m pytest -q          # ~25 s
```

Actual training curve (`run_experiment.py`, seed 0):

```
step  completion  time-to-complete
   0     28.0%   84.0     (untrained net: a lucky constant policy)
1500     91.6%   47.8
3000    100.0%   43.0     (optimal speedrun reached)
...
```

## Behavior Notes / Gotchas

- **The `(1 − done)` mask is the classic DQN bug.** Bootstrap past a terminal
  state and you add a phantom `γ·V(s')` to the value of winning/missing;
  training then chases a target that doesn't exist. `test_terminal_target_is_not_bootstrapped`
  pins it: with `done=1` the loss must reduce to `Q(s,a) → r`.
- **Why a random *network* beats a random *policy*** (28% vs 4% at step 0).
  Greedy-evaluating an untrained net gives a fixed, arbitrary — but *constant*
  and sometimes lucky — action bias, which survives longer than uniform coin
  flips. The uniform-random 4% is the honest floor; both are reported.
- **DNF must cost the full horizon in the metric.** If you averaged steps only
  over completed episodes, a policy that dies fast would look "fast." Capping
  a non-completion at `MAX_STEPS` is what makes "lower time = better" honest —
  a miss at step 12 scores 100, not 12.
- **The speed floor is physical.** Each brick needs one ball round-trip
  (~2·(H−1) steps), so ~43 steps is optimal; the agent can't beat it, only
  reach it. Judging by speed here means judging how close to the floor the
  policy gets *and* how reliably it finishes at all.
- **The task is verified winnable** before any training
  (`test_game_is_completable`): a hand-coded tracker clears the wall 50/50.
  If the environment were unwinnable, a flat learning curve would look like an
  RL bug rather than a broken reward.

## Test map

| Test | Validates |
|------|-----------|
| `test_reset_and_obs_shape`, `test_paddle_stays_in_bounds` | Env basics, normalized obs |
| `test_catch_bounces_ball_up`, `test_miss_ends_episode` | Paddle catch vs fatal miss |
| `test_top_contact_chips_a_brick` | The chip-the-wall rule + reward |
| `test_game_is_completable` | Winnable by a tracker (task sanity) |
| `test_terminal_target_is_not_bootstrapped` | The `(1−done)` Bellman mask |
| `test_replay_buffer_sampling_shapes`, `test_epsilon_greedy_is_greedy_at_zero` | DQN plumbing |
| `test_td_update_reduces_loss` | Gradient step lowers TD error |
| `test_random_baseline_rarely_completes` | Random ~4% (the floor) |
| `test_dqn_completes_reliably` | Trained ≥ 90% completion |
| `test_dqn_speedruns_near_optimal` | Win time ≤ 46 (~43 floor), mean ≤ 55 |
| `test_completion_time_drops_over_training` | The learning curve moves |

## Discussion Questions (interview follow-ups)

- Why is **time-to-complete with DNF = horizon** a better score than mean
  reward here? What failure would reward-only monitoring hide (dying fast to
  dodge the time penalty)?
- The ball clears a brick on the way *up*, before the catch. Why does that
  make the *miss = fatal* rule necessary for the catch skill to matter at all?
- DQN vs policy-gradient (the v2 GRPO/PPO) — when do you reach for a value-based
  method vs a policy-gradient one? What breaks in DQN with a continuous action
  space?
- Scale to pixels: swap the 6-dim observation for the ASCII/pixel frame and a
  CNN. What changes — frame-stacking for velocity, reward clipping, a larger
  replay buffer, the target-sync period?
- The optimal speedrun is physically ~43 steps. How would you design a version
  where *faster* completion is genuinely learnable (variable ball speed,
  multi-brick angled clears) rather than floor-limited?

*PyTorch-only by design — the training loop is the point; a JAX/`lax.scan`
port of the replay-and-update loop is a good follow-up exercise.*
