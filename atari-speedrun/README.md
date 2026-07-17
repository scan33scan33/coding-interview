# RL Post-Training v3: Speedrun a Mini-Atari Game (DQN vs PPO vs GRPO)

**Problem:** Write a small self-contained Atari-style game and train a network
to **complete it as fast as possible**, then judge the network by its
completion time. Solve it three ways — the value-based method that first
cleared Atari (DQN) and the two policy-gradient methods used to post-train
LLMs (PPO, GRPO) — all on CPU in a couple of minutes, no ROM, no gym, no ALE.

- **v1** [`grpo-ppo-loss`](../grpo-ppo-loss/) — the RL losses on synthetic tensors.
- **v2** [`rl-post-training`](../rl-post-training/) — SFT → GRPO/PPO on a real LM, judged by accuracy.
- **v3 (here)** — a game environment + DQN, PPO, and GRPO, judged by **speed of completion**.

```
=== MiniBreakout: how fast does each agent complete the game? ===
algorithm         completion  time-to-finish  speedrun  train (s)
random policy          4.2%            97.6        --         --
DQN                  100.0%            43.0        43         32
PPO                  100.0%            43.0        43         24
GRPO                 100.0%            43.0        43         51
```

All three reach the physical speed limit (43 steps); the interesting part is
*how* — three different answers to "what is the advantage baseline?"

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

## The policy-gradient agents: PPO and GRPO (`pg.py`)

The same game, solved by the two algorithms that post-train LLMs — a clean
contrast in *where the advantage baseline comes from*:

- **PPO** (actor-critic, on-policy). Collect a rollout with the current
  policy, estimate advantages with **GAE** against a learned **value head**,
  and take a few epochs of the **clipped surrogate** update (+ value loss +
  entropy bonus). The critic is the baseline.
- **GRPO** (critic-free). For each start state, sample a **group** of
  trajectories, and use each trajectory's return **standardized within the
  group** as its advantage — broadcast to every timestep. The group mean *is*
  the baseline; there is no value network. This is exactly the LLM GRPO recipe
  (a group of completions per prompt), with "prompt" = initial game state.
  `env.snapshot()`/`restore()` make the whole group start from an identical
  state, so the only variance is the sampled actions.

Both reuse the actor, the clipped surrogate, and the entropy bonus; they
differ only in the baseline. Contrast with the DQN, which has no policy at all
— it acts by `argmax Q` and learns from off-policy replay.

## Run it

```bash
pip install torch numpy pytest
python run_experiment.py     # DQN training curve + rollout, ~1 min
python compare_algos.py      # DQN vs PPO vs GRPO table,      ~2-3 min
python -m pytest -q          # ~2 min (all three algos)
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
- **GAE must cut the bootstrap at episode boundaries** (`(1 − done)` again, in
  a new disguise). A rollout spans several episodes; letting the advantage
  leak from one episode's terminal state into the next teaches the value of a
  future that isn't reachable. Same bug as DQN's terminal mask, same test idea.
- **GRPO collapses when the group agrees.** Once the policy is good, all
  trajectories in a group succeed identically → zero return variance →
  `advantage = 0/ε` = noise, which can *un-learn* a solved policy (visible as a
  late wobble). The fixes used here: an `ε` floor in the standardization
  (never divide by zero), a modest learning rate, and a small entropy bonus to
  keep the group from fully collapsing. This is a real GRPO failure mode, not a
  toy artifact.
- **On-policy vs off-policy cost.** DQN reuses every transition many times from
  replay, so it's sample-efficient but needs the target-net machinery to stay
  stable. PPO/GRPO throw each rollout away after a few epochs (on-policy), and
  GRPO pays extra: it samples a whole *group* per start state, which is why its
  wall-clock is the highest of the three here.

## Test map

DQN (`test_atari_speedrun.py`) and PPO/GRPO (`test_pg.py`):

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
| `test_gae_*` | GAE = discounted return-to-go; bootstrap cut at `done` |
| `test_group_advantages_standardized` | GRPO baseline: mean-0 group, 0 not NaN when collapsed |
| `test_clipped_surrogate_pushes_toward_positive_advantage` | Sign of the PG update |
| `test_snapshot_restore_replays_identical_trajectory` | GRPO's identical-start-state grouping |
| `test_ppo_learns_to_speedrun`, `test_grpo_learns_to_speedrun` | ≥ 90% completion, ≤ 46-step speedrun |
| `test_{ppo,grpo}_completion_time_drops` | Both learning curves move |

## Discussion Questions (interview follow-ups)

- Why is **time-to-complete with DNF = horizon** a better score than mean
  reward here? What failure would reward-only monitoring hide (dying fast to
  dodge the time penalty)?
- The ball clears a brick on the way *up*, before the catch. Why does that
  make the *miss = fatal* rule necessary for the catch skill to matter at all?
- DQN vs PPO vs GRPO — all three land at the same speedrun here. When does the
  choice actually matter (sample efficiency, continuous actions, reward
  sparsity, stability)? What breaks in DQN with a continuous action space?
- PPO learns a value head as its baseline; GRPO replaces it with a group mean.
  What does GRPO *give up* (per-timestep credit — its advantage is constant
  over a whole trajectory) and what does it *gain* (no critic to train, no GAE
  to tune)? On this game, why does that trade barely matter — and on what game
  would it matter a lot?
- GRPO samples a group from an *identical* start state (`snapshot`/`restore`).
  Why is fixing the start state the right call for isolating the effect of the
  actions, and what changes if the environment's dynamics are themselves
  stochastic?
- Scale to pixels: swap the 6-dim observation for the ASCII/pixel frame and a
  CNN. What changes — frame-stacking for velocity, reward clipping, a larger
  replay buffer, the target-sync period?
- The optimal speedrun is physically ~43 steps. How would you design a version
  where *faster* completion is genuinely learnable (variable ball speed,
  multi-brick angled clears) rather than floor-limited?

*PyTorch-only by design — the training loop is the point; a JAX/`lax.scan`
port of the replay-and-update loop is a good follow-up exercise.*
