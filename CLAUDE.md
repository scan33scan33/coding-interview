# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this repo is

A collection of **self-contained coding-interview problems** aimed at
frontier-lab MLE / Research Scientist loops (OpenAI, Anthropic, DeepMind,
Meta GenAI). Each top-level directory is one problem: a `README.md` with the
problem statement, one or more reference implementations, and a pytest smoke
suite. Nothing is a package; there is no shared library, no build system, and
no cross-directory imports. `README.md` at the repo root is the index.

The value of a problem lives in three places that must stay consistent:
1. the **README** (statement, requirements, gotchas, interview follow-ups),
2. the **implementation(s)** (clean, idiomatic reference solutions), and
3. the **tests** (fast smoke suites that pin the correctness bar).

When you change one, check the other two.

## Directory layout

Each problem directory is independent and self-contained:

```
<problem>/
  README.md            # problem statement + gotchas + interview questions
  <impl>.py            # reference implementation(s) — see naming below
  test_<impl>.py       # pytest smoke suite
```

See `README.md` (root) for the full categorized index. Categories:
LLM architecture & training, post-training & alignment, inference & decoding,
data & classical ML, systems & scale, and classic algorithms (C++).

## File-naming conventions (important)

Most tensor/ML problems ship **both PyTorch and JAX** implementations side by
side. There are two layouts in use — match the one already present in a
directory rather than introducing a third:

**Two-way (`impl.py` is the PyTorch reference) + JAX port.** Here the
canonical file is already PyTorch:
```
lora.py   lora_jax.py   test_lora.py   test_lora_jax.py
```
Directories: `adamw-optimizer`, `contrastive-retrieval`, `dpo-loss`,
`gqa-rope-attention`, `grpo-ppo-loss`, `lora-finetuning`, `mixture-of-experts`,
`transformer-debugging`.

**Three-way (`impl.py` is a NumPy/reference) + `_torch` + `_jax` ports.**
Here the canonical file is framework-free and two ports sit beside it, tested
by a single **parametrized** `test_*_ports.py`:
```
kmeans.py  kmeans_torch.py  kmeans_jax.py
test_kmeans.py  test_kmeans_ports.py
```
Directories: `decoding-sampling`, `int8-quantization`, `ivf-vector-search`,
`kmeans-numpy`, `ml-metrics`, `online-softmax-attention`, `ring-allreduce`,
`speculative-decoding`.

**Deliberately framework-free** (building without a framework *is* the
exercise, or no tensors are involved): `micrograd-autodiff`, `simple-mlp`
(manual backprop), `bpe-tokenizer`, `count-min-sketch`, `inmemory-database`,
`data-preprocessing` (pandas), and the two C++ problems.

**PyTorch-only by design** (the training loop is the deliverable; a JAX port
is itself the suggested follow-up): `rl-post-training`, `atari-speedrun`.

If you add a new framework port to a problem, add tests for it too — a port
without a test is incomplete.

## The shape-suffix naming convention

Tensor code throughout this repo names variables by their shape, e.g.
`x_BLD`, `cos_TJ`, `out_BhLK`. A tensor's name tells you its shape; if an op's
output name doesn't follow from its inputs, the op is probably wrong. Each
file/README documents its own letters (typical: `B`=batch, `L`=length,
`D`=model dim, `H`=heads, `K`=head dim, `V`=vocab). **Preserve this
convention** when editing — do not rename shaped tensors to bare names.

## Running tests

```bash
pip install torch jax numpy pytest       # superset of deps; not all needed per problem
cd <problem-dir> && python -m pytest -v
```

- Tests import modules by bare name (`from kmeans import ...`), so **always run
  pytest from inside the problem directory**, not the repo root.
- Suites are deliberately **fast smoke tests** — small tensors, fixed seeds
  (`torch.manual_seed(0)`, `np.random.default_rng(seed)`), tolerance-based
  `allclose` assertions. Keep new tests in the same style; they should run in
  seconds without a GPU.
- Some problems need only NumPy (`kmeans-numpy`, `count-min-sketch`,
  `ml-metrics` base), others need torch and/or jax. Install per problem as
  needed.
- There is **no CI** (`.github/` does not exist) and no repo-wide test runner.
  Verify a change by running that problem's suite locally.

### C++ problems

`a-star-pathfinding` and `semantic-costmap-bfs` are single-file C++ with no
Python tests. Compile and run directly, e.g.:
```bash
cd semantic-costmap-bfs && g++ -std=c++17 -O2 semantic_costmap_bfs.cpp -o semantic_costmap_bfs && ./semantic_costmap_bfs
```
The built binary and `*.o` / `*.dSYM/` are gitignored.

## Coding conventions

- **Python style:** standard library + NumPy / PyTorch / JAX only. Imports
  ordered stdlib → third-party → local. Functions and modules carry short
  docstrings; comments explain *why* (the gotcha), not *what*.
- **JAX idioms** (when porting from PyTorch): explicit PRNG keys, pytrees +
  pure functions instead of `nn.Module`, `stop_gradient` instead of `no_grad`,
  dense dispatch instead of dynamic shapes. Keep the JAX port numerically
  matched to the PyTorch/NumPy reference — the port tests assert `allclose`
  across implementations.
- **Determinism:** every test seeds its RNG. New code and tests must be
  reproducible.
- **READMEs** follow a consistent shape: problem statement → core requirements
  → interface → behavior notes / gotchas → running the test → a table mapping
  tests to what they validate → interview discussion questions. Mirror this
  structure when adding a problem.

## Adding a new problem

1. Create `<problem>/` with a `README.md` following the structure above.
2. Add the reference implementation using the shape-suffix convention; add a
   JAX (and/or PyTorch) port if the problem is tensor-based, choosing the
   two-way or three-way layout to match sibling problems.
3. Add a `test_<impl>.py` smoke suite (and `test_*_ports.py` if there are
   multiple framework ports) — small, seeded, fast.
4. Add a row to the appropriate table in the root `README.md`.

## Git workflow

- Work happens on a feature branch, not `master`. History shows changes land
  via pull requests merged into `master`.
- Commit messages are short and descriptive, scoped to the problem
  (e.g. `atari-speedrun: add PPO and GRPO alongside DQN`).
- Do not create a pull request unless explicitly asked.
