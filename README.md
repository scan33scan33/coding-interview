# Coding Interview Problems — Frontier MLE / Research Scientist

Self-contained interview problems with solutions and pytest smoke suites.
Each directory has a README with the problem statement, requirements,
gotchas, and interview follow-up questions. Topics are drawn from
commonly-reported frontier-lab loops (OpenAI, Anthropic, DeepMind, Meta
GenAI) on 1point3acres, Blind, Reddit, and MLE interview writeups.

## LLM architecture & training

| Problem | Topic |
|---------|-------|
| [`gqa-rope-attention`](gqa-rope-attention/) | Grouped-query attention, RoPE, incremental KV cache |
| [`online-softmax-attention`](online-softmax-attention/) | FlashAttention core: tiled attention, online softmax |
| [`transformer-debugging`](transformer-debugging/) | Find 6 planted bugs in a transformer training script |
| [`jax_transformer`](jax_transformer/) | Transformer classifier in JAX |
| [`simple-mlp`](simple-mlp/) | MLP + backprop from scratch |
| [`lora-finetuning`](lora-finetuning/) | LoRA adapters: init, merge/unmerge, injection |
| [`mixture-of-experts`](mixture-of-experts/) | MoE routing, load-balancing loss, capacity factor |
| [`micrograd-autodiff`](micrograd-autodiff/) | Reverse-mode autodiff engine from scratch |
| [`adamw-optimizer`](adamw-optimizer/) | AdamW, grad clipping, warmup-cosine schedule |

## Post-training & alignment

| Problem | Topic |
|---------|-------|
| [`dpo-loss`](dpo-loss/) | Direct Preference Optimization (RLHF without RL) |
| [`grpo-ppo-loss`](grpo-ppo-loss/) | GRPO/PPO clipped loss, k3 KL estimator |
| [`contrastive-retrieval`](contrastive-retrieval/) | Bi-encoder + InfoNCE, false-negative masking |

## Inference & decoding

| Problem | Topic |
|---------|-------|
| [`decoding-sampling`](decoding-sampling/) | Temperature, top-k, top-p, repetition penalty, beam search |
| [`speculative-decoding`](speculative-decoding/) | Draft/verify rejection sampling, exactness proof |
| [`int8-quantization`](int8-quantization/) | Symmetric/asymmetric INT8, per-channel, W8A8 linear |

## Data & classical ML

| Problem | Topic |
|---------|-------|
| [`bpe-tokenizer`](bpe-tokenizer/) | Byte-level BPE: train / encode / decode |
| [`kmeans-numpy`](kmeans-numpy/) | k-means++, Lloyd's algorithm, vectorized NumPy |
| [`ml-metrics`](ml-metrics/) | ROC-AUC with ties, average precision, calibration (ECE) |
| [`ivf-vector-search`](ivf-vector-search/) | IVF approximate nearest-neighbor index (FAISS-style) |
| [`data-preprocessing`](data-preprocessing/) | Production-grade feature preprocessing pipeline |

## Systems & scale

| Problem | Topic |
|---------|-------|
| [`ring-allreduce`](ring-allreduce/) | Data-parallel gradient sync, bandwidth-optimal collective |
| [`count-min-sketch`](count-min-sketch/) | Streaming frequency estimates, heavy hitters |
| [`inmemory-database`](inmemory-database/) | Anthropic/CodeSignal-style 4-level build: KV store with TTL + backup |

## Classic algorithms

| Problem | Topic |
|---------|-------|
| [`a-star-pathfinding`](a-star-pathfinding/) | A* on a grid (C++) |
| [`semantic-costmap-bfs`](semantic-costmap-bfs/) | BFS over a semantic costmap (C++) |

## Frameworks

Every tensor/ML problem ships **both PyTorch and JAX** implementations
side by side (`*_torch.py` / `*_jax.py` next to the original), with tests
for each — OpenAI-style loops lean PyTorch, DeepMind/Anthropic-style loops
lean JAX, and porting between the two is itself good interview practice
(explicit PRNG keys, pytrees + pure functions instead of Modules,
`stop_gradient` instead of `no_grad`, dense dispatch instead of dynamic
shapes).

Deliberately framework-free (building without a framework *is* the
exercise, or no tensors are involved): `micrograd-autodiff`, `simple-mlp`
(manual backprop), `bpe-tokenizer`, `count-min-sketch`,
`inmemory-database`, `data-preprocessing` (pandas), and the C++ problems.

## Running tests

```bash
pip install torch jax numpy pytest
cd <problem-dir> && python -m pytest -v
```
