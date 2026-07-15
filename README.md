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

## Post-training & alignment

| Problem | Topic |
|---------|-------|
| [`dpo-loss`](dpo-loss/) | Direct Preference Optimization (RLHF without RL) |
| [`contrastive-retrieval`](contrastive-retrieval/) | Bi-encoder + InfoNCE, false-negative masking |

## Inference & decoding

| Problem | Topic |
|---------|-------|
| [`decoding-sampling`](decoding-sampling/) | Temperature, top-k, top-p, repetition penalty, beam search |
| [`speculative-decoding`](speculative-decoding/) | Draft/verify rejection sampling, exactness proof |

## Data & classical ML

| Problem | Topic |
|---------|-------|
| [`bpe-tokenizer`](bpe-tokenizer/) | Byte-level BPE: train / encode / decode |
| [`kmeans-numpy`](kmeans-numpy/) | k-means++, Lloyd's algorithm, vectorized NumPy |
| [`data-preprocessing`](data-preprocessing/) | Production-grade feature preprocessing pipeline |

## Classic algorithms

| Problem | Topic |
|---------|-------|
| [`a-star-pathfinding`](a-star-pathfinding/) | A* on a grid (C++) |
| [`semantic-costmap-bfs`](semantic-costmap-bfs/) | BFS over a semantic costmap (C++) |

## Running tests

```bash
pip install torch numpy pytest
cd <problem-dir> && python -m pytest -v
```
