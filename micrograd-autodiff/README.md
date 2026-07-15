# Reverse-Mode Autodiff Engine (Micrograd-Style)

**Problem:** Build a scalar **reverse-mode automatic differentiation engine**
from scratch: a `Value` node that records its inputs and a local backward
rule, and a `backward()` that runs the chain rule over a **topological sort**
of the computation graph. Then prove it works by training a tiny MLP on XOR
with nothing but your engine.

The "what does `loss.backward()` actually do?" interview question, in
executable form.

---

## Core Requirements

1. **Graph-building ops** — `+`, `*`, `**k`, `exp`, `tanh`, `relu`, plus the
   sugar (`-`, `/`, reflected variants) built *from* the primitives. Each op
   creates the output node and attaches a closure implementing its local
   gradient rule.

2. **`backward()`**
   - Topologically sort the graph from the output.
   - Seed `out.grad = 1.0`.
   - Run each node's `_backward` in **reverse topological order** — a node's
     grad must be complete before it propagates further.

3. **Gradient accumulation** — every local rule does `grad +=`, never
   `grad =`. A node used on two paths (`x*x + x`) receives gradient from
   both. This is *the* classic autodiff bug and has a dedicated test.

4. **Verification, three independent ways** (the tests):
   torch.autograd parity on a composite expression, finite-difference
   numerical gradients, and an end-to-end XOR training run.

---

## Behavior Notes / Gotchas

- **`+=` or death.** With `grad =`, `y = x*x + x` gives `dy/dx = 2x` or `1`
  depending on execution order instead of `2x + 1`. Any weight-sharing
  (every neural net) hits this immediately.
- **Why topological order and not simple recursion from the output?** A
  diamond graph (`x → a, b → out`) would propagate through `x` twice with
  partial gradients; topo order guarantees each node fires its rule exactly
  once, after its own grad is final.
- **Reverse vs forward mode.** One reverse pass gives ∂out/∂(every input) —
  perfect for ML (millions of params, scalar loss). Forward mode gives
  ∂(everything)/∂(one input) per pass — the opposite trade.
- **Local rules reuse the forward result** where possible: `exp` uses
  `out.data`, `tanh` uses the cached `t` — recomputing invites
  inconsistency; this mirrors what real frameworks memoize.
- **Zeroing grads is the caller's job** (`p.grad = 0` before each backward),
  exactly like PyTorch — and forgetting it reproduces PyTorch's accumulation
  semantics, which the XOR test would catch as non-convergence.

---

## Running the Smoke Test

```bash
pip install torch pytest        # torch used only as a gradient oracle
python -m pytest test_autodiff.py -v
```

| Test | Validates |
|------|-----------|
| `test_matches_torch_on_composite_expression` | Parity with torch.autograd |
| `test_gradient_accumulates_on_reused_node` | The `+=` bug |
| `test_diamond_graph` | Multi-path propagation |
| `test_topological_order_deep_chain` | 100-deep graphs, exact gradient |
| `test_relu_gates_gradient` | Non-differentiable point handling |
| `test_numerical_gradient_check` | Finite-difference oracle, 20 random points |
| `test_trains_a_tiny_mlp_on_xor` | End-to-end: engine trains a real (tiny) net |

---

## Discussion Questions (interview follow-ups)

- **From scalars to tensors** — what changes? (Local rules become
  vector-Jacobian products; broadcasting needs an unbroadcast/sum-to-shape
  rule in backward.)
- **Memory** — reverse mode stores the whole forward graph. Where do
  activation checkpointing and "recompute in backward" (FlashAttention)
  fit into this picture?
- **`no_grad` / inference mode** — what exactly gets skipped, and why is it
  more than a speed optimization (leaf mutation)?
- **Higher-order gradients** — what would it take for `backward` itself to
  be differentiable (build graph *during* backward)?
- **ReLU at 0** — the subgradient choice here is 0. Does it matter? What
  does PyTorch pick?
