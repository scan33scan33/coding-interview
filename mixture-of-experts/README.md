# Mixture-of-Experts Layer: Routing, Load Balancing, Capacity

**Problem:** Implement an **MoE feed-forward layer** — the sparse
architecture behind Mixtral, DeepSeek-V3, and Switch Transformer: a learned
**top-k router**, per-token expert **dispatch and weighted combine**, the
Switch **load-balancing auxiliary loss**, and a **capacity factor** with
token dropping.

---

## Core Requirements

1. **Routing**
   - `softmax(router(x))` over experts; take top-k per token.
   - **Renormalize the chosen k gate weights to sum to 1** — scaling expert
     outputs by the raw softmax values (which sum to < 1 over k of E
     experts) is the classic bug; the equal-experts test catches it exactly.

2. **Dispatch / combine**
   - Gather each expert's assigned tokens, run the expert **once on the
     batch of its tokens** (never per-token loops over N), scatter-add the
     weighted outputs back.

3. **Load-balancing loss (Switch Transformer)**
   - `aux = E · Σₑ fₑ · Pₑ` where `fₑ` = fraction of tokens whose **top-1**
     pick is expert e (hard counts, no gradient) and `Pₑ` = mean gate
     probability (soft, differentiable).
   - Minimum value 1.0 exactly at uniform routing; the gradient flows
     through `Pₑ` and is steered by the skew in `fₑ`.

4. **Capacity factor**
   - `cap = ⌈cf · N · k / E⌉` assignments per expert; overflow assignments
     are dropped (that token loses that expert's contribution — the
     transformer's residual connection is what keeps dropped tokens alive).
   - Report the drop count; it's a first-class training health metric.

---

## Interface

```python
class MoELayer(nn.Module):
    def __init__(self, dim, n_experts, hidden=None, k=2, capacity_factor=None)
    def forward(self, x_ND) -> (out_ND, aux_loss, n_dropped)

def load_balancing_loss(gates_NE, expert_index_NK) -> scalar
```

Shape-suffix naming: `N` tokens, `D` dim, `E` experts, `K` top-k.

---

## Behavior Notes / Gotchas

- **Why load balancing needs a loss at all:** routing is self-reinforcing —
  a slightly-preferred expert gets more tokens, learns faster, gets preferred
  more. Untreated, MoEs collapse to 1-2 live experts. The aux loss is the
  counterpressure; watch `f_E` entropy during training.
- **The `f · P` trick.** You want to penalize `Σ fₑ²` (hard load), but hard
  counts have no gradient. `Pₑ` is the differentiable surrogate that moves
  together with `fₑ`; the product gives a gradient that redistributes soft
  mass away from overloaded experts.
- **Renormalization invariant:** with all experts identical, the layer must
  equal a single expert *exactly*, for any router state. This is the
  one-line test that catches missing renormalization, double-weighting, and
  scatter bugs all at once.
- **Dropping order is policy.** This implementation keeps the first `cap`
  assignments in token order (Switch does position-based dropping too);
  production systems may drop by lowest gate weight instead. Say which you
  chose and why.
- **Don't route with `for token in tokens`.** Group by expert and batch;
  the per-expert gather/scatter is what makes MoE viable on accelerators
  (and in real systems becomes an all-to-all across devices).

---

## Running the Smoke Test

```bash
pip install torch pytest
python -m pytest test_moe.py -v
```

| Test | Validates |
|------|-----------|
| `test_identical_experts_reduce_to_single_expert` | Renormalization + combine correctness |
| `test_top1_routes_to_argmax_expert` | Hard routing follows the gate |
| `test_mixture_weights_are_renormalized` | Weights sum to 1 over chosen k |
| `test_aux_loss_minimized_at_uniform_routing` | Value 1.0 at uniform; skew increases it |
| `test_aux_loss_gradient_reaches_router` | The surrogate is differentiable |
| `test_capacity_drops_overflow` | cap formula + dropped tokens zeroed |
| `test_no_capacity_no_drops` | No capacity ⇒ lossless |
| `test_end_to_end_trains` | Router + experts co-train |

---

## Discussion Questions (interview follow-ups)

- **Why do MoEs win on FLOPs?** Params scale with E, per-token compute
  with k. Where does the memory/communication cost actually land (expert
  parallelism, all-to-all)?
- **Aux-loss-free balancing** — DeepSeek-V3 replaces the loss with a
  per-expert bias adjusted by a feedback rule. Why does a loss-based
  penalty fight the LM objective, and how does the bias approach avoid it?
- **Shared experts** — why does DeepSeek route every token through a small
  set of always-on experts alongside the routed ones?
- **Inference batching** — expert load varies per batch; how do serving
  systems handle worst-case expert hotspots (expert replication, capacity
  at inference)?
- **Router z-loss** — what numerical pathology in the router logits does
  `log²Σeᶻ` regularization prevent?
