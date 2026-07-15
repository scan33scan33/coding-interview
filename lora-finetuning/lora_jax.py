"""JAX port of LoRA (see lora.py for the PyTorch original).

Functional flavor: the frozen base weights and the trainable adapter are
SEPARATE pytrees, so "freezing" falls out of which pytree you differentiate —
JAX's version of requires_grad_(False). merge/unmerge return new base
weights instead of mutating in place.
"""

import jax
import jax.numpy as jnp


def init_lora(key, in_features, out_features, r=8, alpha=16):
    """A random (Kaiming-ish), B zero: identity at init, nonzero gradients."""
    a = jax.random.uniform(key, (r, in_features),
                           minval=-1.0, maxval=1.0) * (3.0 / in_features) ** 0.5
    return {"A": a, "B": jnp.zeros((out_features, r)), "scaling": alpha / r}


def lora_apply(base, lora, x_BI):
    """y = x W^T + b + scaling * (x A^T) B^T — two skinny matmuls, never
    materializing the (O, I) product BA."""
    y_BO = x_BI @ base["w"].T
    if base.get("b") is not None:
        y_BO = y_BO + base["b"]
    return y_BO + (x_BI @ lora["A"].T) @ lora["B"].T * lora["scaling"]


def merge(base, lora):
    """Fold the adapter into the base weight; returns a NEW base pytree."""
    return {**base, "w": base["w"] + lora["B"] @ lora["A"] * lora["scaling"]}


def unmerge(base, lora):
    return {**base, "w": base["w"] - lora["B"] @ lora["A"] * lora["scaling"]}


def base_apply(base, x_BI):
    y_BO = x_BI @ base["w"].T
    if base.get("b") is not None:
        y_BO = y_BO + base["b"]
    return y_BO
