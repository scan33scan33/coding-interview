# MLP Implementation Fixes

## Problem Description
This is a from-scratch implementation of a simple Multilayer Perceptron (MLP) neural network with:
- Custom linear layers with weight and bias parameters
- ReLU activation functions between hidden layers
- MSE loss for regression
- Backpropagation algorithm for gradient computation
- Stochastic gradient descent for weight updates

The implementation trains on a simple 4-sample binary classification task, learning to separate data points using learned non-linear decision boundaries.

## Summary
Fixed critical backpropagation bug in ReLU gradient computation and improved overall implementation quality.

## Critical Bug Fix: ReLU Gradient Masking

### The Problem
The backward pass was incorrectly computing gradients through ReLU activations:
```python
# WRONG - scales gradients by activation values
grad = self.hidden_layers[i].backward(grad) * self.hidden_activations[i]
```

This caused incorrect gradient flow because ReLU's derivative is a binary mask (1 where x > 0, 0 otherwise), not the activation values themselves.

### The Fix (part 1 — derivative shape)
Use the binary ReLU mask, not the activation values:
```python
# ... * (self.hidden_activations[i] > 0)   # mask, not values
```

### The Fix (part 2 — mask ordering)
The first fix still masked the wrong tensor. `hidden_layers[i].backward(grad)`
returns `dL/da_{i-1}` (the gradient for the *previous* layer), and it computes
`grad_w` from the *unmasked* incoming gradient. The ReLU mask belongs to this
layer's pre-activation `z_i`, so it must be applied *before* the linear backward:
```python
# CORRECT
grad = grad * (self.hidden_activations[i] > 0)   # dL/dz_i
grad = self.hidden_layers[i].backward(grad)       # dL/da_{i-1}, grad_w now masked
```
Verified with a finite-difference gradient check: the old order gives
`max|analytic - numeric| ≈ 0.85` on `hidden[0].w`; the corrected order gives `0.0`.

## Additional Improvements

### 1. Weight Initialization
- **Before**: Uniform distribution scaled by `1/sqrt(input_size + output_size)`
- **After**: Normal distribution scaled by He initialization `sqrt(2/input_size)`

He initialization is appropriate for ReLU networks and provides better convergence.

### 2. Bias Initialization
- **Before**: Random values via `np.random.rand`
- **After**: Zero initialization

Biases should start at zero; random initialization can cause unnecessary variance.

### 3. Removed Unused Parameter
- Removed unused `use_relu` parameter from `MLP.__init__`
- Made `hidden_dim` a configurable parameter instead for better flexibility

## Training Results
With correct gradients the loss decreases monotonically on the 4-sample
XOR-like task (e.g. ~2.36 → ~1.20 over 100 iterations at lr=0.001; exact
values depend on the random seed). Correctness is established by the
finite-difference gradient check above, not by the loss value on a toy set —
the earlier reported "~0.86" came from the buggy gradient and is not a valid
target.
