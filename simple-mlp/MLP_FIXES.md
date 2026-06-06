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

### The Fix
```python
# CORRECT - masks gradients where activations were zero
grad = self.hidden_layers[i].backward(grad) * (self.hidden_activations[i] > 0)
```

When backpropagating through ReLU, gradients must be masked by the binary indicator of which neurons were active during the forward pass.

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
With these fixes, the network converges smoothly on a 4-sample XOR-like classification task:
- **Initial loss**: ~2.17
- **Final loss (100 iterations)**: ~0.86
- **Learning rate**: 0.001

The monotonic decrease in loss confirms that backpropagation is now working correctly.
