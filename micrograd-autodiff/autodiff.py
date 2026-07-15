"""A scalar reverse-mode autodiff engine (micrograd-style): a Value node
records its inputs and a local backward rule; backward() runs the chain rule
over a topological sort of the graph.

The "build autograd from scratch" question — asked to check you understand
what loss.backward() actually does.
"""

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ----- primitive ops: each defines its own local gradient rule -----

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # += not =: a node used twice must ACCUMULATE gradient from
            # both paths (the single most classic autodiff bug).
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only scalar exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad     # d/dx e^x = e^x (reuse out)
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    # ----- the engine -----

    def backward(self):
        """Reverse-mode: topologically sort the graph, seed d(out)/d(out)=1,
        then run each node's local rule in REVERSE topological order so every
        node's grad is complete before it propagates to its children."""
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad_graph(self):
        stack, seen = [self], set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            v.grad = 0.0
            stack.extend(v._prev)

    # ----- sugar -----

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return Value(other) + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    def __rtruediv__(self, other): return Value(other) * self ** -1
    def __repr__(self): return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"
