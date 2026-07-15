import math
import random

import torch

from autodiff import Value


def test_matches_torch_on_composite_expression():
    def f(a, b):
        c = a * b + b ** 3
        d = (c + a).relu() + c * a
        e = d.tanh() if isinstance(d, Value) else torch.tanh(d)
        return e + d / 2 + a.exp() if isinstance(a, Value) else e + d / 2 + torch.exp(a)

    a, b = Value(1.3), Value(-0.7)
    out = f(a, b)
    out.backward()

    ta = torch.tensor(1.3, requires_grad=True)
    tb = torch.tensor(-0.7, requires_grad=True)
    tout = f(ta, tb)
    tout.backward()

    assert abs(out.data - tout.item()) < 1e-6
    assert abs(a.grad - ta.grad.item()) < 1e-6
    assert abs(b.grad - tb.grad.item()) < 1e-6


def test_gradient_accumulates_on_reused_node():
    # y = x*x + x: dy/dx = 2x + 1. A node used on multiple paths must SUM
    # gradients (writing `grad =` instead of `grad +=` gives 2x or 1).
    x = Value(3.0)
    y = x * x + x
    y.backward()
    assert abs(x.grad - 7.0) < 1e-9


def test_diamond_graph():
    # x -> a, b -> out: both branches contribute through the shared input.
    x = Value(0.5)
    a = x.exp()
    b = x * 2
    out = a * b
    out.backward()
    # d/dx [2x e^x] = 2 e^x (1 + x)
    want = 2 * math.exp(0.5) * 1.5
    assert abs(x.grad - want) < 1e-9


def test_topological_order_deep_chain():
    # 100-deep chain: naive recursion per-node or wrong ordering breaks;
    # d/dx of x + 1 + 1 + ... is exactly 1.
    x = Value(1.0)
    y = x
    for _ in range(100):
        y = y + 1
    y.backward()
    assert x.grad == 1.0


def test_relu_gates_gradient():
    x = Value(-2.0)
    y = x.relu() * 5
    y.backward()
    assert x.grad == 0.0
    x2 = Value(2.0)
    y2 = x2.relu() * 5
    y2.backward()
    assert x2.grad == 5.0


def test_numerical_gradient_check():
    # Finite differences as ground truth, independent of torch.
    random.seed(0)
    for _ in range(20):
        av, bv = random.uniform(-2, 2), random.uniform(0.5, 2)

        def f(a, b):
            return (a * b + a ** 2).tanh() + b / (a * a + 1)

        a, b = Value(av), Value(bv)
        f(a, b).backward()

        h = 1e-6
        num_a = (f(Value(av + h), Value(bv)).data
                 - f(Value(av - h), Value(bv)).data) / (2 * h)
        num_b = (f(Value(av), Value(bv + h)).data
                 - f(Value(av), Value(bv - h)).data) / (2 * h)
        assert abs(a.grad - num_a) < 1e-4
        assert abs(b.grad - num_b) < 1e-4


def test_trains_a_tiny_mlp_on_xor():
    random.seed(42)

    class Neuron:
        def __init__(self, nin):
            self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
            self.b = Value(0.0)

        def __call__(self, x):
            return (sum((wi * xi for wi, xi in zip(self.w, x)), self.b)).tanh()

        def params(self):
            return self.w + [self.b]

    class Layer:
        def __init__(self, nin, nout):
            self.neurons = [Neuron(nin) for _ in range(nout)]

        def __call__(self, x):
            return [n(x) for n in self.neurons]

        def params(self):
            return [p for n in self.neurons for p in n.params()]

    h, out = Layer(2, 4), Layer(4, 1)
    params = h.params() + out.params()
    data = [([0, 0], -1), ([0, 1], 1), ([1, 0], 1), ([1, 1], -1)]

    loss_val = None
    for _ in range(200):
        loss = Value(0.0)
        for x, y in data:
            pred = out(h([Value(v) for v in x]))[0]
            loss = loss + (pred - y) ** 2
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= 0.1 * p.grad
        loss_val = loss.data
    assert loss_val < 0.1                      # XOR learned
