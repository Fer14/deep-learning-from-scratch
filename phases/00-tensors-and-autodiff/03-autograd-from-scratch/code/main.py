import math
import random
import time

import torch
from sklearn.datasets import make_moons


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.op = op
        self.backward_step = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward_step():
            self.grad += out.grad
            other.grad += out.grad

        out.backward_step = backward_step
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward_step():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward_step = backward_step
        return out

    def __pow__(self, exponent):
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def backward_step():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out.backward_step = backward_step
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def backward_step():
            self.grad += out.data * out.grad

        out.backward_step = backward_step
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def backward_step():
            self.grad += out.grad / self.data

        out.backward_step = backward_step
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def backward_step():
            self.grad += (1.0 - t * t) * out.grad

        out.backward_step = backward_step
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def backward_step():
            self.grad += (self.data > 0) * out.grad

        out.backward_step = backward_step
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    def topological_order(self):
        order = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node.parents:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self):
        order = self.topological_order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node.backward_step()


class Neuron:
    def __init__(self, n_inputs, nonlinear, rng):
        self.weights = [Value(rng.uniform(-2.0, 2.0)) for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.nonlinear = nonlinear

    def __call__(self, inputs):
        activation = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        return activation.tanh() if self.nonlinear else activation

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs, nonlinear, rng):
        self.neurons = [Neuron(n_inputs, nonlinear, rng) for _ in range(n_outputs)]

    def __call__(self, inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, sizes, rng):
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlinear=i < len(sizes) - 2, rng=rng)
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def logistic_loss(score, label):
    return (1.0 + (-label * score).exp()).log()


def example_expression(a, b):
    c = a * b + b ** 3
    d = (c / a).tanh() - (a + 1.0).log()
    return d * d.exp() + (a - b).relu()


def main():
    random.seed(0)
    torch.manual_seed(0)

    print("=== Step 1: a single Value tracks data and grad ===")
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + 10.0
    c.backward()
    print(f"  c = a*b + 10 = {c.data}")
    print(f"  dc/da = {a.grad} (= b)   dc/db = {b.grad} (= a)")

    print("\n=== Step 2: a reused value accumulates gradient ===")
    x = Value(3.0)
    y = x * x + x
    y.backward()
    print(f"  y = x*x + x at x=3 -> dy/dx = {x.grad} (expected 2x + 1 = 7)")

    print("\n=== Step 3: a longer expression versus finite differences ===")
    a = Value(1.5)
    b = Value(0.7)
    out = example_expression(a, b)
    out.backward()
    eps = 1e-6
    numeric_a = (example_expression(Value(1.5 + eps), b).data - example_expression(Value(1.5 - eps), b).data) / (2 * eps)
    numeric_b = (example_expression(a, Value(0.7 + eps)).data - example_expression(a, Value(0.7 - eps)).data) / (2 * eps)
    print(f"  out = {out.data:.6f}, graph has {len(out.topological_order())} nodes")
    print(f"  engine : d/da = {a.grad:.6f}   d/db = {b.grad:.6f}")
    print(f"  numeric: d/da = {numeric_a:.6f}   d/db = {numeric_b:.6f}")

    print("\n=== Step 4: train a tiny MLP on two moons with the scalar engine ===")
    X, y = make_moons(n_samples=80, noise=0.1, random_state=0)
    labels = [1.0 if label == 1 else -1.0 for label in y]
    rng = random.Random(0)
    model = MLP([2, 16, 1], rng)
    params = model.parameters()
    print(f"  model 2-16-1 has {len(params)} scalar parameters")
    learning_rate = 1.0
    start = time.perf_counter()
    for step in range(150):
        scores = [model([Value(x0), Value(x1)]) for x0, x1 in X]
        data_loss = sum(logistic_loss(s, t) for s, t in zip(scores, labels)) * (1.0 / len(labels))
        l2 = sum(p * p for p in params) * 1e-4
        loss = data_loss + l2
        loss.backward()
        for p in params:
            p.data -= learning_rate * p.grad
        accuracy = sum((s.data > 0) == (t > 0) for s, t in zip(scores, labels)) / len(labels)
        if step % 25 == 0 or step == 149:
            print(f"  step {step:3d} | loss {loss.data:.4f} | accuracy {accuracy:.3f}")
    print(f"  training took {time.perf_counter() - start:.1f} s, last graph had {len(loss.topological_order())} nodes")

    print("\n=== Use It: torch.autograd computes the same gradients ===")
    ta = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    tc = ta * tb + tb ** 3
    td = torch.tanh(tc / ta) - torch.log(ta + 1.0)
    tout = td * torch.exp(td) + torch.relu(ta - tb)
    tout.backward()
    print(f"  torch  : out = {tout.item():.6f}   d/da = {ta.grad.item():.6f}   d/db = {tb.grad.item():.6f}")
    print(f"  engine : out = {out.data:.6f}   d/da = {a.grad:.6f}   d/db = {b.grad:.6f}")


if __name__ == "__main__":
    main()
