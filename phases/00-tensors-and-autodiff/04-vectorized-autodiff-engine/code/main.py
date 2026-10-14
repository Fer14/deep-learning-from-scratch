import time

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, parents=(), op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = parents
        self.op = op
        self.backward_step = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self.op}')"

    @staticmethod
    def lift(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = Tensor.lift(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def backward_step():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out.backward_step = backward_step
        return out

    def __mul__(self, other):
        other = Tensor.lift(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def backward_step():
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out.backward_step = backward_step
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def backward_step():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out.backward_step = backward_step
        return out

    def __pow__(self, exponent):
        out = Tensor(self.data ** exponent, (self,), f"pow{exponent}")

        def backward_step():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out.backward_step = backward_step
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def backward_step():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.shape)

        out.backward_step = backward_step
        return out

    def mean(self, axis=None, keepdims=False):
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def backward_step():
            self.grad += out.data * out.grad

        out.backward_step = backward_step
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def backward_step():
            self.grad += out.grad / self.data

        out.backward_step = backward_step
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def backward_step():
            self.grad += (self.data > 0) * out.grad

        out.backward_step = backward_step
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), "tanh")

        def backward_step():
            self.grad += (1.0 - out.data ** 2) * out.grad

        out.backward_step = backward_step
        return out

    def max_constant(self, axis):
        return Tensor(self.data.max(axis=axis, keepdims=True))

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor.lift(other))

    def __truediv__(self, other):
        return self * Tensor.lift(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def topological_order(self):
        order, visited, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node.parents if id(p) not in visited)
        return order

    def backward(self):
        order = self.topological_order()
        for node in order:
            node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node.backward_step()


def log_softmax(logits):
    shifted = logits - logits.max_constant(axis=1)
    return shifted - shifted.exp().sum(axis=1, keepdims=True).log()


def cross_entropy(logits, targets_one_hot):
    return -(log_softmax(logits) * targets_one_hot).sum() * (1.0 / logits.shape[0])


def numerical_grad(f, array, eps=1e-6):
    grad = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + eps
        plus = f()
        array[index] = original - eps
        minus = f()
        array[index] = original
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def gradient_check(rng):
    a = Tensor(rng.normal(size=(4, 3)))
    b = Tensor(rng.normal(size=(3,)))
    W = Tensor(rng.normal(size=(3, 5)))
    targets = np.eye(5)[rng.integers(0, 5, size=4)]

    def build():
        hidden = ((a * b + 1.0).tanh() @ W).relu() + (a ** 2).mean(axis=1, keepdims=True)
        return cross_entropy(hidden, targets) + (W / 2.0).exp().mean()

    out = build()
    out.backward()
    for name, tensor in [("a", a), ("b", b), ("W", W)]:
        numeric = numerical_grad(lambda: build().data.item(), tensor.data)
        error = np.abs(numeric - tensor.grad).max()
        print(f"  d/d{name} shape {str(tensor.shape):<7} max abs error vs finite differences = {error:.2e}")


class Linear:
    def __init__(self, n_in, n_out, rng):
        self.W = Tensor(rng.normal(size=(n_in, n_out)) * np.sqrt(2.0 / n_in))
        self.b = Tensor(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    def __init__(self, sizes, rng):
        self.layers = [Linear(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def accuracy(model, X, y):
    return float((model(Tensor(X)).data.argmax(axis=1) == y).mean())


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=== Step 1: broadcasting-aware add and mul ===")
    x = Tensor(rng.normal(size=(5, 3)))
    bias = Tensor(np.zeros((3,)))
    (x + bias).sum().backward()
    print(f"  x {x.shape} + bias {bias.shape} -> bias.grad {bias.grad} (each entry used 5 times)")

    print("\n=== Step 2: matmul backward ===")
    A = Tensor(rng.normal(size=(2, 3)))
    B = Tensor(rng.normal(size=(3, 4)))
    (A @ B).sum().backward()
    print(f"  A.grad shape {A.grad.shape} == A shape {A.shape}; B.grad shape {B.grad.shape} == B shape {B.shape}")
    print(f"  A.grad equals ones(2,4) @ B.T: {np.allclose(A.grad, np.ones((2, 4)) @ B.data.T)}")

    print("\n=== Step 3: gradient check on a mixed expression ===")
    gradient_check(rng)

    print("\n=== Step 4: train an MLP on 8x8 digits ===")
    digits = load_digits()
    X = digits.data / 16.0
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.25, random_state=0, stratify=digits.target)
    model = MLP([64, 64, 10], rng)
    params = model.parameters()
    print(f"  train {X_train.shape}, test {X_test.shape}, parameters {sum(p.data.size for p in params)}")
    batch_size = 64
    learning_rate = 0.1
    start = time.perf_counter()
    for epoch in range(30):
        order = rng.permutation(len(X_train))
        losses = []
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            logits = model(Tensor(X_train[batch]))
            loss = cross_entropy(logits, np.eye(10)[y_train[batch]])
            loss.backward()
            for p in params:
                p.data -= learning_rate * p.grad
            losses.append(loss.data.item())
        if epoch % 5 == 0 or epoch == 29:
            print(f"  epoch {epoch:2d} | loss {np.mean(losses):.4f} | test accuracy {accuracy(model, X_test, y_test):.4f}")
    print(f"  30 epochs took {time.perf_counter() - start:.2f} s, graph per step has {len(loss.topological_order())} nodes")

    print("\n=== Use It: compare gradients with torch.autograd ===")
    X_batch = X_train[:32]
    y_batch = y_train[:32]
    logits = model(Tensor(X_batch))
    loss = cross_entropy(logits, np.eye(10)[y_batch])
    loss.backward()
    torch_params = [torch.tensor(p.data, requires_grad=True) for p in params]
    W1, b1, W2, b2 = torch_params
    torch_logits = torch.relu(torch.from_numpy(X_batch) @ W1 + b1) @ W2 + b2
    torch_loss = torch.nn.functional.cross_entropy(torch_logits, torch.from_numpy(y_batch))
    torch_loss.backward()
    print(f"  loss: engine {loss.data.item():.6f}  torch {torch_loss.item():.6f}")
    for name, mine, theirs in zip(["W1", "b1", "W2", "b2"], params, torch_params):
        print(f"  {name}: max abs grad difference {np.abs(mine.grad - theirs.grad.numpy()).max():.2e}")


if __name__ == "__main__":
    main()
