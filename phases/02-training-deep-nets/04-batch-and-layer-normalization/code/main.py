import numpy as np
import torch
from sklearn.datasets import load_digits


class Linear:
    def __init__(self, fan_in, fan_out, std, rng):
        self.params = {"W": rng.normal(0.0, std, size=(fan_in, fan_out)), "b": np.zeros(fan_out)}
        self.grads = {}

    def forward(self, x, training):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad):
        self.grads["W"] = self.x.T @ grad
        self.grads["b"] = grad.sum(axis=0)
        return grad @ self.params["W"].T


class ReLU:
    params = {}
    grads = {}

    def forward(self, x, training):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


def normalize_forward(x, gamma, beta, axis, eps):
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    return gamma * x_hat + beta, (x_hat, inv_std, mean, var)


def normalize_backward(grad_out, gamma, cache, axis):
    x_hat, inv_std, _, _ = cache
    n = x_hat.shape[axis]
    grad_x_hat = grad_out * gamma
    grad_x = inv_std / n * (
        n * grad_x_hat
        - grad_x_hat.sum(axis=axis, keepdims=True)
        - x_hat * (grad_x_hat * x_hat).sum(axis=axis, keepdims=True)
    )
    return grad_x


class BatchNorm:
    def __init__(self, features, momentum=0.1, eps=1e-5):
        self.params = {"gamma": np.ones(features), "beta": np.zeros(features)}
        self.grads = {}
        self.running_mean = np.zeros(features)
        self.running_var = np.ones(features)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training):
        if not training:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.params["gamma"] * x_hat + self.params["beta"]
        out, self.cache = normalize_forward(x, self.params["gamma"], self.params["beta"], 0, self.eps)
        n = x.shape[0]
        if n == 1:
            return out
        batch_mean, batch_var = self.cache[2][0], self.cache[3][0]
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var * n / (n - 1)
        return out

    def backward(self, grad):
        x_hat = self.cache[0]
        self.grads["gamma"] = (grad * x_hat).sum(axis=0)
        self.grads["beta"] = grad.sum(axis=0)
        return normalize_backward(grad, self.params["gamma"], self.cache, 0)


class LayerNorm:
    def __init__(self, features, eps=1e-5):
        self.params = {"gamma": np.ones(features), "beta": np.zeros(features)}
        self.grads = {}
        self.eps = eps

    def forward(self, x, training):
        out, self.cache = normalize_forward(x, self.params["gamma"], self.params["beta"], 1, self.eps)
        return out

    def backward(self, grad):
        x_hat = self.cache[0]
        self.grads["gamma"] = (grad * x_hat).sum(axis=0)
        self.grads["beta"] = grad.sum(axis=0)
        return normalize_backward(grad, self.params["gamma"], self.cache, 1)


def build_mlp(sizes, norm, std, rng):
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Linear(fan_in, fan_out, std, rng))
        if i < len(sizes) - 2:
            if norm == "batch":
                layers.append(BatchNorm(fan_out))
            elif norm == "layer":
                layers.append(LayerNorm(fan_out))
            layers.append(ReLU())
    return layers


def run_forward(layers, x, training):
    for layer in layers:
        x = layer.forward(x, training)
    return x


def cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    return loss, grad / n


def train(layers, X, y, epochs, lr, seed, batch_size=32, beta=0.9):
    rng = np.random.default_rng(seed)
    velocities = [{name: np.zeros_like(value) for name, value in layer.params.items()} for layer in layers]
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y) - batch_size + 1, batch_size):
            batch = order[start:start + batch_size]
            loss, grad = cross_entropy(run_forward(layers, X[batch], True), y[batch])
            for layer in reversed(layers):
                grad = layer.backward(grad)
            for layer, velocity in zip(layers, velocities):
                for name in layer.params:
                    velocity[name] = beta * velocity[name] + layer.grads[name]
                    layer.params[name] -= lr * velocity[name]
            history.append(loss)
    return history


def evaluate(layers, X, y):
    logits = run_forward(layers, X, False)
    return cross_entropy(logits, y)[0], np.mean(logits.argmax(axis=1) == y)


def numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + eps
        plus = f()
        x[index] = original - eps
        minus = f()
        x[index] = original
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def load_data(seed):
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    order = np.random.default_rng(seed).permutation(len(y))
    X, y = X[order], y[order]
    return X[:1400], y[:1400], X[1400:], y[1400:]


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    print("=== Step 1: normalize a badly scaled batch ===")
    x = rng.normal(loc=[5.0, -3.0, 0.0], scale=[10.0, 0.1, 1.0], size=(64, 3))
    bn = BatchNorm(3)
    out = bn.forward(x, training=True)
    print(f"  input  mean {np.round(x.mean(axis=0), 3)} std {np.round(x.std(axis=0), 3)}")
    print(f"  output mean {np.round(out.mean(axis=0), 3)} std {np.round(out.std(axis=0), 3)}")

    print("\n=== Step 2: gradient check the backward passes ===")
    for name, layer in (("BatchNorm", BatchNorm(4)), ("LayerNorm", LayerNorm(4))):
        layer.params["gamma"] = rng.normal(size=4)
        layer.params["beta"] = rng.normal(size=4)
        x = rng.normal(size=(6, 4))
        upstream = rng.normal(size=(6, 4))
        loss_fn = lambda: np.sum(layer.forward(x, True) * upstream)
        layer.forward(x, True)
        analytic = layer.backward(upstream)
        numeric = numerical_grad(loss_fn, x)
        layer.forward(x, True)
        layer.backward(upstream)
        numeric_gamma = numerical_grad(loss_fn, layer.params["gamma"])
        print(
            f"  {name}: max |dx analytic - numeric| = {np.abs(analytic - numeric).max():.1e}, "
            f"max |dgamma diff| = {np.abs(layer.grads['gamma'] - numeric_gamma).max():.1e}"
        )

    print("\n=== Step 3: train an 8-layer MLP with a poor init (std 0.05) ===")
    X_train, y_train, X_test, y_test = load_data(seed=0)
    sizes = [64] + [128] * 7 + [10]
    for norm in ("none", "batch", "layer"):
        for lr in (0.02, 0.2):
            layers = build_mlp(sizes, norm, 0.05, np.random.default_rng(1))
            with np.errstate(all="ignore"):
                history = train(layers, X_train, y_train, epochs=5, lr=lr, seed=2)
                test_loss, test_acc = evaluate(layers, X_test, y_test)
            print(
                f"  norm={norm:5s} lr={lr:<4} | first-epoch loss {np.mean(history[:43]):.3f}"
                f" | last-epoch loss {np.mean(history[-43:]):.3f} | test acc {test_acc:.3f}"
            )

    print("\n=== Step 4: train mode vs eval mode ===")
    layers = build_mlp(sizes, "batch", 0.05, np.random.default_rng(1))
    train(layers, X_train, y_train, epochs=5, lr=0.2, seed=2)
    print(f"  eval-mode test acc {evaluate(layers, X_test, y_test)[1]:.3f}")
    single = X_test[:1]
    eval_mode_logits = run_forward(layers, single, False)
    train_mode_logits = run_forward(layers, single, True)
    print(f"  true label of one test image: {y_test[0]}")
    print(f"  eval mode on a batch of 1  -> predicted {eval_mode_logits.argmax()} (uses running mean and variance)")
    print(f"  train mode on a batch of 1 -> predicted {train_mode_logits.argmax()} (every feature normalizes to 0, the input is ignored)")
    print(f"  train-mode logits: {np.round(train_mode_logits[0], 2)}")

    print("\n=== Use It: nn.BatchNorm1d and nn.LayerNorm ===")
    x = rng.normal(loc=2.0, scale=3.0, size=(16, 5))
    upstream = rng.normal(size=(16, 5))
    for name, ours, theirs in (
        ("BatchNorm1d", BatchNorm(5), torch.nn.BatchNorm1d(5, dtype=torch.float64)),
        ("LayerNorm", LayerNorm(5), torch.nn.LayerNorm(5, dtype=torch.float64)),
    ):
        out = ours.forward(x, True)
        grad_x = ours.backward(upstream)
        x_torch = torch.tensor(x, requires_grad=True)
        out_torch = theirs(x_torch)
        out_torch.backward(torch.tensor(upstream))
        line = (
            f"  {name:11s} | out diff {np.abs(out - out_torch.detach().numpy()).max():.1e}"
            f" | dx diff {np.abs(grad_x - x_torch.grad.numpy()).max():.1e}"
            f" | dgamma diff {np.abs(ours.grads['gamma'] - theirs.weight.grad.numpy()).max():.1e}"
        )
        if name == "BatchNorm1d":
            line += f" | running_var diff {np.abs(ours.running_var - theirs.running_var.numpy()).max():.1e}"
        print(line)


if __name__ == "__main__":
    main()
