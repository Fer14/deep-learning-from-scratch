from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons


def dropout_forward(x, p, training, rng):
    if not training or p == 0.0:
        return x, None
    mask = (rng.random(x.shape) >= p) / (1.0 - p)
    return x * mask, mask


class AdamOptimizer:
    def __init__(self, lr, l2=0.0, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.l2 = l2
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name in params:
            decays = name.startswith("W")
            grad = grads[name] + (self.l2 * params[name] if decays else 0.0)
            if decays and self.weight_decay:
                params[name] -= self.lr * self.weight_decay * params[name]
            m = self.beta1 * self.m.get(name, np.zeros_like(grad)) + (1 - self.beta1) * grad
            v = self.beta2 * self.v.get(name, np.zeros_like(grad)) + (1 - self.beta2) * grad**2
            self.m[name], self.v[name] = m, v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def init_mlp(sizes, rng):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
        params[f"b{i}"] = np.zeros(fan_out)
    return params


def forward(params, X, dropout, training, rng):
    n_layers = len(params) // 2
    activations, masks = [X], []
    hidden = X
    for i in range(n_layers):
        z = hidden @ params[f"W{i}"] + params[f"b{i}"]
        if i < n_layers - 1:
            hidden, mask = dropout_forward(np.maximum(z, 0.0), dropout, training, rng)
        else:
            hidden, mask = z, None
        activations.append(hidden)
        masks.append(mask)
    return activations, masks


def softmax_cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    return loss, grad / n


def loss_and_grads(params, X, y, dropout, rng):
    n_layers = len(params) // 2
    activations, masks = forward(params, X, dropout, True, rng)
    loss, grad = softmax_cross_entropy(activations[-1], y)
    grads = {}
    for i in reversed(range(n_layers)):
        grads[f"W{i}"] = activations[i].T @ grad
        grads[f"b{i}"] = grad.sum(axis=0)
        if i > 0:
            grad = grad @ params[f"W{i}"].T
            if masks[i - 1] is not None:
                grad = grad * masks[i - 1]
            grad = grad * (activations[i] > 0)
    return loss, grads


def evaluate(params, X, y):
    logits = forward(params, X, 0.0, False, None)[0][-1]
    loss, _ = softmax_cross_entropy(logits, y)
    return loss, np.mean(logits.argmax(axis=1) == y)


def weight_norm(params):
    return np.sqrt(sum(np.sum(value**2) for name, value in params.items() if name.startswith("W")))


def train(X, y, dropout, optimizer, epochs, seed, batch_size=32):
    rng = np.random.default_rng(seed)
    params = init_mlp([X.shape[1], 256, 256, int(y.max()) + 1], rng)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            _, grads = loss_and_grads(params, X[batch], y[batch], dropout, rng)
            optimizer.step(params, grads)
    return params


def load_data():
    X_train, y_train = make_moons(n_samples=80, noise=0.35, random_state=0)
    X_test, y_test = make_moons(n_samples=2000, noise=0.35, random_state=1)
    return X_train, y_train, X_test, y_test


def plot_boundaries(models, X, y, path):
    xs, ys = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
    grid = np.c_[xs.ravel(), ys.ravel()]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6))
    for ax, (name, params) in zip(axes, models.items()):
        logits = forward(params, grid, 0.0, False, None)[0][-1]
        ax.contourf(xs, ys, logits.argmax(axis=1).reshape(xs.shape), alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="k", linewidths=0.3)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    print("=== Step 1: inverted dropout keeps the expected activation ===")
    x = np.ones((10000, 4))
    dropped, mask = dropout_forward(x, 0.5, True, rng)
    print(f"  p=0.5 | fraction zeroed {np.mean(dropped == 0):.3f} | survivors scaled to {dropped.max():.1f} | mean {dropped.mean():.3f}")
    print(f"  eval mode returns the input unchanged: {np.array_equal(dropout_forward(x, 0.5, False, rng)[0], x)}")

    print("\n=== Step 2: a 2-256-256-2 MLP on 80 noisy moons, 400 epochs of Adam lr 3e-3 ===")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  train {X_train.shape}, test {X_test.shape}")
    configs = {
        "baseline": dict(dropout=0.0, weight_decay=0.0),
        "dropout 0.5": dict(dropout=0.5, weight_decay=0.0),
        "weight decay 3.0 (AdamW)": dict(dropout=0.0, weight_decay=3.0),
        "dropout 0.5 + decay 1.0": dict(dropout=0.5, weight_decay=1.0),
    }
    models = {}
    for name, config in configs.items():
        optimizer = AdamOptimizer(lr=3e-3, weight_decay=config["weight_decay"])
        params = train(X_train, y_train, config["dropout"], optimizer, epochs=400, seed=1)
        models[name] = params
        train_loss, train_acc = evaluate(params, X_train, y_train)
        test_loss, test_acc = evaluate(params, X_test, y_test)
        print(
            f"  {name:25s} | train loss {train_loss:.3f} acc {train_acc:.3f} | test loss {test_loss:.3f}"
            f" acc {test_acc:.3f} | ||W|| {weight_norm(params):6.2f}"
        )

    print("\n=== Step 3: L2 in the gradient vs decoupled weight decay, both with Adam ===")
    for name, optimizer in (
        ("L2 penalty 1e-2 (Adam + L2)", AdamOptimizer(lr=3e-3, l2=1e-2)),
        ("decoupled decay 1e-2 (AdamW)", AdamOptimizer(lr=3e-3, weight_decay=1e-2)),
    ):
        params = train(X_train, y_train, 0.0, optimizer, epochs=400, seed=1)
        test_loss, test_acc = evaluate(params, X_test, y_test)
        print(f"  {name:29s} | test loss {test_loss:.3f} acc {test_acc:.3f} | ||W|| {weight_norm(params):6.2f}")

    out_path = Path(__file__).parent / "boundaries.png"
    plot_boundaries(models, X_train, y_train, out_path)
    print(f"  saved decision boundaries to {out_path.name}")

    print("\n=== Use It: nn.Dropout, torch.optim.Adam(weight_decay) and AdamW ===")
    layer = torch.nn.Dropout(p=0.5)
    ones = torch.ones(10000)
    layer.train()
    print(f"  nn.Dropout train mode: fraction zeroed {(layer(ones) == 0).float().mean():.3f}, max {layer(ones).max():.1f}")
    layer.eval()
    print(f"  nn.Dropout eval mode: identity = {torch.equal(layer(ones), ones)}")

    weights = rng.normal(size=(5, 3))
    grads_sequence = [rng.normal(size=(5, 3)) for _ in range(20)]
    for label, torch_cls, ours_kwargs in (
        ("Adam(weight_decay=0.1) == L2", torch.optim.Adam, dict(l2=0.1)),
        ("AdamW(weight_decay=0.1)    ", torch.optim.AdamW, dict(weight_decay=0.1)),
    ):
        ours = {"W": weights.copy()}
        optimizer = AdamOptimizer(lr=0.01, **ours_kwargs)
        theirs = torch.tensor(weights.copy(), requires_grad=True)
        torch_optimizer = torch_cls([theirs], lr=0.01, weight_decay=0.1)
        for grad in grads_sequence:
            optimizer.step(ours, {"W": grad})
            theirs.grad = torch.tensor(grad)
            torch_optimizer.step()
        print(f"  {label} | max diff after 20 steps {np.abs(ours['W'] - theirs.detach().numpy()).max():.1e}")


if __name__ == "__main__":
    main()
