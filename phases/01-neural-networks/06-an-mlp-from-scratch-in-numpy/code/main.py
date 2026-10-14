from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class Linear:
    def __init__(self, fan_in, fan_out, rng):
        self.W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
        self.b = np.zeros(fan_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

    def params_and_grads(self):
        return []


class SoftmaxCrossEntropy:
    def forward(self, logits, labels):
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self.probs = np.exp(log_probs)
        self.labels = labels
        return -np.mean(log_probs[np.arange(len(labels)), labels])

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.labels)), self.labels] -= 1
        return grad / len(self.labels)


class MLP:
    def __init__(self, layer_sizes, seed):
        rng = np.random.default_rng(seed)
        self.layers = []
        for index, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers.append(Linear(fan_in, fan_out, rng))
            if index < len(layer_sizes) - 2:
                self.layers.append(ReLU())
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, learning_rate):
        for layer in self.layers:
            for param, grad in layer.params_and_grads():
                param -= learning_rate * grad

    def predict(self, x):
        return self.forward(x).argmax(axis=1)


def load_data(seed):
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, y_train, y_test


def train_scratch(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate, seed):
    rng = np.random.default_rng(seed)
    model = MLP([64, 64, 32, 10], seed=seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(y_train))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            logits = model.forward(X_train[batch])
            epoch_losses.append(model.loss_fn.forward(logits, y_train[batch]))
            model.backward()
            model.step(learning_rate)
        test_acc = np.mean(model.predict(X_test) == y_test)
        history.append((np.mean(epoch_losses), test_acc))
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:2d} | train loss {history[-1][0]:.4f} | test acc {test_acc:.4f}")
    return model, history


def gradient_check(seed):
    rng = np.random.default_rng(seed)
    model = MLP([5, 6, 3], seed=seed)
    X = rng.normal(size=(4, 5))
    y = np.array([0, 2, 1, 2])
    model.loss_fn.forward(model.forward(X), y)
    model.backward()
    W = model.layers[0].W
    analytic = model.layers[0].dW.copy()
    numeric = np.zeros_like(W)
    eps = 1e-6
    for index in np.ndindex(W.shape):
        original = W[index]
        W[index] = original + eps
        plus = model.loss_fn.forward(model.forward(X), y)
        W[index] = original - eps
        minus = model.loss_fn.forward(model.forward(X), y)
        W[index] = original
        numeric[index] = (plus - minus) / (2 * eps)
    return np.max(np.abs(analytic - numeric))


def train_torch(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate, seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(yt), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = torch.nn.functional.cross_entropy(model(Xt[batch]), yt[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
    return np.mean(predictions == y_test)


def main():
    X_train, X_test, y_train, y_test = load_data(seed=0)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, classes: {np.unique(y_train).size}")

    print("\n=== Step 1: gradient check of the layer stack ===")
    print(f"  max |analytic - numeric| for first-layer weights: {gradient_check(seed=0):.2e}")

    print("\n=== Step 2: train a 64-64-32-10 MLP with minibatch SGD ===")
    model, history = train_scratch(X_train, y_train, X_test, y_test, epochs=30, batch_size=32, learning_rate=0.05, seed=0)

    print("\n=== Step 3: per-class test accuracy ===")
    predictions = model.predict(X_test)
    per_class = [np.mean(predictions[y_test == c] == c) for c in range(10)]
    print("  " + " ".join(f"{c}:{acc:.2f}" for c, acc in enumerate(per_class)))

    plot_path = Path(__file__).parent / "training_curve.png"
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot([h[0] for h in history], label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot([h[1] for h in history], color="tab:orange", label="test accuracy")
    ax2.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"\n  saved {plot_path.name}")

    print("\n=== Use It: the same MLP in torch ===")
    torch_acc = train_torch(X_train, y_train, X_test, y_test, epochs=30, batch_size=32, learning_rate=0.05, seed=0)
    print(f"  scratch test accuracy = {history[-1][1]:.4f}")
    print(f"  torch   test accuracy = {torch_acc:.4f}")


if __name__ == "__main__":
    main()
