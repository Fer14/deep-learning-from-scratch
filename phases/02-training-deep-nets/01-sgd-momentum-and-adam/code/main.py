from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

CURVATURES = np.array([1.0, 100.0])
START = np.array([-4.0, 1.5])


def quadratic_loss(point):
    return 0.5 * np.sum(CURVATURES * point**2)


def quadratic_grad(point):
    return CURVATURES * point


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for name in params:
            params[name] -= self.lr * grads[name]


class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = {}

    def step(self, params, grads):
        for name in params:
            velocity = self.beta * self.velocity.get(name, np.zeros_like(params[name])) + grads[name]
            self.velocity[name] = velocity
            params[name] -= self.lr * velocity


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.first_moment = {}
        self.second_moment = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name in params:
            grad = grads[name]
            m = self.beta1 * self.first_moment.get(name, np.zeros_like(grad)) + (1 - self.beta1) * grad
            v = self.beta2 * self.second_moment.get(name, np.zeros_like(grad)) + (1 - self.beta2) * grad**2
            self.first_moment[name] = m
            self.second_moment[name] = v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def run_on_quadratic(optimizer, steps):
    params = {"point": START.copy()}
    trajectory = [params["point"].copy()]
    for _ in range(steps):
        optimizer.step(params, {"point": quadratic_grad(params["point"])})
        trajectory.append(params["point"].copy())
    return np.array(trajectory)


def init_mlp(sizes, rng):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
        params[f"b{i}"] = np.zeros(fan_out)
    return params


def forward(params, X):
    n_layers = len(params) // 2
    activations = [X]
    hidden = X
    for i in range(n_layers):
        z = hidden @ params[f"W{i}"] + params[f"b{i}"]
        hidden = np.maximum(z, 0.0) if i < n_layers - 1 else z
        activations.append(hidden)
    return activations


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def loss_and_grads(params, X, y):
    n_layers = len(params) // 2
    activations = forward(params, X)
    probs = softmax(activations[-1])
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    grads = {}
    for i in reversed(range(n_layers)):
        grads[f"W{i}"] = activations[i].T @ grad
        grads[f"b{i}"] = grad.sum(axis=0)
        if i > 0:
            grad = (grad @ params[f"W{i}"].T) * (activations[i] > 0)
    return loss, grads


def accuracy(params, X, y):
    return np.mean(forward(params, X)[-1].argmax(axis=1) == y)


def train_mlp(optimizer, X, y, epochs, batch_size, seed):
    rng = np.random.default_rng(seed)
    params = init_mlp([2, 32, 32, 2], rng)
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            _, grads = loss_and_grads(params, X[batch], y[batch])
            optimizer.step(params, grads)
        history.append(loss_and_grads(params, X, y)[0])
    return params, history


def run_torch_on_quadratic(make_optimizer, steps):
    point = torch.tensor(START, dtype=torch.float64, requires_grad=True)
    curvatures = torch.tensor(CURVATURES, dtype=torch.float64)
    optimizer = make_optimizer([point])
    for _ in range(steps):
        optimizer.zero_grad()
        loss = 0.5 * (curvatures * point**2).sum()
        loss.backward()
        optimizer.step()
    return point.detach().numpy()


def plot_trajectories(trajectories, path):
    xs, ys = np.meshgrid(np.linspace(-4.5, 4.5, 200), np.linspace(-2, 2, 200))
    zs = 0.5 * (CURVATURES[0] * xs**2 + CURVATURES[1] * ys**2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.contour(xs, ys, zs, levels=20, cmap="Greys", linewidths=0.6)
    for name, trajectory in trajectories.items():
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker=".", markersize=3, label=name)
    ax.scatter([0], [0], color="black", marker="*", s=120)
    ax.set_title("Optimizer paths on f(x, y) = 0.5 (x^2 + 100 y^2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    print("=== Step 1: an ill-conditioned bowl, f(x, y) = 0.5 * (x^2 + 100 y^2) ===")
    print(f"  start = {START}, loss = {quadratic_loss(START):.3f}, stable SGD needs lr < 2/100 = 0.02")
    optimizers = {
        "sgd lr=0.019": SGD(lr=0.019),
        "momentum lr=0.01 beta=0.9": Momentum(lr=0.01, beta=0.9),
        "adam lr=0.1": Adam(lr=0.1),
    }
    trajectories = {}
    for name, optimizer in optimizers.items():
        trajectory = run_on_quadratic(optimizer, steps=100)
        trajectories[name] = trajectory
        losses = [quadratic_loss(trajectory[k]) for k in (10, 25, 50, 100)]
        print(f"  {name:27s} | loss @10 {losses[0]:.2e} @25 {losses[1]:.2e} @50 {losses[2]:.2e} @100 {losses[3]:.2e}")
    print("  sgd lr=0.021 (too big):", end=" ")
    diverged = run_on_quadratic(SGD(lr=0.021), steps=50)
    print(f"y after 50 steps = {diverged[-1, 1]:.1f}  (oscillates and blows up along the steep axis)")

    print("\n=== Step 2: train a 2-32-32-2 MLP on two moons ===")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
    settings = {
        "sgd lr=0.1": SGD(lr=0.1),
        "momentum lr=0.05 beta=0.9": Momentum(lr=0.05, beta=0.9),
        "adam lr=0.01": Adam(lr=0.01),
    }
    for name, optimizer in settings.items():
        params, history = train_mlp(optimizer, X, y, epochs=30, batch_size=32, seed=0)
        curve = " ".join(f"{history[e]:.3f}" for e in (0, 4, 9, 19, 29))
        print(f"  {name:27s} | loss at epochs 1/5/10/20/30: {curve} | acc {accuracy(params, X, y):.3f}")

    print("\n=== Step 3: Adam's bias correction matters in the first steps ===")
    for t in (1, 2, 5, 10, 100):
        print(f"  t={t:3d}: raw m is scaled by 1-0.9^t = {1 - 0.9**t:.3f}, raw v by 1-0.999^t = {1 - 0.999**t:.4f}")

    print("\n=== Use It: torch.optim on the same bowl ===")
    pairs = [
        ("SGD", lambda: SGD(lr=0.019), lambda p: torch.optim.SGD(p, lr=0.019)),
        ("SGD momentum", lambda: Momentum(lr=0.01, beta=0.9), lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
        ("Adam", lambda: Adam(lr=0.1), lambda p: torch.optim.Adam(p, lr=0.1)),
    ]
    for name, make_ours, make_torch in pairs:
        ours = run_on_quadratic(make_ours(), steps=60)[-1]
        theirs = run_torch_on_quadratic(make_torch, steps=60)
        print(f"  {name:13s} | ours {np.round(ours, 6)} | torch {np.round(theirs, 6)} | max diff {np.abs(ours - theirs).max():.1e}")

    out_path = Path(__file__).parent / "optimizer_paths.png"
    plot_trajectories(trajectories, out_path)
    print(f"\nSaved trajectory plot to {out_path.name}")


if __name__ == "__main__":
    main()
