import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits


def constant(step, total_steps, base_lr):
    return base_lr


def step_decay(step, total_steps, base_lr, drop_every=300, gamma=0.1):
    return base_lr * gamma ** (step // drop_every)


def exponential(step, total_steps, base_lr, gamma=0.996):
    return base_lr * gamma**step


def cosine(step, total_steps, base_lr, min_lr=0.0):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine(step, total_steps, base_lr, warmup_steps=50):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def load_data(seed):
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    order = np.random.default_rng(seed).permutation(len(y))
    X, y = X[order], y[order]
    return X[:1400], y[:1400], X[1400:], y[1400:]


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


def loss_and_grads(params, X, y):
    n_layers = len(params) // 2
    activations = forward(params, X)
    logits = activations[-1] - activations[-1].max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
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


def train(schedule, base_lr, X, y, epochs, batch_size, seed, beta=0.9):
    rng = np.random.default_rng(seed)
    params = init_mlp([64, 128, 128, 10], rng)
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    steps_per_epoch = math.ceil(len(y) / batch_size)
    total_steps = epochs * steps_per_epoch
    step = 0
    lrs, losses = [], []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            loss, grads = loss_and_grads(params, X[batch], y[batch])
            lr = schedule(step, total_steps, base_lr)
            for name in params:
                velocity[name] = beta * velocity[name] + grads[name]
                params[name] -= lr * velocity[name]
            lrs.append(lr)
            losses.append(loss)
            step += 1
    return params, np.array(lrs), np.array(losses)


def lr_range_test(X, y, seed, min_lr=1e-4, max_lr=10.0, steps=200, batch_size=32, beta=0.9):
    rng = np.random.default_rng(seed)
    params = init_mlp([64, 128, 128, 10], rng)
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    lrs = min_lr * (max_lr / min_lr) ** (np.arange(steps) / (steps - 1))
    smoothed, losses = None, []
    for lr in lrs:
        batch = rng.choice(len(y), size=batch_size, replace=False)
        loss, grads = loss_and_grads(params, X[batch], y[batch])
        if not np.isfinite(loss) or loss > 10:
            break
        smoothed = loss if smoothed is None else 0.9 * smoothed + 0.1 * loss
        losses.append(smoothed)
        for name in params:
            velocity[name] = beta * velocity[name] + grads[name]
            params[name] -= lr * velocity[name]
    return lrs[:len(losses)], np.array(losses)


def plot_schedules(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, (lrs, losses) in runs.items():
        axes[0].plot(lrs, label=name)
        window = 45
        axes[1].plot(np.convolve(losses, np.ones(window) / window, mode="valid"), label=name)
    axes[0].set_title("learning rate per step")
    axes[1].set_title("training loss (moving average)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def torch_lrs(make_scheduler, base_lr, total_steps):
    weight = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([weight], lr=base_lr, momentum=0.9)
    scheduler = make_scheduler(optimizer)
    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return np.array(lrs)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    X_train, y_train, X_test, y_test = load_data(seed=0)
    print(f"digits: train {X_train.shape}, test {X_test.shape}")

    print("\n=== Step 1: find a learning rate with a range test ===")
    range_lrs, range_losses = lr_range_test(X_train, y_train, seed=1)
    best = range_lrs[np.argmin(range_losses)]
    print(f"  swept lr from {range_lrs[0]:.0e} up to {range_lrs[-1]:.2f} before the loss blew up")
    print(f"  lowest smoothed loss at lr = {best:.3f}; pick roughly a tenth to a half of that")
    for lr, loss in zip(range_lrs[::25], range_losses[::25]):
        print(f"    lr {lr:8.4f} | smoothed loss {loss:.3f}")

    print("\n=== Step 2: train the same network with five schedules, base lr 0.15 ===")
    base_lr = 0.15
    schedules = {
        "constant": constant,
        "step decay": step_decay,
        "exponential": exponential,
        "cosine": cosine,
        "warmup + cosine": warmup_cosine,
    }
    runs = {}
    for name, schedule in schedules.items():
        params, lrs, losses = train(schedule, base_lr, X_train, y_train, epochs=20, batch_size=32, seed=0)
        runs[name] = (lrs, losses)
        train_loss = loss_and_grads(params, X_train, y_train)[0]
        test_loss = loss_and_grads(params, X_test, y_test)[0]
        print(
            f"  {name:16s} | lr {lrs[0]:.4f} -> {lrs[-1]:.5f} | train loss {train_loss:.4f}"
            f" | test loss {test_loss:.4f} | test acc {accuracy(params, X_test, y_test):.3f}"
        )

    print("\n=== Step 3: the bigger the learning rate, the more decay matters ===")
    for lr in (0.05, 0.15, 0.3):
        results = []
        for name in ("constant", "cosine"):
            params, _, _ = train(schedules[name], lr, X_train, y_train, epochs=10, batch_size=32, seed=0)
            results.append(f"{name} train loss {loss_and_grads(params, X_train, y_train)[0]:.4f}")
        print(f"  base lr {lr:.2f} | " + " | ".join(results))

    print("\n=== Use It: torch.optim.lr_scheduler ===")
    total = 900
    comparisons = {
        "StepLR": (
            lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.1),
            step_decay,
        ),
        "ExponentialLR": (
            lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.996),
            exponential,
        ),
        "CosineAnnealingLR": (
            lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total),
            cosine,
        ),
        "LambdaLR(warmup+cos)": (
            lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total, 1.0)),
            warmup_cosine,
        ),
    }
    for name, (make_scheduler, ours) in comparisons.items():
        theirs = torch_lrs(make_scheduler, base_lr, total)
        mine = np.array([ours(s, total, base_lr) for s in range(total)])
        print(f"  {name:21s} | max |ours - torch| over {total} steps = {np.abs(mine - theirs).max():.1e}")

    out_path = Path(__file__).parent / "schedules.png"
    plot_schedules(runs, out_path)
    print(f"\nSaved schedule plot to {out_path.name}")


if __name__ == "__main__":
    main()
