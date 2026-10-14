import warnings

import numpy as np
import torch
from sklearn.datasets import load_digits


def init_mlp(sizes, rng, output_scale=1.0):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = np.sqrt(2.0 / fan_in) * (output_scale if i == len(sizes) - 2 else 1.0)
        params[f"W{i}"] = rng.normal(0.0, scale, size=(fan_in, fan_out))
        params[f"b{i}"] = np.zeros(fan_out)
    return params


def forward(params, X):
    n_layers = len(params) // 2
    pre_activations, activations = [], [X]
    hidden = X
    for i in range(n_layers):
        z = hidden @ params[f"W{i}"] + params[f"b{i}"]
        pre_activations.append(z)
        hidden = np.maximum(z, 0.0) if i < n_layers - 1 else z
        activations.append(hidden)
    return pre_activations, activations


def loss_only(params, X, y):
    logits = forward(params, X)[1][-1]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -np.mean(log_probs[np.arange(len(y)), y])


def loss_and_grads(params, X, y, bug=None):
    n_layers = len(params) // 2
    pre_activations, activations = forward(params, X)
    logits = activations[-1]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-300))
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    grads = {}
    for i in reversed(range(n_layers)):
        grads[f"W{i}"] = activations[i].T @ grad
        grads[f"b{i}"] = grad.sum(axis=0)
        if i > 0:
            grad = grad @ params[f"W{i}"].T
            if bug != "missing_relu_mask":
                grad = grad * (pre_activations[i - 1] > 0)
    if bug == "transposed_bias":
        grads["b0"] = grads["b0"][::-1].copy()
    return loss, grads


def gradient_check(params, X, y, bug=None, eps=1e-5):
    _, analytic = loss_and_grads(params, X, y, bug)
    worst = {}
    for name, value in params.items():
        numeric = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + eps
            plus = loss_only(params, X, y)
            value[index] = original - eps
            minus = loss_only(params, X, y)
            value[index] = original
            numeric[index] = (plus - minus) / (2 * eps)
        worst[name] = np.linalg.norm(analytic[name] - numeric) / max(np.linalg.norm(analytic[name]) + np.linalg.norm(numeric), 1e-12)
    return worst


def sgd_train(params, X, y, steps, lr, batch_size=None, bug=None, seed=0, monitor_every=None):
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(steps):
        batch = rng.choice(len(y), size=batch_size, replace=False) if batch_size else np.arange(len(y))
        with np.errstate(all="ignore"):
            loss, grads = loss_and_grads(params, X[batch], y[batch], bug)
        if not np.isfinite(loss):
            return losses, step
        if monitor_every and step % monitor_every == 0:
            report_health(params, grads, X[batch], lr, step)
        for name in params:
            params[name] -= lr * grads[name]
        losses.append(loss)
    return losses, None


def report_health(params, grads, X, lr, step):
    pre_activations, _ = forward(params, X)
    parts = []
    for i in range(len(params) // 2):
        name = f"W{i}"
        ratio = lr * np.linalg.norm(grads[name]) / (np.linalg.norm(params[name]) + 1e-12)
        dead = np.mean((pre_activations[i] <= 0).all(axis=0)) if i < len(params) // 2 - 1 else 0.0
        parts.append(f"{name}: |grad| {np.linalg.norm(grads[name]):.2e} update/weight {ratio:.1e} dead {dead:.0%}")
    print(f"    step {step:4d} | " + " | ".join(parts))


def load_data(seed):
    digits = load_digits()
    X = (digits.data - digits.data.mean(axis=0)) / (digits.data.std(axis=0) + 1e-8)
    y = digits.target
    order = np.random.default_rng(seed).permutation(len(y))
    X, y = X[order], y[order]
    return X[:1400], y[:1400], X[1400:], y[1400:]


def accuracy(params, X, y):
    return np.mean(forward(params, X)[1][-1].argmax(axis=1) == y)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    X_train, y_train, X_test, y_test = load_data(seed=0)
    sizes = [64, 64, 64, 10]

    print("=== Check 1: the loss at initialization ===")
    print(f"  expected for 10 balanced classes: -log(1/10) = {np.log(10):.3f}")
    for output_scale in (1.0, 0.01, 10.0):
        params = init_mlp(sizes, np.random.default_rng(0), output_scale)
        print(f"  last-layer init scale x{output_scale:<5} | initial loss {loss_only(params, X_train, y_train):.3f}")

    print("\n=== Check 2: gradient check against finite differences ===")
    rng = np.random.default_rng(1)
    tiny_params = init_mlp([5, 6, 6, 3], rng)
    for name in ("b0", "b1", "b2"):
        tiny_params[name] = rng.normal(0.0, 0.5, size=tiny_params[name].shape)
    tiny_X = rng.normal(size=(8, 5))
    tiny_y = rng.integers(0, 3, size=8)
    for bug in (None, "missing_relu_mask", "transposed_bias"):
        worst = gradient_check(tiny_params, tiny_X, tiny_y, bug)
        verdict = "OK" if max(worst.values()) < 1e-6 else "BUG"
        summary = " ".join(f"{name} {value:.0e}" for name, value in worst.items())
        print(f"  backward={str(bug):18s} | relative error per param: {summary} -> {verdict}")

    print("\n=== Check 3: overfit a single batch of 20 examples ===")
    for bug in (None, "missing_relu_mask"):
        params = init_mlp(sizes, np.random.default_rng(0))
        losses, _ = sgd_train(params, X_train[:20], y_train[:20], steps=300, lr=0.1, bug=bug)
        print(
            f"  backward={str(bug):18s} | loss {losses[0]:.3f} -> {losses[-1]:.4f}"
            f" | batch acc {accuracy(params, X_train[:20], y_train[:20]):.2f}"
        )

    print("\n=== Check 4: a learning rate that is too high ===")
    for lr in (0.1, 2.0, 3.0):
        params = init_mlp(sizes, np.random.default_rng(0))
        losses, nan_step = sgd_train(params, X_train, y_train, steps=200, lr=lr, batch_size=64)
        status = f"loss became NaN/inf at step {nan_step}" if nan_step is not None else f"final loss {np.mean(losses[-20:]):.3f}"
        print(f"  lr {lr:<4} | max loss seen {max(losses):10.3f} | {status}")

    print("\n=== Check 5: monitor gradient norms, update/weight ratios and dead units ===")
    params = init_mlp(sizes, np.random.default_rng(0))
    sgd_train(params, X_train, y_train, steps=301, lr=0.1, batch_size=64, monitor_every=100)
    print(f"  healthy run test acc {accuracy(params, X_test, y_test):.3f}")
    params = init_mlp(sizes, np.random.default_rng(0))
    params["b0"] -= 3.0
    print("  same net with first-layer biases set to -3:")
    sgd_train(params, X_train, y_train, steps=301, lr=0.1, batch_size=64, monitor_every=150)
    print(f"  test acc {accuracy(params, X_test, y_test):.3f}")

    print("\n=== Check 6: inputs and labels out of sync ===")
    shuffled_labels = np.random.default_rng(3).permutation(y_train)
    for name, labels in (("aligned labels", y_train), ("labels shuffled separately", shuffled_labels)):
        params = init_mlp(sizes, np.random.default_rng(0))
        sgd_train(params, X_train, labels, steps=1500, lr=0.1, batch_size=64)
        print(f"  {name:27s} | train acc {accuracy(params, X_train, labels):.3f} | test acc {accuracy(params, X_test, y_test):.3f}")

    print("\n=== Use It: PyTorch debugging tools ===")
    def scaled_softplus(x, w):
        return torch.log1p(torch.exp(x @ w)) * 2.0
    inputs = (torch.randn(4, 3, dtype=torch.float64, requires_grad=True), torch.randn(3, 2, dtype=torch.float64, requires_grad=True))
    print(f"  torch.autograd.gradcheck(scaled_softplus) -> {torch.autograd.gradcheck(scaled_softplus, inputs)}")

    x = torch.tensor([4.0, -1.0], requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                torch.sqrt(x).sum().backward()
        except RuntimeError as error:
            print(f"  detect_anomaly caught: {str(error).splitlines()[0][:90]}")

    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        parameter.register_hook(lambda grad, name=name: gradient_norms.__setitem__(name, grad.norm().item()))
    loss = torch.nn.functional.cross_entropy(model(torch.tensor(X_train[:64], dtype=torch.float32)), torch.tensor(y_train[:64]))
    loss.backward()
    print(f"  initial loss {loss.item():.3f} (expect ~{np.log(10):.3f})")
    print("  gradient norms from hooks: " + ", ".join(f"{k} {v:.3f}" for k, v in gradient_norms.items()))
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    clipped = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters()))
    print(f"  clip_grad_norm_: total norm before {total:.3f}, after {clipped:.3f}")


if __name__ == "__main__":
    main()
