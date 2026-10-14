from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def leaky_relu(z, slope=0.01):
    return np.where(z > 0, z, slope * z)


def leaky_relu_grad(z, slope=0.01):
    return np.where(z > 0, 1.0, slope)


def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))


def gelu_grad(z, eps=1e-5):
    return (gelu(z + eps) - gelu(z - eps)) / (2 * eps)


def softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "leaky_relu": (leaky_relu, leaky_relu_grad),
    "gelu": (gelu, gelu_grad),
}


def gradient_through_depth(name, depth, width, seed, gain=1.0):
    rng = np.random.default_rng(seed)
    activation, derivative = ACTIVATIONS[name]
    h = rng.normal(size=(64, width))
    weights, pre_activations = [], []
    for _ in range(depth):
        W = rng.normal(0.0, np.sqrt(gain / width), size=(width, width))
        z = h @ W
        weights.append(W)
        pre_activations.append(z)
        h = activation(z)
    grad = np.ones_like(h)
    norms = []
    for W, z in zip(reversed(weights), reversed(pre_activations)):
        grad = (grad * derivative(z)) @ W.T
        norms.append(np.linalg.norm(grad) / np.sqrt(grad.size))
    return norms[::-1]


def plot_activations(path):
    z = np.linspace(-4, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, (fn, grad) in ACTIVATIONS.items():
        axes[0].plot(z, fn(z), label=name)
        axes[1].plot(z, grad(z), label=name)
    axes[0].set_title("activation")
    axes[1].set_title("derivative")
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])

    print("=== Step 1: values and derivatives ===")
    print(f"  z            = {z}")
    for name, (fn, grad) in ACTIVATIONS.items():
        print(f"  {name:10s} f = {np.round(fn(z), 3)}")
        print(f"  {'':10s} f'= {np.round(grad(z), 3)}")

    print("\n=== Step 2: saturation - the largest possible derivative ===")
    grid = np.linspace(-10, 10, 10001)
    for name, (_, grad) in ACTIVATIONS.items():
        print(f"  max {name:10s}' = {grad(grid).max():.3f}")

    print("\n=== Step 3: gradient size after flowing back through 20 layers ===")
    for name in ACTIVATIONS:
        norms = gradient_through_depth(name, depth=20, width=64, seed=0)
        print(f"  {name:10s} layer 20 -> {norms[-1]:.2e}   layer 1 -> {norms[0]:.2e}")
    norms = gradient_through_depth("relu", depth=20, width=64, seed=0, gain=2.0)
    print(f"  relu + He  layer 20 -> {norms[-1]:.2e}   layer 1 -> {norms[0]:.2e}")

    print("\n=== Step 4: softmax turns scores into a distribution ===")
    logits = np.array([2.0, 1.0, 0.1])
    print(f"  softmax({logits}) = {np.round(softmax(logits), 4)}, sum = {softmax(logits).sum():.1f}")
    print(f"  softmax(logits + 1000) = {np.round(softmax(logits + 1000), 4)} (stable thanks to max shift)")

    plot_path = Path(__file__).parent / "activations.png"
    plot_activations(plot_path)
    print(f"\n  saved {plot_path.name}")

    print("\n=== Use It: torch.nn.functional ===")
    zt = torch.tensor(z)
    torch_versions = {
        "sigmoid": torch.sigmoid(zt),
        "tanh": torch.tanh(zt),
        "relu": F.relu(zt),
        "leaky_relu": F.leaky_relu(zt, 0.01),
        "gelu": F.gelu(zt, approximate="tanh"),
    }
    for name, values in torch_versions.items():
        diff = np.max(np.abs(values.numpy() - ACTIVATIONS[name][0](z)))
        print(f"  {name:10s} max |numpy - torch| = {diff:.1e}")
    exact = F.gelu(zt).numpy()
    print(f"  exact (erf) GELU vs tanh approximation: max diff = {np.max(np.abs(exact - gelu(z))):.1e}")
    diff = np.max(np.abs(torch.softmax(torch.tensor(logits), dim=0).numpy() - softmax(logits)))
    print(f"  softmax    max |numpy - torch| = {diff:.1e}")


if __name__ == "__main__":
    main()
