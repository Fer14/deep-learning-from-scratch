from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn

OUT_DIR = Path(__file__).parent


def load_data(seed):
    X = (load_digits().data / 16.0).astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(X))
    return torch.tensor(X[order[300:]]), torch.tensor(X[order[:300]])


def init_linear(n_in, n_out, generator):
    weight = torch.randn(n_in, n_out, generator=generator) * np.sqrt(2.0 / n_in)
    return [weight.requires_grad_(), torch.zeros(n_out).requires_grad_()]


class ScratchMLP:
    def __init__(self, sizes, generator, output="sigmoid"):
        self.layers = [init_linear(a, b, generator) for a, b in zip(sizes[:-1], sizes[1:])]
        self.output = output

    def parameters(self):
        return [p for layer in self.layers for p in layer]

    def __call__(self, h):
        for index, (weight, bias) in enumerate(self.layers):
            h = h @ weight + bias
            if index < len(self.layers) - 1:
                h = torch.relu(h)
        return torch.sigmoid(h) if self.output == "sigmoid" else h


def gaussian_noise(x, sigma, generator):
    return (x + sigma * torch.randn(x.shape, generator=generator)).clamp(0, 1)


def masking_noise(x, drop_prob, generator):
    keep = (torch.rand(x.shape, generator=generator) > drop_prob).float()
    return x * keep


def train(model, X_train, corrupt, epochs, lr, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), batch_size):
            clean = X_train[order[start:start + batch_size]]
            noisy = corrupt(clean, generator)
            loss = ((model(noisy) - clean) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def mse(a, b):
    return float(((a - b) ** 2).mean())


def ring_points(n, generator):
    angle = 2 * np.pi * torch.rand(n, generator=generator)
    radius = 1.0 + 0.05 * torch.randn(n, generator=generator)
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)


def save_grid(rows, path):
    fig, axes = plt.subplots(len(rows), 10, figsize=(10, 1.1 * len(rows)))
    for r, (name, images) in enumerate(rows):
        for c in range(10):
            axes[r, c].imshow(images[c].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_vector_field(model, data, path):
    grid = torch.linspace(-1.8, 1.8, 19)
    gx, gy = torch.meshgrid(grid, grid, indexing="xy")
    points = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    with torch.no_grad():
        arrows = model(points) - points
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data[:, 0], data[:, 1], s=2, color="0.6")
    ax.quiver(points[:, 0], points[:, 1], arrows[:, 0], arrows[:, 1], angles="xy", scale_units="xy", scale=2.5, width=0.004)
    ax.set_aspect("equal")
    ax.set_title("denoiser(x) - x (arrows shrunk 2.5x)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    X_train, X_test = load_data(seed=0)
    test_generator = torch.Generator().manual_seed(123)
    sigma = 0.35
    X_test_noisy = gaussian_noise(X_test, sigma, test_generator)
    print(f"Gaussian noise sigma={sigma}: noisy test input MSE to clean {mse(X_test_noisy, X_test):.4f}")

    print("\n=== Step 1: plain autoencoder vs denoising autoencoder (code 16) ===")
    identity = lambda x, generator: x
    add_noise = lambda x, generator: gaussian_noise(x, sigma, generator)
    plain = train(ScratchMLP([64, 128, 16, 128, 64], torch.Generator().manual_seed(1)), X_train, identity, 120, 3e-3, 128, 0)
    denoiser = train(ScratchMLP([64, 128, 16, 128, 64], torch.Generator().manual_seed(1)), X_train, add_noise, 120, 3e-3, 128, 0)
    with torch.no_grad():
        plain_out, dae_out = plain(X_test_noisy), denoiser(X_test_noisy)
        print(f"  clean input  -> plain AE MSE {mse(plain(X_test), X_test):.4f} | DAE MSE {mse(denoiser(X_test), X_test):.4f}")
        print(f"  noisy input  -> plain AE MSE {mse(plain_out, X_test):.4f} | DAE MSE {mse(dae_out, X_test):.4f}")

    print("\n=== Step 2: an overcomplete code (256 > 64 pixels) ===")
    plain_wide = train(ScratchMLP([64, 256, 64], torch.Generator().manual_seed(2)), X_train, identity, 120, 3e-3, 128, 0)
    dae_wide = train(ScratchMLP([64, 256, 64], torch.Generator().manual_seed(2)), X_train, add_noise, 120, 3e-3, 128, 0)
    with torch.no_grad():
        print(f"  plain AE (wide): clean MSE {mse(plain_wide(X_test), X_test):.4f} | noisy-in MSE {mse(plain_wide(X_test_noisy), X_test):.4f}")
        print(f"  DAE (wide):      clean MSE {mse(dae_wide(X_test), X_test):.4f} | noisy-in MSE {mse(dae_wide(X_test_noisy), X_test):.4f}")

    print("\n=== Step 3: masking noise (30% of pixels set to 0) ===")
    drop = lambda x, generator: masking_noise(x, 0.3, generator)
    X_test_masked = masking_noise(X_test, 0.3, torch.Generator().manual_seed(7))
    dae_mask = train(ScratchMLP([64, 128, 16, 128, 64], torch.Generator().manual_seed(1)), X_train, drop, 120, 3e-3, 128, 0)
    with torch.no_grad():
        print(f"  masked input MSE {mse(X_test_masked, X_test):.4f} | plain AE {mse(plain(X_test_masked), X_test):.4f} | "
              f"masking DAE {mse(dae_mask(X_test_masked), X_test):.4f}")
        save_grid([("clean", X_test[:10].numpy()), ("noisy", X_test_noisy[:10].numpy()),
                   ("plain AE", plain_out[:10].numpy()), ("DAE", dae_out[:10].numpy()),
                   ("masked", X_test_masked[:10].numpy()), ("masking DAE", dae_mask(X_test_masked)[:10].numpy())],
                  OUT_DIR / "denoising.png")

    print("\n=== Step 4: a denoiser on 2D points learns where the data is ===")
    ring_generator = torch.Generator().manual_seed(3)
    ring = ring_points(2000, ring_generator)
    ring_noise = 0.3
    add_ring_noise = lambda x, generator: x + ring_noise * torch.randn(x.shape, generator=generator)
    ring_denoiser = train(ScratchMLP([2, 64, 64, 2], torch.Generator().manual_seed(4), output="linear"),
                          ring, add_ring_noise, 200, 3e-3, 256, 0)
    probes = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.6, 0.0], [0.0, -0.4]])
    with torch.no_grad():
        for point, out in zip(probes, ring_denoiser(probes)):
            print(f"  ({point[0]:+.1f}, {point[1]:+.1f}) |r|={point.norm():.2f} -> ({out[0]:+.2f}, {out[1]:+.2f}) |r|={out.norm():.2f}")
    save_vector_field(ring_denoiser, ring, OUT_DIR / "ring_field.png")
    print("  saved denoising.png and ring_field.png")

    print("\n=== Use It: nn.Dropout as the corruption ===")
    torch.manual_seed(1)
    library_dae = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 16), nn.ReLU(),
                                nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 64), nn.Sigmoid())
    library_dae.train()
    train(library_dae, X_train, identity, 120, 3e-3, 128, 0)
    library_dae.eval()
    with torch.no_grad():
        rescaled = X_test_masked / 0.7
        print(f"  nn.Dropout DAE on masked test input: MSE {mse(library_dae(rescaled), X_test):.4f}")
        print(f"  (dropout multiplies kept pixels by 1/(1-p) during training, so the eval input is rescaled to match)")


if __name__ == "__main__":
    main()
