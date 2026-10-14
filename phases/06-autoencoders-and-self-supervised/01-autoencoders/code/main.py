from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from torch import nn

OUT_DIR = Path(__file__).parent


def load_data(seed):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = 300
    test_idx, train_idx = order[:n_test], order[n_test:]
    return (torch.tensor(X[train_idx]), torch.tensor(X[test_idx]),
            digits.target[train_idx], digits.target[test_idx])


def init_linear(n_in, n_out, generator):
    weight = torch.randn(n_in, n_out, generator=generator) * np.sqrt(2.0 / n_in)
    bias = torch.zeros(n_out)
    return [weight.requires_grad_(), bias.requires_grad_()]


class ScratchAutoencoder:
    def __init__(self, sizes, generator, nonlinear=True):
        self.nonlinear = nonlinear
        encoder_sizes = sizes
        decoder_sizes = sizes[::-1]
        self.encoder = [init_linear(a, b, generator) for a, b in zip(encoder_sizes[:-1], encoder_sizes[1:])]
        self.decoder = [init_linear(a, b, generator) for a, b in zip(decoder_sizes[:-1], decoder_sizes[1:])]

    def parameters(self):
        return [p for layer in self.encoder + self.decoder for p in layer]

    def run(self, layers, h, last_activation):
        for index, (weight, bias) in enumerate(layers):
            h = h @ weight + bias
            is_last = index == len(layers) - 1
            if not is_last and self.nonlinear:
                h = torch.relu(h)
            if is_last and last_activation:
                h = torch.sigmoid(h)
        return h

    def encode(self, x):
        return self.run(self.encoder, x, last_activation=False)

    def decode(self, z):
        return self.run(self.decoder, z, last_activation=self.nonlinear)

    def __call__(self, x):
        return self.decode(self.encode(x))


def train(model, X_train, epochs, lr, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), batch_size):
            batch = X_train[order[start:start + batch_size]]
            loss = ((model(batch) - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def reconstruction_mse(model, X):
    with torch.no_grad():
        return float(((model(X) - X) ** 2).mean())


def save_reconstructions(X, recon_by_name, path):
    rows = [("original", X)] + list(recon_by_name.items())
    fig, axes = plt.subplots(len(rows), 10, figsize=(10, 1.1 * len(rows)))
    for row, (name, images) in enumerate(rows):
        for col in range(10):
            axes[row, col].imshow(images[col].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_latent_scatter(Z, labels, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=6)
    ax.legend(*scatter.legend_elements(), fontsize=6, loc="best")
    ax.set_title("2D bottleneck of the nonlinear autoencoder")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = load_data(seed=0)
    print(f"train {tuple(X_train.shape)}  test {tuple(X_test.shape)}  pixel range [{X_train.min():.1f}, {X_train.max():.1f}]")

    print("\n=== Step 1: a linear autoencoder vs PCA ===")
    for code_size in [2, 8]:
        linear = ScratchAutoencoder([64, code_size], torch.Generator().manual_seed(1), nonlinear=False)
        train(linear, X_train, epochs=150, lr=1e-2, batch_size=128, seed=0)
        pca = PCA(n_components=code_size).fit(X_train.numpy())
        pca_recon = pca.inverse_transform(pca.transform(X_test.numpy()))
        pca_mse = float(((pca_recon - X_test.numpy()) ** 2).mean())
        print(f"  code {code_size}: linear AE test MSE {reconstruction_mse(linear, X_test):.4f} | PCA test MSE {pca_mse:.4f}")

    print("\n=== Step 2: a nonlinear autoencoder ===")
    results = {}
    for code_size in [2, 8]:
        model = ScratchAutoencoder([64, 128, 32, code_size], torch.Generator().manual_seed(2))
        train(model, X_train, epochs=150, lr=3e-3, batch_size=128, seed=0)
        results[code_size] = model
        print(f"  code {code_size}: nonlinear AE test MSE {reconstruction_mse(model, X_test):.4f}  "
              f"(compression 64 -> {code_size})")

    print("\n=== Step 3: what the 2D code learned ===")
    with torch.no_grad():
        Z_test = results[2].encode(X_test).numpy()
    centroids = np.stack([Z_test[y_test == digit].mean(axis=0) for digit in range(10)])
    nearest_centroid = np.argmin(((Z_test[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
    print(f"  nearest-centroid accuracy in the 2D code (labels never used in training): {np.mean(nearest_centroid == y_test):.3f}")
    save_latent_scatter(Z_test, y_test, OUT_DIR / "latent_2d.png")

    pca8 = PCA(n_components=8).fit(X_train.numpy())
    with torch.no_grad():
        recon = {
            "AE code 2": results[2](X_test[:10]).numpy(),
            "AE code 8": results[8](X_test[:10]).numpy(),
            "PCA 8": pca8.inverse_transform(pca8.transform(X_test[:10].numpy())),
        }
    save_reconstructions(X_test[:10].numpy(), recon, OUT_DIR / "reconstructions.png")
    print("  saved reconstructions.png and latent_2d.png")

    print("\n=== Use It: nn.Sequential ===")
    torch.manual_seed(2)
    encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 8))
    decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 64), nn.Sigmoid())
    library_model = nn.Sequential(encoder, decoder)
    train(library_model, X_train, epochs=150, lr=3e-3, batch_size=128, seed=0)
    print(f"  nn.Sequential AE (code 8) test MSE {reconstruction_mse(library_model, X_test):.4f}")
    print(f"  parameters: {sum(p.numel() for p in library_model.parameters())}")


if __name__ == "__main__":
    main()
