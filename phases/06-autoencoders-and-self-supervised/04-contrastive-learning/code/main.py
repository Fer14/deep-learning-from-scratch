from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

OUT_DIR = Path(__file__).parent
SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def load_data(seed):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(X))
    return (torch.tensor(X[order[300:]]), torch.tensor(X[order[:300]]),
            digits.target[order[300:]], digits.target[order[:300]])


def shift(x, generator):
    images = x.reshape(-1, 1, 8, 8)
    padded = F.pad(images, (1, 1, 1, 1))
    choice = torch.randint(len(SHIFTS), (len(x),), generator=generator)
    shifted = torch.empty_like(images)
    for index, (dy, dx) in enumerate(SHIFTS):
        selected = choice == index
        shifted[selected] = padded[selected, :, 1 + dy:9 + dy, 1 + dx:9 + dx]
    return shifted.reshape(len(x), 64)


def augment(x, generator):
    images = shift(x, generator).reshape(-1, 1, 8, 8)
    contrast = 0.7 + 0.6 * torch.rand(len(x), 1, 1, 1, generator=generator)
    keep = (torch.rand(images.shape, generator=generator) > 0.1).float()
    noise = 0.05 * torch.randn(images.shape, generator=generator)
    return (images * contrast * keep + noise).clamp(0, 1).reshape(len(x), 64)


def init_conv(n_in, n_out, generator):
    weight = torch.randn(n_out, n_in, 3, 3, generator=generator) * np.sqrt(2.0 / (n_in * 9))
    return [weight.requires_grad_(), torch.zeros(n_out).requires_grad_()]


def init_linear(n_in, n_out, generator):
    weight = torch.randn(n_in, n_out, generator=generator) * np.sqrt(2.0 / n_in)
    return [weight.requires_grad_(), torch.zeros(n_out).requires_grad_()]


def mlp(layers, h):
    for index, (weight, bias) in enumerate(layers):
        h = h @ weight + bias
        if index < len(layers) - 1:
            h = torch.relu(h)
    return h


class ContrastiveModel:
    def __init__(self, generator):
        self.encoder = [init_conv(1, 32, generator), init_conv(32, 64, generator), init_conv(64, 128, generator)]
        self.head = [init_linear(128, 128, generator), init_linear(128, 32, generator)]

    def parameters(self):
        return [p for layer in self.encoder + self.head for p in layer]

    def represent(self, x):
        h = x.reshape(-1, 1, 8, 8)
        for index, (weight, bias) in enumerate(self.encoder):
            h = torch.relu(F.conv2d(h, weight, bias, stride=1 if index == 0 else 2, padding=1))
        return h.mean(dim=(2, 3))

    def project(self, x):
        return mlp(self.head, self.represent(x))


def nt_xent(z_a, z_b, temperature):
    z = torch.cat([z_a, z_b], dim=0)
    z = z / z.norm(dim=1, keepdim=True)
    similarity = z @ z.T / temperature
    n = len(z_a)
    similarity = similarity.masked_fill(torch.eye(2 * n, dtype=torch.bool), float("-inf"))
    positive_index = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    positive = similarity[torch.arange(2 * n), positive_index]
    return (torch.logsumexp(similarity, dim=1) - positive).mean()


def train_contrastive(X_train, temperature, epochs, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    model = ContrastiveModel(torch.Generator().manual_seed(seed + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train) - batch_size + 1, batch_size):
            batch = X_train[order[start:start + batch_size]]
            view_a, view_b = augment(batch, generator), augment(batch, generator)
            loss = nt_xent(model.project(view_a), model.project(view_b), temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d} | NT-Xent {loss.item():.3f}")
    return model


def few_label_indices(y, per_class, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y == digit), per_class, replace=False) for digit in range(10)])


def linear_probe(features_train, y_train, features_test, y_test, per_class):
    if per_class is not None:
        index = few_label_indices(y_train, per_class, seed=0)
        features_train, y_train = features_train[index], y_train[index]
    mean, std = features_train.mean(0), features_train.std(0) + 1e-6
    probe = LogisticRegression(max_iter=3000).fit((features_train - mean) / std, y_train)
    return probe.score((features_test - mean) / std, y_test)


def save_embedding_plot(features, labels, path):
    points = PCA(n_components=2).fit_transform(features)
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=6)
    ax.legend(*scatter.legend_elements(), fontsize=6)
    ax.set_title("contrastive representation (PCA to 2D)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = load_data(seed=0)

    print("=== Step 1: two augmented views of the same digit ===")
    generator = torch.Generator().manual_seed(0)
    view_a, view_b = augment(X_train[:200], generator), augment(X_train[:200], generator)
    same = F.cosine_similarity(view_a, view_b).mean()
    different = F.cosine_similarity(view_a, view_b.roll(1, dims=0)).mean()
    print(f"  raw-pixel cosine: same digit two views {same:.3f} | different digits {different:.3f}")

    print("\n=== Step 2: NT-Xent on a toy batch ===")
    aligned = torch.eye(4)
    print(f"  perfect pairs loss {nt_xent(aligned, aligned, 0.5):.3f} | random pairs loss "
          f"{nt_xent(torch.randn(4, 4, generator=generator), torch.randn(4, 4, generator=generator), 0.5):.3f}"
          f" | chance level log(2N-1) = {np.log(7):.3f}")

    print("\n=== Step 3: train SimCLR-style (temperature 0.2, batch 256) ===")
    model = train_contrastive(X_train, temperature=0.2, epochs=80, batch_size=256, seed=0)
    random_model = ContrastiveModel(torch.Generator().manual_seed(1))
    X_test_shifted = shift(X_test, torch.Generator().manual_seed(99))

    print("\n=== Step 4: linear probe (frozen features, logistic regression) ===")
    encoders = {
        "raw pixels": lambda x: x,
        "random conv encoder": random_model.represent,
        "contrastive h": model.represent,
        "contrastive z (after head)": model.project,
    }
    features = {}
    print(f"  {'features':27s} | clean, 5 labels/class | clean, all labels | shifted test, all labels")
    for name, encode in encoders.items():
        with torch.no_grad():
            train_features, test_features, shifted_features = (encode(X).numpy() for X in (X_train, X_test, X_test_shifted))
        features[name] = test_features
        few = linear_probe(train_features, y_train, test_features, y_test, per_class=5)
        full = linear_probe(train_features, y_train, test_features, y_test, per_class=None)
        moved = linear_probe(train_features, y_train, shifted_features, y_test, per_class=None)
        print(f"  {name:27s} | {few:21.3f} | {full:17.3f} | {moved:.3f}")
    save_embedding_plot(features["contrastive h"], y_test, OUT_DIR / "contrastive_embedding.png")
    print("  saved contrastive_embedding.png")

    print("\n=== Use It: NT-Xent as F.cross_entropy ===")
    z_a, z_b = torch.randn(8, 16, generator=generator), torch.randn(8, 16, generator=generator)
    z = F.normalize(torch.cat([z_a, z_b]), dim=1)
    logits = (z @ z.T / 0.2).fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(8, 16), torch.arange(0, 8)])
    print(f"  scratch NT-Xent {nt_xent(z_a, z_b, 0.2):.5f} | F.cross_entropy {F.cross_entropy(logits, targets):.5f}")


if __name__ == "__main__":
    main()
