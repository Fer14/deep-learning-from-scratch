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
from torch import nn

OUT_DIR = Path(__file__).parent


def load_data(seed):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(X))
    return (torch.tensor(X[order[300:]]), torch.tensor(X[order[:300]]),
            torch.tensor(digits.target[order[300:]]), torch.tensor(digits.target[order[:300]]))


def init_linear(n_in, n_out, generator):
    weight = torch.randn(n_in, n_out, generator=generator) * np.sqrt(2.0 / n_in)
    return [weight.requires_grad_(), torch.zeros(n_out).requires_grad_()]


def mlp(layers, h):
    for index, (weight, bias) in enumerate(layers):
        h = h @ weight + bias
        if index < len(layers) - 1:
            h = torch.relu(h)
    return h


def fit(parameters, loss_fn, X, epochs, seed, lr=3e-3, batch_size=128):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for epoch in range(epochs):
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), batch_size):
            index = order[start:start + batch_size]
            loss = loss_fn(index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_autoencoder_encoder(X, code_size, seed):
    generator = torch.Generator().manual_seed(seed)
    encoder = [init_linear(64, 128, generator), init_linear(128, code_size, generator)]
    decoder = [init_linear(code_size, 128, generator), init_linear(128, 64, generator)]
    loss_fn = lambda index: ((torch.sigmoid(mlp(decoder, mlp(encoder, X[index]))) - X[index]) ** 2).mean()
    fit([p for layer in encoder + decoder for p in layer], loss_fn, X, epochs=100, seed=seed)
    return lambda x: mlp(encoder, x)


def train_supervised_encoder(X, y, code_size, n_classes, seed):
    generator = torch.Generator().manual_seed(seed)
    encoder = [init_linear(64, 128, generator), init_linear(128, code_size, generator)]
    classifier = [init_linear(code_size, n_classes, generator)]
    loss_fn = lambda index: F.cross_entropy(mlp(classifier, torch.relu(mlp(encoder, X[index]))), y[index])
    fit([p for layer in encoder + classifier for p in layer], loss_fn, X, epochs=60, seed=seed)
    return lambda x: torch.relu(mlp(encoder, x))


def normalize(features):
    return features / (features.norm(dim=1, keepdim=True) + 1e-8)


def knn_accuracy(train_features, y_train, test_features, y_test, k=5):
    similarity = normalize(test_features) @ normalize(train_features).T
    neighbor_labels = y_train[similarity.topk(k, dim=1).indices]
    votes = F.one_hot(neighbor_labels, num_classes=10).sum(dim=1)
    return float((votes.argmax(dim=1) == y_test).float().mean())


def precision_at_k(features, labels, k=5):
    unit = normalize(features)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(float("-inf"))
    neighbors = similarity.topk(k, dim=1).indices
    return float((labels[neighbors] == labels[:, None]).float().mean())


def linear_probe(train_features, y_train, test_features, y_test):
    mean, std = train_features.mean(0), train_features.std(0) + 1e-6
    probe = LogisticRegression(max_iter=3000).fit(((train_features - mean) / std).numpy(), y_train.numpy())
    return probe.score(((test_features - mean) / std).numpy(), y_test.numpy())


def save_retrieval(X_test, embeddings, query_indices, path):
    rows = []
    for name, features in embeddings.items():
        unit = normalize(features)
        similarity = unit @ unit.T
        similarity.fill_diagonal_(float("-inf"))
        rows.append((name, similarity[query_indices].topk(5, dim=1).indices))
    fig, axes = plt.subplots(len(query_indices), 1 + 5 * len(rows), figsize=(1 + 5 * len(rows), 1.1 * len(query_indices)))
    for r, query in enumerate(query_indices):
        axes[r, 0].imshow(X_test[query].reshape(8, 8), cmap="gray_r")
        for block, (name, neighbors) in enumerate(rows):
            for c in range(5):
                axes[r, 1 + 5 * block + c].imshow(X_test[neighbors[r, c]].reshape(8, 8), cmap="gray_r")
            axes[0, 1 + 5 * block].set_title(name, fontsize=8, loc="left")
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].set_title("query", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = load_data(seed=0)

    print("=== Step 1: an embedding table is a matrix you index ===")
    generator = torch.Generator().manual_seed(0)
    table = torch.randn(10, 4, generator=generator, requires_grad=True)
    ids = torch.tensor([3, 3, 7])
    by_lookup = table[ids]
    by_one_hot = F.one_hot(ids, num_classes=10).float() @ table
    print(f"  lookup == one-hot @ table: {torch.allclose(by_lookup, by_one_hot)}")
    by_lookup.sum().backward()
    print(f"  rows with nonzero gradient: {torch.nonzero(table.grad.abs().sum(dim=1)).flatten().tolist()} "
          f"(row 3 gradient {table.grad[3, 0].item():.0f}, used twice)")

    print("\n=== Step 2: build five representations of the same digits (32 numbers or fewer) ===")
    random_layer = init_linear(64, 32, torch.Generator().manual_seed(5))
    pca = PCA(n_components=32).fit(X_train.numpy())
    autoencoder = train_autoencoder_encoder(X_train, 32, seed=1)
    supervised = train_supervised_encoder(X_train, y_train, 32, 10, seed=2)
    encoders = {
        "raw pixels (64)": lambda x: x,
        "random projection": lambda x: torch.relu(x @ random_layer[0]),
        "PCA": lambda x: torch.tensor(pca.transform(x.numpy())),
        "autoencoder": autoencoder,
        "supervised": supervised,
    }
    with torch.no_grad():
        features = {name: (encode(X_train), encode(X_test)) for name, encode in encoders.items()}
    for name, (train_features, test_features) in features.items():
        print(f"  {name:18s} train {tuple(train_features.shape)} test {tuple(test_features.shape)}")

    print("\n=== Step 3: three ways to score an embedding ===")
    print(f"  {'representation':18s} | kNN acc (cosine) | precision@5 | linear probe")
    for name, (train_features, test_features) in features.items():
        print(f"  {name:18s} | {knn_accuracy(train_features, y_train, test_features, y_test):16.3f} | "
              f"{precision_at_k(test_features, y_test):11.3f} | {linear_probe(train_features, y_train, test_features, y_test):.3f}")

    print("\n=== Step 4: cosine vs Euclidean neighbors ===")
    train_features, test_features = features["autoencoder"]
    scaled_test = test_features * torch.rand(len(test_features), 1, generator=generator).mul(3).add(0.2)
    euclidean_neighbors = torch.cdist(scaled_test, train_features).topk(5, dim=1, largest=False).indices
    euclidean_accuracy = float((F.one_hot(y_train[euclidean_neighbors], 10).sum(1).argmax(1) == y_test).float().mean())
    print(f"  test embeddings rescaled by random factors in [0.2, 3.2]:")
    print(f"  Euclidean kNN {euclidean_accuracy:.3f} | cosine kNN {knn_accuracy(train_features, y_train, scaled_test, y_test):.3f}")

    print("\n=== Step 5: do representations transfer to classes they never saw? ===")
    seen = y_train < 5
    supervised_seen = train_supervised_encoder(X_train[seen], y_train[seen], 32, 5, seed=2)
    unseen_train, unseen_test = y_train >= 5, y_test >= 5
    transfer = {"raw pixels (64)": lambda x: x, "autoencoder (all digits, no labels)": autoencoder,
                "supervised on 0-4 only": supervised_seen}
    with torch.no_grad():
        for name, encode in transfer.items():
            accuracy = knn_accuracy(encode(X_train[unseen_train]), y_train[unseen_train],
                                    encode(X_test[unseen_test]), y_test[unseen_test])
            print(f"  kNN on digits 5-9 with {name:36s}: {accuracy:.3f}")

    query_indices = [0, 1, 2, 3, 4, 5]
    save_retrieval(X_test, {"raw pixels": X_test, "supervised": features["supervised"][1]}, query_indices,
                   OUT_DIR / "retrieval.png")
    print("  saved retrieval.png")

    print("\n=== Use It: nn.Embedding, F.cosine_similarity and topk ===")
    embedding = nn.Embedding(10, 4)
    with torch.no_grad():
        embedding.weight.copy_(table)
    print(f"  nn.Embedding matches the table lookup: {torch.allclose(embedding(ids), by_lookup)}")
    supervised_test = features["supervised"][1]
    scores = F.cosine_similarity(supervised_test[:1], supervised_test, dim=1)
    scores[0] = float("-inf")
    top = scores.topk(5)
    print(f"  query label {y_test[0].item()} -> top-5 labels {y_test[top.indices].tolist()} "
          f"similarities {[round(v, 3) for v in top.values.tolist()]}")


if __name__ == "__main__":
    main()
