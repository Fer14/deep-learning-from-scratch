import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from torch import nn

OUT_DIR = Path(__file__).parent
PATCH = 2
GRID = 8 // PATCH
N_PATCHES = GRID * GRID
PATCH_DIM = PATCH * PATCH


def load_data(seed):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(X))
    return (torch.tensor(X[order[300:]]), torch.tensor(X[order[:300]]),
            digits.target[order[300:]], digits.target[order[:300]])


def patchify(x):
    images = x.reshape(-1, GRID, PATCH, GRID, PATCH)
    return images.permute(0, 1, 3, 2, 4).reshape(-1, N_PATCHES, PATCH_DIM)


def unpatchify(patches):
    images = patches.reshape(-1, GRID, GRID, PATCH, PATCH)
    return images.permute(0, 1, 3, 2, 4).reshape(-1, 64)


def random_masking(n, mask_ratio, generator):
    n_keep = int(round(N_PATCHES * (1 - mask_ratio)))
    noise = torch.rand(n, N_PATCHES, generator=generator)
    shuffle = noise.argsort(dim=1)
    restore = shuffle.argsort(dim=1)
    keep_index = shuffle[:, :n_keep]
    mask = torch.ones(n, N_PATCHES)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, restore)
    return keep_index, restore, mask


def gather_tokens(tokens, index):
    return torch.gather(tokens, 1, index[:, :, None].expand(-1, -1, tokens.shape[-1]))


class MaskedAutoencoder(nn.Module):
    def __init__(self, dim=64, decoder_dim=32, depth=2):
        super().__init__()
        self.patch_embed = nn.Parameter(torch.randn(PATCH_DIM, dim) * 0.3)
        self.patch_bias = nn.Parameter(torch.zeros(dim))
        self.position = nn.Parameter(torch.randn(1, N_PATCHES, dim) * 0.02)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, 2 * dim, dropout=0.0, batch_first=True,
                                                                 norm_first=True) for _ in range(depth)])
        self.encoder_norm = nn.LayerNorm(dim)
        self.to_decoder = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_position = nn.Parameter(torch.randn(1, N_PATCHES, decoder_dim) * 0.02)
        self.decoder = nn.TransformerEncoderLayer(decoder_dim, 4, 2 * decoder_dim, dropout=0.0, batch_first=True,
                                                  norm_first=True)
        self.to_pixels = nn.Linear(decoder_dim, PATCH_DIM)

    def encode(self, patches, keep_index=None):
        tokens = patches @ self.patch_embed + self.patch_bias + self.position
        if keep_index is not None:
            tokens = gather_tokens(tokens, keep_index)
        for layer in self.encoder:
            tokens = layer(tokens)
        return self.encoder_norm(tokens)

    def decode(self, visible, restore):
        visible = self.to_decoder(visible)
        n_masked = N_PATCHES - visible.shape[1]
        full = torch.cat([visible, self.mask_token.expand(len(visible), n_masked, -1)], dim=1)
        full = gather_tokens(full, restore) + self.decoder_position
        return self.to_pixels(self.decoder(full))

    def forward(self, x, mask_ratio, generator):
        patches = patchify(x)
        keep_index, restore, mask = random_masking(len(x), mask_ratio, generator)
        prediction = self.decode(self.encode(patches, keep_index), restore)
        per_patch = ((prediction - patches) ** 2).mean(dim=-1)
        loss = (per_patch * mask).sum() / mask.sum()
        return loss, prediction, mask

    def features(self, x):
        return self.encode(patchify(x)).mean(dim=1)


def train_mae(X_train, mask_ratio, epochs, seed):
    torch.manual_seed(seed)
    model = MaskedAutoencoder()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    for epoch in range(epochs):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), 128):
            loss, _, _ = model(X_train[order[start:start + 128]], mask_ratio, generator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def few_label_indices(y, per_class, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y == digit), per_class, replace=False) for digit in range(10)])


def linear_probe(train_features, y_train, test_features, y_test, per_class=None):
    if per_class is not None:
        index = few_label_indices(y_train, per_class, seed=0)
        train_features, y_train = train_features[index], y_train[index]
    mean, std = train_features.mean(0), train_features.std(0) + 1e-6
    probe = LogisticRegression(max_iter=3000).fit((train_features - mean) / std, y_train)
    return probe.score((test_features - mean) / std, y_test)


def fine_tune(model, X, y, X_test, y_test, steps, seed):
    torch.manual_seed(seed)
    head = nn.Linear(model.position.shape[-1], 10)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.05)
    for step in range(steps):
        loss = F.cross_entropy(head(model.features(X)), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((head(model.features(X_test)).argmax(dim=1) == y_test).float().mean())


def save_reconstructions(x, prediction, mask, path):
    masked_view = patchify(x) * (1 - mask[:, :, None]) + 0.5 * mask[:, :, None]
    pasted = patchify(x) * (1 - mask[:, :, None]) + prediction * mask[:, :, None]
    rows = [("original", x), ("visible (gray = masked)", unpatchify(masked_view)),
            ("prediction", unpatchify(prediction)), ("visible + prediction", unpatchify(pasted))]
    fig, axes = plt.subplots(len(rows), 10, figsize=(10, 1.1 * len(rows)))
    for r, (name, images) in enumerate(rows):
        for c in range(10):
            axes[r, c].imshow(images[c].reshape(8, 8).clamp(0, 1), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = load_data(seed=0)

    print("=== Step 1: images as a sequence of patches ===")
    patches = patchify(X_train[:2])
    print(f"  image (64,) -> {N_PATCHES} patches of {PATCH}x{PATCH} = {tuple(patches.shape[1:])}, "
          f"round trip exact: {torch.equal(unpatchify(patches), X_train[:2])}")

    print("\n=== Step 2: random masking by shuffling ===")
    keep_index, restore, mask = random_masking(2, 0.75, torch.Generator().manual_seed(0))
    print(f"  keep patches {keep_index[0].tolist()} | mask row {mask[0].int().tolist()}")
    print(f"  masked fraction {mask.mean():.2f}, encoder sees {keep_index.shape[1]} of {N_PATCHES} tokens")

    print("\n=== Step 3: train MAEs with different mask ratios ===")
    raw_few = linear_probe(X_train.numpy(), y_train, X_test.numpy(), y_test, per_class=5)
    with torch.no_grad():
        untrained = MaskedAutoencoder()
        random_train, random_test = untrained.features(X_train).numpy(), untrained.features(X_test).numpy()
    print(f"  baselines (5 labels/class probe): raw pixels {raw_few:.3f} | untrained encoder "
          f"{linear_probe(random_train, y_train, random_test, y_test, per_class=5):.3f}")
    print(f"  {'mask ratio':10s} | masked-patch MSE | probe 5/class | probe all labels")
    models = {}
    for mask_ratio in [0.25, 0.5, 0.75]:
        model = train_mae(X_train, mask_ratio, epochs=40, seed=1)
        models[mask_ratio] = model
        with torch.no_grad():
            test_loss, _, _ = model(X_test, mask_ratio, torch.Generator().manual_seed(42))
            train_features, test_features = model.features(X_train).numpy(), model.features(X_test).numpy()
        print(f"  {mask_ratio:10.2f} | {test_loss.item():16.4f} | "
              f"{linear_probe(train_features, y_train, test_features, y_test, per_class=5):13.3f} | "
              f"{linear_probe(train_features, y_train, test_features, y_test):.3f}")

    print("\n=== Step 4: what the model fills in (mask ratio 0.5) ===")
    with torch.no_grad():
        _, prediction, mask = models[0.5](X_test[:10], 0.5, torch.Generator().manual_seed(3))
        mean_fill = patchify(X_train).mean(dim=0, keepdim=True).expand(10, -1, -1)
        mae_error = (((prediction - patchify(X_test[:10])) ** 2).mean(-1) * mask).sum() / mask.sum()
        mean_error = (((mean_fill - patchify(X_test[:10])) ** 2).mean(-1) * mask).sum() / mask.sum()
    print(f"  masked-patch MSE: MAE {mae_error:.4f} | fill with the average patch {mean_error:.4f}")
    save_reconstructions(X_test[:10], prediction, mask, OUT_DIR / "mae_reconstructions.png")
    print("  saved mae_reconstructions.png")

    print("\n=== Step 5: fine-tune the whole encoder with only 50 labels ===")
    index = few_label_indices(y_train, 5, seed=0)
    X_few, y_few, y_test_tensor = X_train[index], torch.tensor(y_train[index]), torch.tensor(y_test)
    torch.manual_seed(1)
    from_scratch = fine_tune(MaskedAutoencoder(), X_few, y_few, X_test, y_test_tensor, steps=300, seed=5)
    pretrained = fine_tune(copy.deepcopy(models[0.5]), X_few, y_few, X_test, y_test_tensor, steps=300, seed=5)
    print(f"  test accuracy: random init {from_scratch:.3f} | MAE-pretrained init {pretrained:.3f}")

    print("\n=== Use It: patch embedding is a strided convolution ===")
    model = models[0.5]
    conv = nn.Conv2d(1, model.patch_embed.shape[1], kernel_size=PATCH, stride=PATCH)
    with torch.no_grad():
        conv.weight.copy_(model.patch_embed.T.reshape(-1, 1, PATCH, PATCH))
        conv.bias.copy_(model.patch_bias)
        by_conv = conv(X_test[:4].reshape(-1, 1, 8, 8)).flatten(2).transpose(1, 2)
        by_matmul = patchify(X_test[:4]) @ model.patch_embed + model.patch_bias
    print(f"  Conv2d(kernel=stride={PATCH}) == patchify @ W: {torch.allclose(by_conv, by_matmul, atol=1e-6)}")
    print(f"  encoder parameters: {sum(p.numel() for n, p in model.named_parameters() if not n.startswith(('to_decoder', 'mask_token', 'decoder', 'to_pixels'))):,}"
          f" | decoder parameters: {sum(p.numel() for n, p in model.named_parameters() if n.startswith(('to_decoder', 'mask_token', 'decoder', 'to_pixels'))):,}")


if __name__ == "__main__":
    main()
