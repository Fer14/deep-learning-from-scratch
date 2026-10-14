import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from torch import nn

OUT_DIR = Path(__file__).parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T)
ALPHA_BARS = torch.cumprod(1 - BETAS, dim=0)
LATENT = 8


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, LATENT))
        self.decoder = nn.Sequential(nn.Linear(LATENT, 128), nn.SiLU(), nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 64))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X, epochs, seed):
    torch.manual_seed(seed)
    model = Autoencoder()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), 128):
            batch = X[order[start:start + 128]]
            loss = ((model(batch) - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def timestep_embedding(t, dim=32):
    half = dim // 2
    frequencies = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class NoisePredictor(nn.Module):
    def __init__(self, dim, hidden=192, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output = nn.Linear(hidden, dim)

    def forward(self, x, t):
        time_features = self.time_mlp(timestep_embedding(t, self.embed_dim))
        h = self.input(x) + time_features
        for block in self.blocks:
            h = h + block(torch.nn.functional.silu(h)) + time_features
        return self.output(torch.nn.functional.silu(h))


def train_diffusion(data, steps, seed):
    torch.manual_seed(seed)
    device = data.device
    model = NoisePredictor(data.shape[1]).to(device)
    alpha_bars = ALPHA_BARS.to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for step in range(steps):
        x0 = data[torch.randint(len(data), (256,), generator=generator).to(device)]
        t = torch.randint(1, T + 1, (256,), generator=generator).to(device)
        eps = torch.randn(x0.shape, generator=generator).to(device)
        alpha_bar = alpha_bars[t - 1][:, None]
        loss = ((model(alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps, t) - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
    return model


@torch.no_grad()
def ddim_sample(model, n, dim, n_steps, seed):
    device = next(model.parameters()).device
    x = torch.randn(n, dim, generator=torch.Generator().manual_seed(seed)).to(device)
    timesteps = np.linspace(T, 0, n_steps + 1).round().astype(int)
    for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
        eps = model(x, torch.full((n,), int(t), device=device))
        ab = ALPHA_BARS[t - 1]
        ab_prev = ALPHA_BARS[t_prev - 1] if t_prev > 0 else torch.tensor(1.0)
        x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
    return x.cpu()


def judge(classifier, images):
    pixels = ((images + 1) / 2).clamp(0, 1).numpy()
    probabilities = classifier.predict_proba(pixels)
    counts = np.bincount(probabilities.argmax(1), minlength=10)
    return float(np.mean(probabilities.max(1) > 0.9)), counts


def save_grid(rows, path):
    fig, axes = plt.subplots(len(rows), 12, figsize=(12 * 0.8, len(rows) * 0.95))
    for r, (name, images) in enumerate(rows):
        for c in range(12):
            axes[r, c].imshow(((images[c] + 1) / 2).clamp(0, 1).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    digits = load_digits()
    pixels = (digits.data / 16.0).astype(np.float32)
    classifier = LogisticRegression(max_iter=3000).fit(pixels, digits.target)
    X = torch.tensor(pixels * 2 - 1)
    real_confident, _ = judge(classifier, X)
    print(f"device {device} | {len(X)} digits as 64-dim vectors in [-1, 1] | real digits judged confident: {real_confident:.2f}")

    print("\n=== Step 1: compress 64 pixels into an 8-number latent ===")
    autoencoder = train_autoencoder(X, epochs=150, seed=0)
    with torch.no_grad():
        latents = autoencoder.encoder(X)
        reconstructions = autoencoder.decoder(latents)
    reconstruction_confident, _ = judge(classifier, reconstructions)
    print(f"  reconstruction MSE {((reconstructions - X) ** 2).mean():.4f} | reconstructions judged confident: {reconstruction_confident:.2f}"
          f" (the ceiling for latent diffusion)")
    scale = latents.std()
    print(f"  raw latent std {scale:.3f} -> divide by it so the diffusion sees unit-scale data (scaling factor {1 / scale:.3f})")
    scaled_latents = latents / scale

    print("\n=== Step 2: decode random Gaussian latents (no diffusion) ===")
    with torch.no_grad():
        gaussian_decoded = autoencoder.decoder(torch.randn(1000, LATENT, generator=torch.Generator().manual_seed(1)) * scale)
    confident, counts = judge(classifier, gaussian_decoded)
    print(f"  confident {confident:.2f} | class counts {counts.tolist()}")

    print("\n=== Step 3: diffusion in latent space vs in pixel space (same network, same steps) ===")
    start = time.perf_counter()
    latent_model = train_diffusion(scaled_latents.to(device), steps=3000, seed=2)
    latent_train_time = time.perf_counter() - start
    start = time.perf_counter()
    pixel_model = train_diffusion(X.to(device), steps=3000, seed=2)
    pixel_train_time = time.perf_counter() - start

    start = time.perf_counter()
    with torch.no_grad():
        latent_samples = autoencoder.decoder(ddim_sample(latent_model, 1000, LATENT, 50, seed=3) * scale)
        if device == "cuda":
            torch.cuda.synchronize()
    latent_sample_time = time.perf_counter() - start
    start = time.perf_counter()
    pixel_samples = ddim_sample(pixel_model, 1000, 64, 50, seed=3).clamp(-1, 1)
    pixel_sample_time = time.perf_counter() - start

    for name, samples, train_time, sample_time in [("latent diffusion", latent_samples, latent_train_time, latent_sample_time),
                                                    ("pixel diffusion", pixel_samples, pixel_train_time, pixel_sample_time)]:
        confident, counts = judge(classifier, samples)
        print(f"  {name:16s} | train {train_time:5.1f}s | sample {sample_time:4.2f}s | confident {confident:.2f} | class counts {counts.tolist()}")

    print("\n=== Step 4: where does the work go? ===")
    latent_ops = 2 * (LATENT * 192 + 3 * 192 * 192 + 192 * LATENT)
    pixel_ops = 2 * (64 * 192 + 3 * 192 * 192 + 192 * 64)
    print(f"  FLOPs per denoiser call (excluding time MLP): latent {latent_ops:,} | pixel {pixel_ops:,}")
    print(f"  data dimension the diffusion must model: latent {LATENT} | pixel 64 ({64 // LATENT}x larger)")
    save_grid([("real", X[:12]), ("AE reconstruction", reconstructions[:12]), ("Gaussian latent, decoded", gaussian_decoded[:12]),
               ("latent diffusion", latent_samples[:12]), ("pixel diffusion", pixel_samples[:12])],
              OUT_DIR / "latent_diffusion.png")
    print("  saved latent_diffusion.png")

    print("\n=== Use It: why the scaling factor matters ===")
    for name, data_std in [("scaled latents", 1.0), ("raw latents", float(scale))]:
        snr = [float(ALPHA_BARS[t - 1] * data_std ** 2 / (1 - ALPHA_BARS[t - 1])) for t in (250, 500, 750, 1000)]
        print(f"  {name:14s} (std {data_std:.2f}) | SNR at t=250/500/750/1000: " + " / ".join(f"{v:.4f}" for v in snr))


if __name__ == "__main__":
    main()
