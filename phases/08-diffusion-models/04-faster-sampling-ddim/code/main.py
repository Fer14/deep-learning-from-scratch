import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from torch import nn

OUT_DIR = Path(__file__).parent
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T)
ALPHA_BARS = torch.cumprod(1 - BETAS, dim=0)


def swiss_roll(n, seed):
    points, _ = make_swiss_roll(n, noise=0.3, random_state=seed)
    return torch.tensor(points[:, [0, 2]] / 10.0, dtype=torch.float32)


def timestep_embedding(t, dim=32):
    half = dim // 2
    frequencies = torch.exp(-np.log(10000.0) * torch.arange(half) / half)
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class NoisePredictor(nn.Module):
    def __init__(self, hidden=128, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input = nn.Linear(2, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output = nn.Linear(hidden, 2)

    def forward(self, x, t):
        time_features = self.time_mlp(timestep_embedding(t, self.embed_dim))
        h = self.input(x) + time_features
        for block in self.blocks:
            h = h + block(torch.nn.functional.silu(h)) + time_features
        return self.output(torch.nn.functional.silu(h))


def train(model, data, steps, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for step in range(steps):
        x0 = data[torch.randint(len(data), (512,), generator=generator)]
        t = torch.randint(1, T + 1, (512,), generator=generator)
        eps = torch.randn(x0.shape, generator=generator)
        alpha_bar = ALPHA_BARS[t - 1][:, None]
        loss = ((model(alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps, t) - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
    return model


def alpha_bar_at(t):
    return ALPHA_BARS[t - 1] if t > 0 else torch.tensor(1.0)


@torch.no_grad()
def ddim_sample(model, x, n_steps, eta, generator):
    timesteps = np.linspace(T, 0, n_steps + 1).round().astype(int)
    for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
        eps = model(x, torch.full((len(x),), int(t)))
        ab, ab_prev = alpha_bar_at(t), alpha_bar_at(t_prev)
        x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
        direction = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        x = ab_prev.sqrt() * x0_hat + direction
        if eta > 0 and t_prev > 0:
            x = x + sigma * torch.randn(x.shape, generator=generator)
    return x


@torch.no_grad()
def ddpm_sample(model, x, generator):
    for t in range(T, 0, -1):
        eps = model(x, torch.full((len(x),), t))
        beta, ab = BETAS[t - 1], ALPHA_BARS[t - 1]
        x = (x - beta / (1 - ab).sqrt() * eps) / (1 - beta).sqrt()
        if t > 1:
            x = x + beta.sqrt() * torch.randn(x.shape, generator=generator)
    return x


@torch.no_grad()
def ddim_invert(model, x0, n_steps):
    timesteps = np.linspace(0, T, n_steps + 1).round().astype(int)
    x = x0
    for t, t_next in zip(timesteps[:-1], timesteps[1:]):
        eps = model(x, torch.full((len(x),), int(max(t, 1))))
        ab, ab_next = alpha_bar_at(t), alpha_bar_at(t_next)
        x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x = ab_next.sqrt() * x0_hat + (1 - ab_next).sqrt() * eps
    return x


def sliced_wasserstein(a, b, n_projections=128, seed=8):
    directions = torch.randn(2, n_projections, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)
    return float(((a @ directions).sort(dim=0).values - (b @ directions).sort(dim=0).values).abs().mean())


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = swiss_roll(10000, seed=0)
    held_out = swiss_roll(2000, seed=1)
    print(f"reference: sliced W between two real samples {sliced_wasserstein(swiss_roll(2000, seed=2), held_out):.4f}")

    print("\n=== Step 1: train one noise predictor (the DDPM objective, unchanged) ===")
    start = time.perf_counter()
    model = train(NoisePredictor(), data, steps=4000, seed=0)
    print(f"  trained in {time.perf_counter() - start:.1f}s")

    noise = torch.randn(2000, 2, generator=torch.Generator().manual_seed(3))
    print("\n=== Step 2: DDPM ancestral sampling, 1000 steps ===")
    start = time.perf_counter()
    ddpm = ddpm_sample(model, noise, torch.Generator().manual_seed(4))
    print(f"  1000 network calls | {time.perf_counter() - start:5.2f}s | sliced W {sliced_wasserstein(ddpm, held_out):.4f}")

    print("\n=== Step 3: DDIM (eta = 0, deterministic) with fewer steps ===")
    results = {}
    for n_steps in [100, 50, 20, 10, 5, 2]:
        start = time.perf_counter()
        samples = ddim_sample(model, noise, n_steps, eta=0.0, generator=None)
        results[n_steps] = samples
        print(f"  {n_steps:4d} network calls | {time.perf_counter() - start:5.2f}s | sliced W {sliced_wasserstein(samples, held_out):.4f}")

    print("\n=== Step 4: deterministic means invertible ===")
    x0 = held_out[:500]
    latents = ddim_invert(model, x0, n_steps=100)
    reconstructed = ddim_sample(model, latents, 100, eta=0.0, generator=None)
    print(f"  inverted latents: mean ({latents[:, 0].mean():+.3f}, {latents[:, 1].mean():+.3f}), "
          f"std ({latents[:, 0].std():.3f}, {latents[:, 1].std():.3f})")
    print(f"  reconstruction error after invert -> sample: {(reconstructed - x0).norm(dim=1).mean():.4f} "
          f"(data spread {x0.std(0).mean():.3f})")
    same_noise_twice = ddim_sample(model, noise[:500], 20, eta=0.0, generator=None)
    print(f"  same noise, sampled twice with DDIM: max difference {(same_noise_twice - results[20][:500]).abs().max():.2e}")

    print("\n=== Use It: the eta knob (eta = 1 is DDPM-like) ===")
    for n_steps in [100, 20, 10]:
        for eta in [0.0, 0.5, 1.0]:
            samples = ddim_sample(model, noise, n_steps, eta=eta, generator=torch.Generator().manual_seed(5))
            print(f"  {n_steps:3d} steps, eta {eta:.1f} | sliced W {sliced_wasserstein(samples, held_out):.4f}")

    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
    panels = [("DDPM 1000 steps", ddpm), ("DDIM 50", results[50]), ("DDIM 10", results[10]), ("DDIM 5", results[5]), ("DDIM 2", results[2])]
    for ax, (title, samples) in zip(axes, panels):
        ax.scatter(samples[:, 0], samples[:, 1], s=1, color="tab:purple")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ddim_steps.png", dpi=110)
    plt.close(fig)
    print("  saved ddim_steps.png")


if __name__ == "__main__":
    main()
