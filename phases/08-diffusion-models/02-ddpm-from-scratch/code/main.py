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


def swiss_roll(n, seed):
    points, _ = make_swiss_roll(n, noise=0.3, random_state=seed)
    return torch.tensor(points[:, [0, 2]] / 10.0, dtype=torch.float32)


def make_schedule(n_steps):
    betas = torch.linspace(1e-4, 0.02, n_steps)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
    posterior_variance = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars, "posterior_variance": posterior_variance}


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


def train(model, data, schedule, steps, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for step in range(steps):
        x0 = data[torch.randint(len(data), (batch_size,), generator=generator)]
        t = torch.randint(1, T + 1, (batch_size,), generator=generator)
        eps = torch.randn(x0.shape, generator=generator)
        alpha_bar = schedule["alpha_bars"][t - 1][:, None]
        x_t = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
        loss = ((model(x_t, t) - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"    step {step:4d} | noise-prediction MSE {loss.item():.4f}")
    return model


@torch.no_grad()
def sample(model, schedule, n, seed, variance="beta", keep=()):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=generator)
    trajectory = {T: x.clone()}
    for t in range(T, 0, -1):
        t_batch = torch.full((n,), t)
        eps = model(x, t_batch)
        beta, alpha, alpha_bar = schedule["betas"][t - 1], schedule["alphas"][t - 1], schedule["alpha_bars"][t - 1]
        mean = (x - beta / (1 - alpha_bar).sqrt() * eps) / alpha.sqrt()
        if t > 1:
            sigma2 = beta if variance == "beta" else schedule["posterior_variance"][t - 1]
            x = mean + sigma2.sqrt() * torch.randn(x.shape, generator=generator)
        else:
            x = mean
        if t - 1 in keep:
            trajectory[t - 1] = x.clone()
    return x, trajectory


def sliced_wasserstein(a, b, n_projections=128, seed=8):
    directions = torch.randn(2, n_projections, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)
    return float(((a @ directions).sort(dim=0).values - (b @ directions).sort(dim=0).values).abs().mean())


def save_trajectory(trajectory, path):
    steps = sorted(trajectory, reverse=True)
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 2.6))
    for ax, t in zip(axes, steps):
        points = trajectory[t]
        ax.scatter(points[:, 0], points[:, 1], s=1, color="tab:purple")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(f"x_{t}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = swiss_roll(10000, seed=0)
    held_out = swiss_roll(2000, seed=1)
    schedule = make_schedule(T)

    print("=== Step 1: the training objective on one batch ===")
    t = torch.tensor([10, 500, 990])
    print(f"  sinusoidal embedding of t={t.tolist()} has shape {tuple(timestep_embedding(t).shape)}")
    print(f"  sqrt(alpha_bar) at those t: {[round(v, 3) for v in schedule['alpha_bars'][t - 1].sqrt().tolist()]}")

    print("\n=== Step 2: train the noise predictor ===")
    model = NoisePredictor()
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    start = time.perf_counter()
    train(model, data, schedule, steps=4000, batch_size=512, seed=0)
    print(f"  trained in {time.perf_counter() - start:.1f}s")

    print("\n=== Step 3: ancestral sampling, 1000 reverse steps ===")
    start = time.perf_counter()
    samples, trajectory = sample(model, schedule, 2000, seed=3, keep=(750, 500, 250, 100, 0))
    print(f"  sampled 2000 points in {time.perf_counter() - start:.1f}s")
    gaussian = torch.randn(2000, 2, generator=torch.Generator().manual_seed(4)) * held_out.std(0) + held_out.mean(0)
    print(f"  sliced Wasserstein to held-out data: DDPM {sliced_wasserstein(samples, held_out):.4f} | "
          f"Gaussian with the data's mean and std {sliced_wasserstein(gaussian, held_out):.4f} | "
          f"another real sample {sliced_wasserstein(swiss_roll(2000, seed=2), held_out):.4f}")
    save_trajectory(trajectory, OUT_DIR / "ddpm_trajectory.png")
    print("  saved ddpm_trajectory.png")

    print("\n=== Step 4: what the network predicts at different noise levels ===")
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for t_value in [10, 100, 500, 900]:
            x0 = held_out[:1000]
            eps = torch.randn(x0.shape, generator=generator)
            alpha_bar = schedule["alpha_bars"][t_value - 1]
            x_t = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
            mse = ((model(x_t, torch.full((1000,), t_value)) - eps) ** 2).mean()
            x0_hat = (x_t - (1 - alpha_bar).sqrt() * model(x_t, torch.full((1000,), t_value))) / alpha_bar.sqrt()
            print(f"  t={t_value:3d} | eps MSE {mse:.3f} | implied x0 error {(x0_hat - x0).norm(dim=1).mean():.3f}")

    print("\n=== Use It: the reverse-variance choice ===")
    for variance in ["beta", "posterior"]:
        samples_v, _ = sample(model, schedule, 2000, seed=3, variance=variance)
        print(f"  sigma_t^2 = {variance:9s} | sliced W {sliced_wasserstein(samples_v, held_out):.4f}")


if __name__ == "__main__":
    main()
