from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

OUT_DIR = Path(__file__).parent
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T)
ALPHA_BARS = torch.cumprod(1 - BETAS, dim=0)
N_CLASSES = 2
NULL = N_CLASSES


def moons(n, seed):
    points, labels = make_moons(n, noise=0.25, random_state=seed)
    points = (points - np.array([0.5, 0.25])) / 0.6
    return torch.tensor(points, dtype=torch.float32), torch.tensor(labels)


def timestep_embedding(t, dim=32):
    half = dim // 2
    frequencies = torch.exp(-np.log(10000.0) * torch.arange(half) / half)
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ConditionalNoisePredictor(nn.Module):
    def __init__(self, hidden=128, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.label_table = nn.Parameter(torch.randn(N_CLASSES + 1, hidden) * 0.1)
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input = nn.Linear(2, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output = nn.Linear(hidden, 2)

    def forward(self, x, t, labels):
        condition = self.time_mlp(timestep_embedding(t, self.embed_dim)) + self.label_table[labels]
        h = self.input(x) + condition
        for block in self.blocks:
            h = h + block(torch.nn.functional.silu(h)) + condition
        return self.output(torch.nn.functional.silu(h))


def train(model, data, labels, steps, drop_prob, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for step in range(steps):
        index = torch.randint(len(data), (512,), generator=generator)
        x0, y = data[index], labels[index].clone()
        y[torch.rand(512, generator=generator) < drop_prob] = NULL
        t = torch.randint(1, T + 1, (512,), generator=generator)
        eps = torch.randn(x0.shape, generator=generator)
        alpha_bar = ALPHA_BARS[t - 1][:, None]
        loss = ((model(alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps, t, y) - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"    step {step:4d} | loss {loss.item():.4f}")
    return model


def guided_eps(model, x, t, labels, scale):
    eps_cond = model(x, t, labels)
    eps_uncond = model(x, t, torch.full_like(labels, NULL))
    return eps_uncond + scale * (eps_cond - eps_uncond)


@torch.no_grad()
def sample(model, labels, scale, n_steps, seed, eps_fn=guided_eps):
    x = torch.randn(len(labels), 2, generator=torch.Generator().manual_seed(seed))
    timesteps = np.linspace(T, 0, n_steps + 1).round().astype(int)
    for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
        eps = eps_fn(model, x, torch.full((len(x),), int(t)), labels, scale)
        ab = ALPHA_BARS[t - 1]
        ab_prev = ALPHA_BARS[t_prev - 1] if t_prev > 0 else torch.tensor(1.0)
        x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
    return x


def sliced_wasserstein(a, b, n_projections=128, seed=8):
    directions = torch.randn(2, n_projections, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)
    return float(((a @ directions).sort(dim=0).values - (b @ directions).sort(dim=0).values).abs().mean())


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data, labels = moons(10000, seed=0)
    held_out, held_labels = moons(4000, seed=1)
    oracle = KNeighborsClassifier(n_neighbors=25).fit(data.numpy(), labels.numpy())
    print(f"two overlapping moons | oracle kNN accuracy on held-out real data {oracle.score(held_out.numpy(), held_labels.numpy()):.3f}")

    print("\n=== Step 1: one network, conditional and unconditional (labels dropped 20% of the time) ===")
    model = train(ConditionalNoisePredictor(), data, labels, steps=4000, drop_prob=0.2, seed=0)

    print("\n=== Step 2: sample class 0 at different guidance scales (DDIM, 50 steps) ===")
    requested = torch.zeros(2000, dtype=torch.long)
    real_class0 = held_out[held_labels == 0][:2000]
    print(f"  real class 0: oracle says class 0 for {np.mean(oracle.predict(real_class0.numpy()) == 0):.3f}, "
          f"spread {real_class0.std(0).mean():.3f}")
    print(f"  {'scale':>5s} | oracle says class 0 | mean P(class 0) | spread | sliced W to real class 0")
    samples_by_scale = {}
    for scale in [0.0, 1.0, 2.0, 4.0, 8.0]:
        samples = sample(model, requested, scale, 50, seed=3)
        samples_by_scale[scale] = samples
        probability = oracle.predict_proba(samples.numpy())[:, 0]
        print(f"  {scale:5.1f} | {np.mean(probability > 0.5):19.3f} | {probability.mean():15.3f} | {samples.std(0).mean():6.3f} | "
              f"{sliced_wasserstein(samples, real_class0):.4f}")
    unconditional = samples_by_scale[0.0]
    print(f"  scale 0 is the unconditional model: sliced W to all held-out data {sliced_wasserstein(unconditional, held_out[:2000]):.4f}")

    fig, axes = plt.subplots(1, 5, figsize=(14, 2.9))
    for ax, (scale, samples) in zip(axes, samples_by_scale.items()):
        ax.scatter(held_out[:, 0], held_out[:, 1], s=1, c=np.where(held_labels.numpy() == 0, "0.75", "0.9"))
        ax.scatter(samples[:, 0], samples[:, 1], s=1, color="tab:red")
        ax.set_title(f"guidance scale {scale}", fontsize=9)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cfg_scales.png", dpi=110)
    plt.close(fig)
    print("  saved cfg_scales.png")

    print("\n=== Use It: one batched forward pass for both predictions ===")

    def batched_eps(model, x, t, labels, scale):
        both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([labels, torch.full_like(labels, NULL)]))
        eps_cond, eps_uncond = both.chunk(2)
        return eps_uncond + scale * (eps_cond - eps_uncond)

    two_calls = sample(model, requested[:200], 4.0, 50, seed=7)
    one_call = sample(model, requested[:200], 4.0, 50, seed=7, eps_fn=batched_eps)
    print(f"  two forward passes vs one doubled batch: max difference {(two_calls - one_call).abs().max():.2e}")


if __name__ == "__main__":
    main()
