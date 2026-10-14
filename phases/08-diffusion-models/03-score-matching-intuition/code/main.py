from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

OUT_DIR = Path(__file__).parent
MEANS = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
WEIGHTS = torch.tensor([0.8, 0.2])
STD = 0.3
SIGMAS = torch.tensor(np.geomspace(3.0, 0.05, 10), dtype=torch.float32)


def sample_data(n, generator):
    component = torch.multinomial(WEIGHTS, n, replacement=True, generator=generator)
    return MEANS[component] + STD * torch.randn(n, 2, generator=generator)


def log_density(x, extra_std=0.0):
    variance = STD ** 2 + extra_std ** 2
    squared = ((x[:, None, :] - MEANS[None]) ** 2).sum(-1)
    log_components = torch.log(WEIGHTS)[None] - squared / (2 * variance) - np.log(2 * np.pi * variance)
    return torch.logsumexp(log_components, dim=1)


def true_score(x, extra_std=0.0):
    x = x.detach().requires_grad_(True)
    grad, = torch.autograd.grad(log_density(x, extra_std).sum(), x)
    return grad


def langevin(score_fn, x, step_size, n_steps, generator):
    for _ in range(n_steps):
        x = x + step_size * score_fn(x) + np.sqrt(2 * step_size) * torch.randn(x.shape, generator=generator)
    return x


def annealed_langevin(score_fn, x, steps_per_level, generator, base_step=0.05):
    for sigma in SIGMAS:
        step_size = base_step * sigma ** 2
        for _ in range(steps_per_level):
            x = x + step_size * score_fn(x, sigma) + torch.sqrt(2 * step_size) * torch.randn(x.shape, generator=generator)
    return x


def left_fraction(x):
    return float((x[:, 0] < 0).float().mean())


class ScoreNetwork(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, sigma):
        sigma = torch.as_tensor(sigma).expand(len(x))[:, None]
        return self.net(torch.cat([x, torch.log(sigma)], dim=1)) / sigma


def train_score_network(steps, seed):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ScoreNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        x = sample_data(512, generator)
        sigma = SIGMAS[torch.randint(len(SIGMAS), (512,), generator=generator)]
        eps = torch.randn(x.shape, generator=generator)
        noisy = x + sigma[:, None] * eps
        target = -eps / sigma[:, None]
        loss = ((sigma[:, None] * (model(noisy, sigma) - target)) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"    step {step:4d} | weighted DSM loss {loss.item():.4f}")
    return model


def sliced_wasserstein(a, b, n_projections=128, seed=8):
    directions = torch.randn(2, n_projections, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)
    return float(((a @ directions).sort(dim=0).values - (b @ directions).sort(dim=0).values).abs().mean())


def save_plot(model, samples_by_name, path):
    grid = torch.linspace(-4, 4, 21)
    gx, gy = torch.meshgrid(grid, grid, indexing="xy")
    points = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    fig, axes = plt.subplots(1, 1 + len(samples_by_name), figsize=(4 * (1 + len(samples_by_name)), 4))
    with torch.no_grad():
        learned = model(points, SIGMAS[5])
    axes[0].quiver(points[:, 0], points[:, 1], learned[:, 0], learned[:, 1], color="tab:green")
    axes[0].set_title(f"learned score, sigma = {SIGMAS[5]:.2f}", fontsize=9)
    for ax, (name, samples) in zip(axes[1:], samples_by_name.items()):
        ax.scatter(samples[:, 0], samples[:, 1], s=1)
        ax.set_title(f"{name}: left fraction {left_fraction(samples):.2f}", fontsize=9)
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = sample_data(4000, torch.Generator().manual_seed(1))
    print(f"data: mixture with weights 0.8 at x = -2 and 0.2 at x = +2 (std {STD}), left fraction {left_fraction(data):.2f}")

    print("\n=== Step 1: the score points toward higher density ===")
    probes = torch.tensor([[-2.0, 0.0], [-1.5, 0.5], [0.0, 0.0], [1.0, 0.0], [3.0, 1.0]])
    for point, score in zip(probes, true_score(probes)):
        print(f"  x = ({point[0]:+.1f}, {point[1]:+.1f}) -> score ({score[0]:+8.3f}, {score[1]:+8.3f})")

    print("\n=== Step 2: Langevin dynamics with the exact score ===")
    generator = torch.Generator().manual_seed(2)
    start = torch.rand(4000, 2, generator=generator) * 8 - 4
    plain = langevin(true_score, start, step_size=0.01, n_steps=1000, generator=generator)
    print(f"  1000 steps from uniform starts: left fraction {left_fraction(plain):.2f} (target 0.80) | "
          f"sliced W {sliced_wasserstein(plain, data):.3f}")

    print("\n=== Step 3: annealed Langevin with the exact noisy scores ===")
    annealed_true = annealed_langevin(lambda x, sigma: true_score(x, float(sigma)), start, 100, torch.Generator().manual_seed(3))
    print(f"  10 noise levels x 100 steps: left fraction {left_fraction(annealed_true):.2f} | sliced W {sliced_wasserstein(annealed_true, data):.3f}")

    print("\n=== Step 4: learn the score with denoising score matching ===")
    model = train_score_network(steps=4000, seed=0)
    check_points = sample_data(2000, torch.Generator().manual_seed(4))
    for level in [0, 4, 9]:
        sigma = float(SIGMAS[level])
        noisy = check_points + sigma * torch.randn(check_points.shape, generator=torch.Generator().manual_seed(level))
        with torch.no_grad():
            learned = model(noisy, SIGMAS[level])
        exact = true_score(noisy, sigma)
        cosine = torch.nn.functional.cosine_similarity(learned, exact).mean()
        relative = ((learned - exact).norm(dim=1) / exact.norm(dim=1).clamp(min=1e-3)).median()
        print(f"  sigma {sigma:.3f} | cosine(learned, true) {cosine:.3f} | median relative error {relative:.3f}")

    with torch.no_grad():
        annealed_learned = annealed_langevin(lambda x, sigma: model(x, sigma), start, 100, torch.Generator().manual_seed(3))
    print(f"  annealed Langevin with the learned score: left fraction {left_fraction(annealed_learned):.2f} | "
          f"sliced W {sliced_wasserstein(annealed_learned, data):.3f}")

    print("\n=== Step 5: score and noise are the same thing ===")
    sigma = SIGMAS[6]
    eps = torch.randn(check_points.shape, generator=torch.Generator().manual_seed(6))
    noisy = check_points + sigma * eps
    with torch.no_grad():
        eps_from_score = -sigma * model(noisy, sigma)
    print(f"  sigma {sigma:.3f}: -sigma * score vs the true eps, MSE {((eps_from_score - eps) ** 2).mean():.3f} "
          f"(predicting zero gives {(eps ** 2).mean():.3f})")
    save_plot(model, {"plain Langevin": plain, "annealed, true score": annealed_true, "annealed, learned score": annealed_learned},
              OUT_DIR / "score_langevin.png")
    print("  saved score_langevin.png")

    print("\n=== Use It: torch.distributions + autograd for the exact score ===")
    mixture = MixtureSameFamily(Categorical(probs=WEIGHTS), Independent(Normal(MEANS, torch.full_like(MEANS, STD)), 1))
    x = probes.clone().requires_grad_(True)
    library_score, = torch.autograd.grad(mixture.log_prob(x).sum(), x)
    print(f"  log p matches: {torch.allclose(mixture.log_prob(probes), log_density(probes), atol=1e-5)} | "
          f"score matches: {torch.allclose(library_score, true_score(probes), atol=1e-4)}")


if __name__ == "__main__":
    main()
