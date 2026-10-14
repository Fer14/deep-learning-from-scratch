import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.datasets import make_swiss_roll
from torch import nn

OUT_DIR = Path(__file__).parent


def swiss_roll(n, seed):
    points, _ = make_swiss_roll(n, noise=0.3, random_state=seed)
    return torch.tensor(points[:, [0, 2]] / 10.0, dtype=torch.float32)


def time_embedding(t, dim=32):
    half = dim // 2
    frequencies = torch.exp(-np.log(10000.0) * torch.arange(half) / half)
    angles = 1000 * t[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class VelocityField(nn.Module):
    def __init__(self, hidden=128, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input = nn.Linear(2, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output = nn.Linear(hidden, 2)

    def forward(self, x, t):
        time_features = self.time_mlp(time_embedding(t, self.embed_dim))
        h = self.input(x) + time_features
        for block in self.blocks:
            h = h + block(torch.nn.functional.silu(h)) + time_features
        return self.output(torch.nn.functional.silu(h))


def train_flow(pair_sampler, steps, seed):
    torch.manual_seed(seed)
    model = VelocityField()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for step in range(steps):
        x0, x1 = pair_sampler(512, generator)
        t = torch.rand(512, generator=generator)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = ((model(x_t, t) - (x1 - x0)) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"    step {step:4d} | velocity MSE {loss.item():.4f}")
    return model


@torch.no_grad()
def euler(model, x, n_steps, keep_path=False):
    path = [x]
    for k in range(n_steps):
        t = torch.full((len(x),), k / n_steps)
        x = x + model(x, t) / n_steps
        path.append(x)
    return (x, torch.stack(path)) if keep_path else x


def sliced_wasserstein(a, b, n_projections=128, seed=8):
    directions = torch.randn(2, n_projections, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)
    return float(((a @ directions).sort(dim=0).values - (b @ directions).sort(dim=0).values).abs().mean())


def straightness(path):
    chord = path[-1] - path[0]
    steps = path[1:] - path[:-1]
    length = steps.norm(dim=2).sum(dim=0)
    return float((chord.norm(dim=1) / length.clamp(min=1e-8)).mean())


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = swiss_roll(10000, seed=0)
    held_out = swiss_roll(2000, seed=1)
    print(f"reference: sliced W between two real samples {sliced_wasserstein(swiss_roll(2000, seed=2), held_out):.4f}")

    print("\n=== Step 1: the conditional path and its velocity ===")
    x0, x1 = torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 2.0]])
    for t in [0.0, 0.25, 0.5, 1.0]:
        x_t = (1 - t) * x0 + t * x1
        print(f"  t={t:.2f} | x_t = ({x_t[0, 0]:.2f}, {x_t[0, 1]:.2f}) | target velocity x1 - x0 = (1.00, 2.00)")

    print("\n=== Step 2: train the velocity field (noise -> data, independent pairs) ===")

    def independent_pairs(n, generator):
        return torch.randn(n, 2, generator=generator), data[torch.randint(len(data), (n,), generator=generator)]

    start = time.perf_counter()
    model = train_flow(independent_pairs, steps=3000, seed=0)
    print(f"  trained in {time.perf_counter() - start:.1f}s")

    print("\n=== Step 3: sample by integrating the ODE with Euler steps ===")
    noise = torch.randn(2000, 2, generator=torch.Generator().manual_seed(3))
    euler_results = {}
    for n_steps in [1, 2, 5, 10, 20, 100]:
        samples = euler(model, noise, n_steps)
        euler_results[n_steps] = samples
        print(f"  {n_steps:3d} Euler steps | sliced W {sliced_wasserstein(samples, held_out):.4f}")
    _, path = euler(model, noise[:500], 100, keep_path=True)
    print(f"  straightness of 100-step trajectories (chord / arc length, 1 = straight): {straightness(path):.3f}")

    print("\n=== Step 4: reflow, retrain on the model's own (noise, sample) pairs ===")
    with torch.no_grad():
        coupled_noise = torch.randn(10000, 2, generator=torch.Generator().manual_seed(4))
        coupled_data = euler(model, coupled_noise, 100)

    def coupled_pairs(n, generator):
        index = torch.randint(len(coupled_noise), (n,), generator=generator)
        return coupled_noise[index], coupled_data[index]

    reflowed = train_flow(coupled_pairs, steps=2000, seed=1)
    _, reflow_path = euler(reflowed, noise[:500], 100, keep_path=True)
    print(f"  straightness after reflow: {straightness(reflow_path):.3f}")
    for n_steps in [1, 2, 5]:
        print(f"  {n_steps} Euler steps | original {sliced_wasserstein(euler_results[n_steps], held_out):.4f} | "
              f"reflowed {sliced_wasserstein(euler(reflowed, noise, n_steps), held_out):.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [("1 step", euler_results[1]), ("5 steps", euler_results[5]), ("100 steps", euler_results[100]),
              ("reflowed, 1 step", euler(reflowed, noise, 1))]
    for ax, (title, samples) in zip(axes, panels):
        ax.scatter(samples[:, 0], samples[:, 1], s=1, color="tab:orange")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "flow_matching.png", dpi=110)
    plt.close(fig)
    print("  saved flow_matching.png")

    print("\n=== Use It: an adaptive ODE solver from SciPy ===")

    def velocity(t, flat_x):
        x = torch.tensor(flat_x, dtype=torch.float32).reshape(-1, 2)
        with torch.no_grad():
            v = model(x, torch.full((len(x),), float(t)))
        return v.numpy().ravel().astype(np.float64)

    for tolerance in [1e-2, 1e-4]:
        solution = solve_ivp(velocity, (0.0, 1.0), noise.numpy().ravel().astype(np.float64), method="RK45",
                             rtol=tolerance, atol=tolerance)
        samples = torch.tensor(solution.y[:, -1], dtype=torch.float32).reshape(-1, 2)
        print(f"  RK45 tolerance {tolerance:.0e} | {solution.nfev:3d} network calls | sliced W {sliced_wasserstein(samples, held_out):.4f}")


if __name__ == "__main__":
    main()
