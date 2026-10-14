from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

OUT_DIR = Path(__file__).parent
CENTERS = torch.tensor([[2 * np.cos(a), 2 * np.sin(a)] for a in np.arange(8) * np.pi / 4], dtype=torch.float32)
NOISE_DIM = 16


def sample_mixture(n, generator):
    return CENTERS[torch.randint(8, (n,), generator=generator)] + 0.05 * torch.randn(n, 2, generator=generator)


def mode_report(samples):
    distance, nearest = torch.cdist(samples, CENTERS).min(dim=1)
    good = distance < 0.2
    counts = torch.bincount(nearest[good], minlength=8)
    covered = int((counts >= 0.02 * len(samples)).sum())
    return covered, float(good.float().mean()), counts


def mlp(n_in, n_out, hidden=128, wrap=lambda layer: layer):
    return nn.Sequential(wrap(nn.Linear(n_in, hidden)), nn.ReLU(), wrap(nn.Linear(hidden, hidden)), nn.ReLU(),
                         wrap(nn.Linear(hidden, n_out)))


def train(name, lr_g, lr_d, pack=1, steps=3000, seed=0, wrap=lambda layer: layer, snapshot_every=500):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    generator_net = mlp(NOISE_DIM, 2)
    discriminator_net = mlp(2 * pack, 1, wrap=wrap)
    opt_g = torch.optim.Adam(generator_net.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator_net.parameters(), lr=lr_d, betas=(0.5, 0.999))
    fixed_noise = torch.randn(2000, NOISE_DIM, generator=torch.Generator().manual_seed(1))
    batch_size = 256
    snapshots = {}
    print(f"\n  {name}")
    for step in range(steps + 1):
        real = sample_mixture(batch_size * pack, rng).reshape(batch_size, 2 * pack)
        fake = generator_net(torch.randn(batch_size * pack, NOISE_DIM, generator=rng)).reshape(batch_size, 2 * pack)
        loss_d = F.softplus(-discriminator_net(real)).mean() + F.softplus(discriminator_net(fake.detach())).mean()
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        fake = generator_net(torch.randn(batch_size * pack, NOISE_DIM, generator=rng)).reshape(batch_size, 2 * pack)
        loss_g = F.softplus(-discriminator_net(fake)).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if step % snapshot_every == 0:
            with torch.no_grad():
                samples = generator_net(fixed_noise)
            covered, quality, counts = mode_report(samples)
            center = samples.mean(dim=0)
            snapshots[step] = samples
            print(f"    step {step:4d} | loss D {loss_d.item():.3f} | loss G {loss_g.item():.3f} | modes {covered}/8 | "
                  f"on-mode {quality:.2f} | sample mean ({center[0]:+.2f}, {center[1]:+.2f}) | spread {samples.std(dim=0).mean():.2f}")
    return snapshots


def save_rows(rows, path):
    fig, axes = plt.subplots(len(rows), len(next(iter(rows.values()))), figsize=(14, 2.2 * len(rows)))
    for r, (name, snapshots) in enumerate(rows.items()):
        for c, (step, samples) in enumerate(snapshots.items()):
            ax = axes[r, c]
            ax.scatter(CENTERS[:, 0], CENTERS[:, 1], s=25, color="0.6")
            ax.scatter(samples[:, 0], samples[:, 1], s=1, color="tab:red")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {step}", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    print("=== Step 1: saturating vs non-saturating generator loss ===")
    fake_logits = torch.tensor([-8.0, -4.0, 0.0, 4.0], requires_grad=True)
    saturating = -F.softplus(fake_logits).sum()
    saturating_grad, = torch.autograd.grad(saturating, fake_logits)
    non_saturating = F.softplus(-fake_logits).sum()
    non_saturating_grad, = torch.autograd.grad(non_saturating, fake_logits)
    for logit, d, g_sat, g_ns in zip(fake_logits.tolist(), torch.sigmoid(fake_logits).tolist(),
                                     saturating_grad.tolist(), non_saturating_grad.tolist()):
        print(f"  fake logit {logit:+.0f} (D = {d:.4f}) | d loss / d logit: log(1-D) {g_sat:+.4f} | -log D {g_ns:+.4f}")

    print("\n=== Step 2: a slow discriminator (lr_D = lr_G / 10) ===")
    collapsed = train("G lr 1e-3, D lr 1e-4", lr_g=1e-3, lr_d=1e-4)

    print("\n=== Step 3: balance the players (lr_D = lr_G) ===")
    balanced = train("G lr 1e-3, D lr 1e-3", lr_g=1e-3, lr_d=1e-3)

    print("\n=== Step 4: PacGAN, D judges 4 samples at once (still the slow D) ===")
    packed = train("G lr 1e-3, D lr 1e-4, pack 4", lr_g=1e-3, lr_d=1e-4, pack=4)

    print("\n=== Use It: spectral normalization on D (still the slow D) ===")
    normalized = train("G lr 1e-3, D lr 1e-4, spectral_norm", lr_g=1e-3, lr_d=1e-4, wrap=spectral_norm)

    save_rows({"slow D": collapsed, "balanced": balanced, "pack 4": packed, "spectral norm": normalized},
              OUT_DIR / "mode_collapse.png")
    print("\n  saved mode_collapse.png")


if __name__ == "__main__":
    main()
