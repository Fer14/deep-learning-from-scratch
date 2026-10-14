from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

OUT_DIR = Path(__file__).parent


def sample_ring(n, generator):
    angle = 2 * np.pi * torch.rand(n, generator=generator)
    radius = 1.0 + 0.05 * torch.randn(n, generator=generator)
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)


def init_linear(n_in, n_out, generator):
    weight = torch.randn(n_in, n_out, generator=generator) * np.sqrt(2.0 / n_in)
    return [weight.requires_grad_(), torch.zeros(n_out).requires_grad_()]


class ScratchMLP:
    def __init__(self, sizes, generator, slope):
        self.layers = [init_linear(a, b, generator) for a, b in zip(sizes[:-1], sizes[1:])]
        self.slope = slope

    def parameters(self):
        return [p for layer in self.layers for p in layer]

    def __call__(self, h):
        for index, (weight, bias) in enumerate(self.layers):
            h = h @ weight + bias
            if index < len(self.layers) - 1:
                h = F.leaky_relu(h, self.slope)
        return h


def discriminator_loss(real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def generator_loss(fake_logits):
    return F.softplus(-fake_logits).mean()


def sliced_wasserstein(a, b, n_projections, generator):
    directions = torch.randn(2, n_projections, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True)
    projected_a = (a @ directions).sort(dim=0).values
    projected_b = (b @ directions).sort(dim=0).values
    return float((projected_a - projected_b).abs().mean())


def describe(points):
    radius = points.norm(dim=1)
    return f"radius mean {radius.mean():.3f} std {radius.std():.3f}"


def train_gan(generator_net, discriminator_net, steps, batch_size, seed, log_every=None, snapshots=None):
    rng = torch.Generator().manual_seed(seed)
    opt_g = torch.optim.Adam(generator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    fixed_noise = torch.randn(1000, 2, generator=rng)
    for step in range(steps + 1):
        real = sample_ring(batch_size, rng)
        fake = generator_net(torch.randn(batch_size, 2, generator=rng))
        loss_d = discriminator_loss(discriminator_net(real), discriminator_net(fake.detach()))
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        fake = generator_net(torch.randn(batch_size, 2, generator=rng))
        loss_g = generator_loss(discriminator_net(fake))
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if log_every and step % log_every == 0:
            with torch.no_grad():
                d_real = torch.sigmoid(discriminator_net(real)).mean()
                d_fake = torch.sigmoid(discriminator_net(fake)).mean()
                swd = sliced_wasserstein(generator_net(fixed_noise), sample_ring(1000, torch.Generator().manual_seed(7)),
                                         64, torch.Generator().manual_seed(8))
            print(f"  step {step:4d} | loss D {loss_d.item():.3f} | loss G {loss_g.item():.3f} | "
                  f"D(real) {d_real:.2f} | D(fake) {d_fake:.2f} | sliced W {swd:.3f}")
        if snapshots is not None and step in snapshots:
            with torch.no_grad():
                snapshots[step] = generator_net(fixed_noise).clone()
    return generator_net


def save_snapshots(snapshots, real, path):
    fig, axes = plt.subplots(1, len(snapshots), figsize=(3 * len(snapshots), 3))
    for ax, (step, points) in zip(axes, snapshots.items()):
        ax.scatter(real[:, 0], real[:, 1], s=2, color="0.75")
        ax.scatter(points[:, 0], points[:, 1], s=2, color="tab:red")
        ax.set_title(f"step {step}", fontsize=9)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    real_eval = sample_ring(1000, torch.Generator().manual_seed(7))
    print(f"real data: {describe(real_eval)}")

    print("\n=== Step 1: the discriminator loss is binary cross-entropy ===")
    logits = torch.tensor([3.0, -1.0, 0.0])
    print(f"  logits {logits.tolist()}")
    print(f"  softplus(-s)    = {[round(v, 4) for v in F.softplus(-logits).tolist()]}")
    print(f"  -log sigmoid(s) = {[round(v, 4) for v in (-torch.log(torch.sigmoid(logits))).tolist()]}")
    print(f"  untrained D, D(x) = 0.5 everywhere -> loss D = 2 log 2 = {2 * np.log(2):.3f}")

    print("\n=== Step 2: train the GAN (noise 2D -> generator -> ring) ===")
    generator_net = ScratchMLP([2, 64, 64, 2], torch.Generator().manual_seed(1), slope=0.2)
    discriminator_net = ScratchMLP([2, 64, 64, 1], torch.Generator().manual_seed(2), slope=0.2)
    snapshots = {0: None, 200: None, 600: None, 3000: None}
    train_gan(generator_net, discriminator_net, steps=3000, batch_size=256, seed=0, log_every=500, snapshots=snapshots)
    save_snapshots(snapshots, real_eval, OUT_DIR / "gan_ring.png")

    print("\n=== Step 3: compare samples with the data ===")
    with torch.no_grad():
        samples = generator_net(torch.randn(1000, 2, generator=torch.Generator().manual_seed(9)))
    gaussian_baseline = torch.randn(1000, 2, generator=torch.Generator().manual_seed(9)) * 0.7
    metric_rng = lambda: torch.Generator().manual_seed(8)
    print(f"  generated: {describe(samples)}")
    print(f"  sliced Wasserstein to real: GAN {sliced_wasserstein(samples, real_eval, 64, metric_rng()):.3f} | "
          f"a Gaussian blob {sliced_wasserstein(gaussian_baseline, real_eval, 64, metric_rng()):.3f} | "
          f"a second real sample {sliced_wasserstein(sample_ring(1000, torch.Generator().manual_seed(10)), real_eval, 64, metric_rng()):.3f}")
    print("  saved gan_ring.png")

    print("\n=== Use It: nn.Sequential + BCEWithLogitsLoss ===")
    real_logits, fake_logits = torch.randn(5), torch.randn(5)
    bce = nn.BCEWithLogitsLoss()
    library_d = bce(real_logits, torch.ones(5)) + bce(fake_logits, torch.zeros(5))
    print(f"  scratch D loss {discriminator_loss(real_logits, fake_logits):.5f} | BCEWithLogitsLoss {library_d:.5f}")
    print(f"  scratch G loss {generator_loss(fake_logits):.5f} | BCEWithLogitsLoss {bce(fake_logits, torch.ones(5)):.5f}")
    torch.manual_seed(1)
    library_g = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 2))
    library_d_net = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    train_gan(library_g, library_d_net, steps=3000, batch_size=256, seed=0)
    with torch.no_grad():
        library_samples = library_g(torch.randn(1000, 2, generator=torch.Generator().manual_seed(9)))
    print(f"  nn.Sequential GAN: {describe(library_samples)} | sliced W {sliced_wasserstein(library_samples, real_eval, 64, metric_rng()):.3f}")


if __name__ == "__main__":
    main()
