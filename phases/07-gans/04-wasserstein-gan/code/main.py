from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    return int((counts >= 0.02 * len(samples)).sum()), float(good.float().mean())


def sliced_wasserstein(a, b, n_projections=64, seed=8):
    directions = torch.randn(2, n_projections, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)
    return float(((a @ directions).sort(dim=0).values - (b @ directions).sort(dim=0).values).abs().mean())


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def gradient_penalty(critic, real, fake, generator):
    alpha = torch.rand(len(real), 1, generator=generator)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(mixed)
    grads, = torch.autograd.grad(scores.sum(), mixed, create_graph=True)
    return ((grads.norm(dim=1) - 1) ** 2).mean()


def train_wgan(name, mode, generator_steps, seed=0, n_critic=5):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    generator_net, critic = mlp(NOISE_DIM, 2), mlp(2, 1)
    if mode == "gp":
        opt_g = torch.optim.Adam(generator_net.parameters(), lr=1e-3, betas=(0.0, 0.9))
        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.9))
    else:
        opt_g = torch.optim.RMSprop(generator_net.parameters(), lr=5e-4)
        opt_c = torch.optim.RMSprop(critic.parameters(), lr=5e-4)
    batch_size = 256
    eval_noise = torch.randn(2000, NOISE_DIM, generator=torch.Generator().manual_seed(1))
    eval_real = sample_mixture(2000, torch.Generator().manual_seed(2))
    print(f"\n  {name}")
    history = []
    for step in range(generator_steps + 1):
        for _ in range(n_critic):
            real = sample_mixture(batch_size, rng)
            with torch.no_grad():
                fake = generator_net(torch.randn(batch_size, NOISE_DIM, generator=rng))
            wasserstein_estimate = critic(real).mean() - critic(fake).mean()
            loss_c = -wasserstein_estimate
            if mode == "gp":
                loss_c = loss_c + 1.0 * gradient_penalty(critic, real, fake, rng)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            if mode == "clip":
                with torch.no_grad():
                    for parameter in critic.parameters():
                        parameter.clamp_(-0.01, 0.01)
        loss_g = -critic(generator_net(torch.randn(batch_size, NOISE_DIM, generator=rng))).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if step % 250 == 0:
            with torch.no_grad():
                samples = generator_net(eval_noise)
                critic_gap = float(critic(eval_real).mean() - critic(samples).mean())
            covered, quality = mode_report(samples)
            swd = sliced_wasserstein(samples, eval_real)
            history.append((step, critic_gap, swd))
            print(f"    G step {step:4d} | critic W estimate {critic_gap:7.4f} | sliced W {swd:.3f} | modes {covered}/8 | on-mode {quality:.2f}")
    return generator_net, critic, history


def power_iteration(weight, steps, generator):
    u = torch.randn(weight.shape[0], generator=generator)
    for _ in range(steps):
        v = weight.T @ u
        v = v / v.norm()
        u = weight @ v
        u = u / u.norm()
    return float(u @ weight @ v)


def main():
    torch.set_num_threads(1)
    print("=== Step 1: JS divergence vs Wasserstein distance for two point masses ===")
    print("  P = point mass at 0, Q = point mass at theta")
    for theta in [0.0, 0.5, 1.0, 4.0]:
        js = 0.0 if theta == 0 else np.log(2)
        print(f"  theta {theta:3.1f} | JS {js:.4f} | W1 {abs(theta):.4f}")
    print("  JS is flat (zero gradient) for every theta != 0; W1 shrinks smoothly as Q moves toward P")

    print("\n=== Step 2: WGAN-GP ===")
    gp_generator, gp_critic, gp_history = train_wgan("WGAN-GP (Adam lr 1e-3, betas (0, 0.9), lambda 1, 5 critic steps)", "gp", generator_steps=1500)
    correlation = np.corrcoef([h[1] for h in gp_history[1:]], [h[2] for h in gp_history[1:]])[0, 1]
    print(f"  correlation between the critic's W estimate and the sliced W metric: {correlation:.2f}")
    with torch.no_grad():
        probe = sample_mixture(512, torch.Generator().manual_seed(4)) + 0.5 * torch.randn(512, 2, generator=torch.Generator().manual_seed(5))
    probe.requires_grad_(True)
    grads, = torch.autograd.grad(gp_critic(probe).sum(), probe)
    print(f"  critic gradient norm on probe points: mean {grads.norm(dim=1).mean():.3f}, max {grads.norm(dim=1).max():.3f} (target 1)")

    print("\n=== Step 3: the original WGAN with weight clipping (c = 0.01) ===")
    clip_generator, clip_critic, _ = train_wgan("WGAN-clip (RMSprop lr 5e-4, 5 critic steps)", "clip", generator_steps=1000)
    weights = torch.cat([p.detach().flatten() for name, p in clip_critic.named_parameters() if "weight" in name])
    at_bound = float((weights.abs() > 0.0099).float().mean())
    print(f"  fraction of critic weights sitting at +-0.01: {at_bound:.2f}")

    with torch.no_grad():
        noise = torch.randn(2000, NOISE_DIM, generator=torch.Generator().manual_seed(1))
        gp_samples, clip_samples = gp_generator(noise), clip_generator(noise)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, samples, title in zip(axes, [gp_samples, clip_samples], ["WGAN-GP", "WGAN weight clipping"]):
        ax.scatter(CENTERS[:, 0], CENTERS[:, 1], s=30, color="0.6")
        ax.scatter(samples[:, 0], samples[:, 1], s=1, color="tab:red")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wgan_samples.png", dpi=110)
    plt.close(fig)
    print("  saved wgan_samples.png")

    print("\n=== Use It: spectral normalization as a Lipschitz constraint ===")
    layer = nn.Linear(64, 32)
    weight = layer.weight.detach()
    print(f"  largest singular value: power iteration {power_iteration(weight, 50, torch.Generator().manual_seed(0)):.4f} | "
          f"torch.linalg.matrix_norm(ord=2) {torch.linalg.matrix_norm(weight, ord=2):.4f}")
    normalized = spectral_norm(nn.Linear(64, 32))
    for _ in range(20):
        normalized(torch.randn(1, 64))
    print(f"  after spectral_norm: largest singular value {torch.linalg.matrix_norm(normalized.weight.detach(), ord=2):.4f}")


if __name__ == "__main__":
    main()
