from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.linear_model import LogisticRegression

OUT_DIR = Path(__file__).parent
T = 1000


def linear_schedule(n_steps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)


def cosine_schedule(n_steps, s=0.008):
    steps = torch.arange(n_steps + 1, dtype=torch.float64)
    f = torch.cos((steps / n_steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)


def swiss_roll(n, seed):
    points, _ = make_swiss_roll(n, noise=0.3, random_state=seed)
    points = points[:, [0, 2]] / 10.0
    return torch.tensor(points, dtype=torch.float64)


def noise_step_by_step(x0, betas, t, generator):
    x = x0.clone()
    for step in range(t):
        x = torch.sqrt(1 - betas[step]) * x + torch.sqrt(betas[step]) * torch.randn(x.shape, generator=generator, dtype=x.dtype)
    return x


def noise_closed_form(x0, alpha_bar, t, generator):
    eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
    return torch.sqrt(alpha_bar[t - 1]) * x0 + torch.sqrt(1 - alpha_bar[t - 1]) * eps


def describe(x):
    covariance = torch.cov(x.T)
    return (f"mean ({x[:, 0].mean():+.3f}, {x[:, 1].mean():+.3f}) | var ({covariance[0, 0]:.3f}, {covariance[1, 1]:.3f}) | "
            f"cov {covariance[0, 1]:+.3f}")


def save_2d_snapshots(x0, alpha_bar, steps, path):
    generator = torch.Generator().manual_seed(4)
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 2.6))
    for ax, t in zip(axes, steps):
        x = x0 if t == 0 else noise_closed_form(x0, alpha_bar, t, generator)
        ax.scatter(x[:, 0], x[:, 1], s=1, color="tab:blue")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_digit_snapshots(images, schedules, steps, path):
    generator = torch.Generator().manual_seed(5)
    fig, axes = plt.subplots(len(schedules), len(steps), figsize=(1.2 * len(steps), 1.4 * len(schedules)))
    for r, (name, alpha_bar) in enumerate(schedules.items()):
        for c, t in enumerate(steps):
            x = images[:1] if t == 0 else noise_closed_form(images[:1], alpha_bar, t, generator)
            axes[r, c].imshow(x.reshape(8, 8), cmap="gray_r", vmin=-1.5, vmax=1.5)
            axes[r, c].axis("off")
            axes[r, c].set_title(f"{name} t={t}", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    schedules = {"linear": linear_schedule(T), "cosine": cosine_schedule(T)}
    alpha_bars = {name: torch.cumprod(1 - betas, dim=0) for name, betas in schedules.items()}

    print("=== Step 1: noise schedules ===")
    for name, alpha_bar in alpha_bars.items():
        betas = schedules[name]
        print(f"  {name:6s} beta_1 {betas[0]:.5f} beta_T {betas[-1]:.4f} | alpha_bar at t=1 {alpha_bar[0]:.4f}, t=250 {alpha_bar[249]:.4f}, "
              f"t=500 {alpha_bar[499]:.4f}, t=750 {alpha_bar[749]:.4f}, t=T {alpha_bar[-1]:.2e}")

    print("\n=== Step 2: step-by-step chain vs closed form q(x_t | x_0) ===")
    x0 = swiss_roll(5000, seed=0)
    print(f"  data      : {describe(x0)}")
    for t in [100, 400]:
        chain = noise_step_by_step(x0, schedules["linear"], t, torch.Generator().manual_seed(1))
        direct = noise_closed_form(x0, alpha_bars["linear"], t, torch.Generator().manual_seed(2))
        print(f"  t={t:3d} chain  : {describe(chain)}")
        print(f"  t={t:3d} closed : {describe(direct)}")

    print("\n=== Step 3: at t = T the data is gone ===")
    x_T = noise_closed_form(x0, alpha_bars["linear"], T, torch.Generator().manual_seed(3))
    print(f"  x_T       : {describe(x_T)}  (target N(0, I))")
    correlation = np.corrcoef(x0[:, 0], x_T[:, 0])[0, 1]
    print(f"  correlation between x_0 and x_T (first coordinate): {correlation:+.4f}")

    print("\n=== Step 4: signal-to-noise ratio ===")
    for name, alpha_bar in alpha_bars.items():
        snr_db = 10 * torch.log10(alpha_bar / (1 - alpha_bar))
        half = int(torch.nonzero(alpha_bar < 0.5)[0]) + 1
        print(f"  {name:6s} SNR (dB) at t=1 {snr_db[0]:+.1f}, t=250 {snr_db[249]:+.1f}, t=500 {snr_db[499]:+.1f}, "
              f"t=750 {snr_db[749]:+.1f}, t=T {snr_db[-1]:+.1f} | signal and noise balance at t={half}")
    save_2d_snapshots(x0, alpha_bars["linear"], [0, 50, 150, 300, 600, 1000], OUT_DIR / "forward_2d.png")

    print("\n=== Use It: how long does a digit stay recognizable? ===")
    digits = load_digits()
    images = torch.tensor(digits.data / 8.0 - 1.0, dtype=torch.float64)
    order = np.random.default_rng(0).permutation(len(images))
    train_index, test_index = order[300:], order[:300]
    classifier = LogisticRegression(max_iter=3000).fit(images[train_index].numpy(), digits.target[train_index])
    print(f"  classifier trained on clean digits, tested on noised test digits x_t")
    for name, alpha_bar in alpha_bars.items():
        accuracies = []
        for t in [100, 300, 500, 700, 900]:
            noisy = noise_closed_form(images[test_index], alpha_bar, t, torch.Generator().manual_seed(t))
            accuracies.append(f"t={t}: {classifier.score(noisy.numpy(), digits.target[test_index]):.2f}")
        print(f"  {name:6s} | " + " | ".join(accuracies))
    save_digit_snapshots(images, alpha_bars, [0, 100, 300, 500, 700, 900], OUT_DIR / "forward_digits.png")
    print("  saved forward_2d.png and forward_digits.png")


if __name__ == "__main__":
    main()
