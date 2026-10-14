import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from torch import nn

OUT_DIR = Path(__file__).parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
NOISE_DIM = 32
N_CLASSES = 10


class ConditionalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(NOISE_DIM + N_CLASSES, 64 * 2 * 2)
        self.norm0 = nn.BatchNorm2d(64)
        self.up1 = nn.ConvTranspose2d(64 + N_CLASSES, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.up2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, N_CLASSES).float()
        h = F.relu(self.norm0(self.project(torch.cat([z, one_hot], dim=1)).view(-1, 64, 2, 2)))
        label_maps = one_hot[:, :, None, None].expand(-1, -1, 2, 2)
        h = F.relu(self.norm1(self.up1(torch.cat([h, label_maps], dim=1))))
        return torch.tanh(self.up2(h))


class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1 + N_CLASSES, 32, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.out = nn.Linear(64 * 2 * 2, 1)

    def forward(self, x, labels):
        label_maps = F.one_hot(labels, N_CLASSES).float()[:, :, None, None].expand(-1, -1, 8, 8)
        h = F.leaky_relu(self.down1(torch.cat([x, label_maps], dim=1)), 0.2)
        h = F.leaky_relu(self.down2(h), 0.2)
        return self.out(h.flatten(1)).squeeze(1)


def train_cgan(images, labels, steps, device, seed):
    torch.manual_seed(seed)
    generator_net, discriminator_net = ConditionalGenerator().to(device), ConditionalDiscriminator().to(device)
    opt_g = torch.optim.Adam(generator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    batch_size = 128
    for step in range(steps):
        index = torch.randint(len(images), (batch_size,), device=device)
        real, real_labels = images[index], labels[index]
        fake_labels = torch.randint(N_CLASSES, (batch_size,), device=device)
        fake = generator_net(torch.randn(batch_size, NOISE_DIM, device=device), fake_labels)
        loss_d = (F.softplus(-discriminator_net(real, real_labels)).mean()
                  + F.softplus(discriminator_net(fake.detach(), fake_labels)).mean())
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        loss_g = F.softplus(-discriminator_net(fake, fake_labels)).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"    step {step:4d} | loss D {loss_d.item():.3f} | loss G {loss_g.item():.3f}")
    return generator_net, discriminator_net


def to_pixels(samples):
    return ((samples.flatten(1).cpu().numpy() + 1) / 2).clip(0, 1)


def save_grid(images, path, title):
    fig, axes = plt.subplots(10, 10, figsize=(6, 6))
    for ax, image in zip(axes.flat, images):
        ax.imshow(image.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    digits = load_digits()
    pixels = (digits.data / 16.0).astype(np.float32)
    classifier = LogisticRegression(max_iter=3000).fit(pixels, digits.target)
    images = torch.tensor(pixels * 2 - 1).view(-1, 1, 8, 8).to(device)
    labels = torch.tensor(digits.target).to(device)
    print(f"device {device} | reference classifier accuracy on its own training digits {classifier.score(pixels, digits.target):.3f}")

    print("\n=== Step 1: how the label enters each network ===")
    z = torch.randn(3, NOISE_DIM)
    example_labels = torch.tensor([0, 7, 7])
    one_hot = F.one_hot(example_labels, N_CLASSES).float()
    print(f"  G input: concat(z {tuple(z.shape)}, one-hot {tuple(one_hot.shape)}) -> {tuple(torch.cat([z, one_hot], 1).shape)}")
    print(f"  D input: concat(image (3, 1, 8, 8), label maps (3, 10, 8, 8)) -> (3, 11, 8, 8)")

    print("\n=== Step 2: train the conditional GAN ===")
    generator_net, discriminator_net = train_cgan(images, labels, steps=3000, device=device, seed=0)
    generator_net.eval()

    print("\n=== Step 3: ask for each digit, check what you got ===")
    rng = torch.Generator(device).manual_seed(3)
    requested = torch.arange(N_CLASSES, device=device).repeat_interleave(100)
    with torch.no_grad():
        samples = to_pixels(generator_net(torch.randn(len(requested), NOISE_DIM, device=device, generator=rng), requested))
    predicted = classifier.predict(samples)
    requested_np = requested.cpu().numpy()
    print(f"  classifier agrees with the requested label on {np.mean(predicted == requested_np):.3f} of 1000 samples")
    per_class = [np.mean(predicted[requested_np == digit] == digit) for digit in range(N_CLASSES)]
    print(f"  per digit: {[round(float(v), 2) for v in per_class]}")
    within_std = np.mean([samples[requested_np == digit].std(axis=0).mean() for digit in range(N_CLASSES)])
    real_std = np.mean([pixels[digits.target == digit].std(axis=0).mean() for digit in range(N_CLASSES)])
    print(f"  mean per-pixel std within a class: generated {within_std:.3f} | real {real_std:.3f}")

    print("\n=== Step 4: same noise, different labels ===")
    shared_noise = torch.randn(10, NOISE_DIM, device=device, generator=torch.Generator(device).manual_seed(5))
    with torch.no_grad():
        rows = [to_pixels(generator_net(shared_noise, torch.full((10,), digit, device=device))) for digit in range(N_CLASSES)]
    grid = np.concatenate(rows)
    swap_accuracy = np.mean(classifier.predict(grid) == np.repeat(np.arange(N_CLASSES), 10))
    print(f"  10 noise vectors x 10 labels: classifier agrees on {swap_accuracy:.2f}")
    save_grid(grid, OUT_DIR / "cgan_grid.png", "rows: requested digit 0-9, columns: shared noise z")
    print("  saved cgan_grid.png")

    print("\n=== Use It: nn.Embedding for the label ===")
    embedding = nn.Embedding(N_CLASSES, 16)
    by_lookup = embedding(example_labels)
    by_one_hot = one_hot @ embedding.weight
    print(f"  nn.Embedding(labels) == one_hot @ weight: {torch.allclose(by_lookup, by_one_hot)}")
    print(f"  a one-hot concat into a Linear layer is the same as adding a learned per-class vector: "
          f"{torch.allclose(torch.cat([z, one_hot], 1) @ torch.cat([torch.eye(NOISE_DIM, 4), embedding.weight[:, :4]]), z[:, :4] + embedding.weight[example_labels, :4])}")


if __name__ == "__main__":
    main()
