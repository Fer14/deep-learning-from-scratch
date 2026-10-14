import os
import time
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


def conv_transpose2d_scratch(x, weight, stride, padding):
    batch, channels, height, width = x.shape
    kernel = weight.shape[-1]
    dilated = torch.zeros(batch, channels, (height - 1) * stride + 1, (width - 1) * stride + 1, dtype=x.dtype)
    dilated[:, :, ::stride, ::stride] = x
    border = kernel - 1 - padding
    dilated = F.pad(dilated, (border, border, border, border))
    flipped = weight.flip(-1, -2).transpose(0, 1)
    return F.conv2d(dilated, flipped)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(NOISE_DIM, 64 * 2 * 2)
        self.norm0 = nn.BatchNorm2d(64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.up2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.norm0(self.project(z).view(-1, 64, 2, 2)))
        h = F.relu(self.norm1(self.up1(h)))
        return torch.tanh(self.up2(h))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.out = nn.Linear(64 * 2 * 2, 1)

    def forward(self, x):
        h = F.leaky_relu(self.down1(x), 0.2)
        h = F.leaky_relu(self.norm2(self.down2(h)), 0.2)
        return self.out(h.flatten(1)).squeeze(1)


def dcgan_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    if isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class MLPGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NOISE_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(-1, 1, 8, 8)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(1)


def train_gan(generator_net, discriminator_net, images, steps, device, seed):
    torch.manual_seed(seed)
    opt_g = torch.optim.Adam(generator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    batch_size = 128
    for step in range(steps):
        real = images[torch.randint(len(images), (batch_size,), device=device)]
        fake = generator_net(torch.randn(batch_size, NOISE_DIM, device=device))
        loss_d = F.softplus(-discriminator_net(real)).mean() + F.softplus(discriminator_net(fake.detach())).mean()
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        loss_g = F.softplus(-discriminator_net(fake)).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"    step {step:4d} | loss D {loss_d.item():.3f} | loss G {loss_g.item():.3f}")
    return generator_net


def evaluate(generator_net, classifier, device, n=1000):
    generator_net.eval()
    with torch.no_grad():
        samples = generator_net(torch.randn(n, NOISE_DIM, device=device, generator=torch.Generator(device).manual_seed(3)))
    generator_net.train()
    pixels = ((samples.flatten(1).cpu().numpy() + 1) / 2).clip(0, 1)
    probabilities = classifier.predict_proba(pixels)
    counts = np.bincount(probabilities.argmax(axis=1), minlength=10)
    return pixels, float(np.mean(probabilities.max(axis=1) > 0.9)), counts


def save_grid(images, path, title):
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
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
    print(f"device {device} | {len(pixels)} digits | reference classifier confident (>0.9) on "
          f"{np.mean(classifier.predict_proba(pixels).max(axis=1) > 0.9):.2f} of real digits")
    images = torch.tensor(pixels * 2 - 1).view(-1, 1, 8, 8).to(device)

    print("\n=== Step 1: transposed convolution from scratch ===")
    x = torch.randn(2, 3, 2, 2)
    weight = torch.randn(3, 5, 4, 4)
    ours = conv_transpose2d_scratch(x, weight, stride=2, padding=1)
    library = F.conv_transpose2d(x, weight, stride=2, padding=1)
    print(f"  input {tuple(x.shape)} -> output {tuple(ours.shape)} | max diff vs F.conv_transpose2d {(ours - library).abs().max():.2e}")
    print(f"  output size = (in - 1) * stride - 2 * padding + kernel = (2 - 1) * 2 - 2 + 4 = {ours.shape[-1]}")

    print("\n=== Step 2: the DCGAN pair ===")
    generator_net = Generator().to(device)
    discriminator_net = Discriminator().to(device)
    generator_net.apply(dcgan_init)
    discriminator_net.apply(dcgan_init)
    with torch.no_grad():
        shape = generator_net(torch.randn(4, NOISE_DIM, device=device)).shape
    print(f"  G: z {NOISE_DIM} -> 64x2x2 -> 32x4x4 -> {tuple(shape[1:])} | params {sum(p.numel() for p in generator_net.parameters()):,}")
    print(f"  D: 1x8x8 -> 32x4x4 -> 64x2x2 -> logit | params {sum(p.numel() for p in discriminator_net.parameters()):,}")

    print("\n=== Step 3: train the DCGAN ===")
    start = time.perf_counter()
    train_gan(generator_net, discriminator_net, images, steps=3000, device=device, seed=0)
    dcgan_pixels, dcgan_confident, dcgan_counts = evaluate(generator_net, classifier, device)
    print(f"  trained in {time.perf_counter() - start:.1f}s")

    print("\n=== Step 4: compare with an MLP GAN of similar size ===")
    mlp_g, mlp_d = MLPGenerator().to(device), MLPDiscriminator().to(device)
    print(f"  MLP G params {sum(p.numel() for p in mlp_g.parameters()):,} | MLP D params {sum(p.numel() for p in mlp_d.parameters()):,}")
    train_gan(mlp_g, mlp_d, images, steps=3000, device=device, seed=0)
    _, mlp_confident, mlp_counts = evaluate(mlp_g, classifier, device)

    print("\n=== Step 5: score the samples ===")
    print(f"  DCGAN   confident samples {dcgan_confident:.2f} | class counts {dcgan_counts.tolist()}")
    print(f"  MLP GAN confident samples {mlp_confident:.2f} | class counts {mlp_counts.tolist()}")
    save_grid(dcgan_pixels[:64], OUT_DIR / "dcgan_samples.png", "DCGAN samples")
    print("  saved dcgan_samples.png")

    print("\n=== Use It: the generator as torchvision would build it ===")
    sequential = nn.Sequential(
        nn.Unflatten(1, (NOISE_DIM, 1, 1)),
        nn.ConvTranspose2d(NOISE_DIM, 64, kernel_size=2, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), nn.Tanh(),
    )
    print(f"  all-convolutional generator output {tuple(sequential(torch.randn(2, NOISE_DIM)).shape)} "
          f"(the first layer turns a 1x1 'image' of noise into 2x2)")


if __name__ == "__main__":
    main()
