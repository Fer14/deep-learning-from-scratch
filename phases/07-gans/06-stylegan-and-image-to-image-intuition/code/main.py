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
LATENT = 32


def adain(x, style_scale, style_bias, eps=1e-5):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + eps).sqrt()
    return style_scale[:, :, None, None] * (x - mean) / std + style_bias[:, :, None, None]


class StyleBlock(nn.Module):
    def __init__(self, n_in, n_out, upsample):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=3, padding=1)
        self.noise_strength = nn.Parameter(torch.zeros(1, n_out, 1, 1))
        self.style = nn.Linear(LATENT, 2 * n_out)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = x + self.noise_strength * torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = F.leaky_relu(x, 0.2)
        scale, bias = self.style(w).chunk(2, dim=1)
        return adain(x, 1 + scale, bias)


class StyleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(LATENT, LATENT), nn.LeakyReLU(0.2), nn.Linear(LATENT, LATENT),
                                     nn.LeakyReLU(0.2), nn.Linear(LATENT, LATENT))
        self.constant = nn.Parameter(torch.randn(1, 32, 2, 2))
        self.blocks = nn.ModuleList([StyleBlock(32, 32, False), StyleBlock(32, 32, True), StyleBlock(32, 16, True)])
        self.to_image = nn.Conv2d(16, 1, kernel_size=1)

    def synthesize(self, ws):
        x = self.constant.expand(len(ws[0]), -1, -1, -1)
        for block, w in zip(self.blocks, ws):
            x = block(x, w)
        return torch.tanh(self.to_image(x))

    def forward(self, z):
        w = self.mapping(z)
        return self.synthesize([w, w, w])


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, minibatch_std=False):
        super().__init__()
        self.minibatch_std = minibatch_std
        self.features = nn.Sequential(nn.Conv2d(in_channels, 32, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(32, 64, 4, 2, 1),
                                      nn.LeakyReLU(0.2))
        self.out = nn.Linear((64 + int(minibatch_std)) * 2 * 2, 1)

    def forward(self, x):
        h = self.features(x)
        if self.minibatch_std:
            spread = h.std(dim=0).mean().expand(len(h), 1, 2, 2)
            h = torch.cat([h, spread], dim=1)
        return self.out(h.flatten(1)).squeeze(1)


def train_style_gan(images, steps, device, seed):
    torch.manual_seed(seed)
    generator_net, discriminator_net = StyleGenerator().to(device), Discriminator(minibatch_std=True).to(device)
    opt_g = torch.optim.Adam(generator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    for step in range(steps):
        real = images[torch.randint(len(images), (128,), device=device)]
        fake = generator_net(torch.randn(128, LATENT, device=device))
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
    return generator_net.eval()


class Translator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, low_res):
        upsampled = F.interpolate(low_res, scale_factor=2, mode="nearest")
        return torch.tanh(self.net(upsampled) + upsampled)


def train_translator(low, high, steps, device, seed, adversarial_weight):
    torch.manual_seed(seed)
    translator = Translator().to(device)
    discriminator_net = Discriminator(in_channels=2).to(device)
    opt_t = torch.optim.Adam(translator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator_net.parameters(), lr=1e-3, betas=(0.5, 0.999))
    for step in range(steps):
        index = torch.randint(len(low), (128,), device=device)
        condition, target = low[index], high[index]
        condition_maps = F.interpolate(condition, scale_factor=2, mode="nearest")
        output = translator(condition)
        if adversarial_weight > 0:
            real_pair = torch.cat([condition_maps, target], dim=1)
            fake_pair = torch.cat([condition_maps, output.detach()], dim=1)
            loss_d = F.softplus(-discriminator_net(real_pair)).mean() + F.softplus(discriminator_net(fake_pair)).mean()
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
        loss_t = (output - target).abs().mean()
        if adversarial_weight > 0:
            loss_t = loss_t + adversarial_weight * F.softplus(-discriminator_net(torch.cat([condition_maps, output], dim=1))).mean()
        opt_t.zero_grad()
        loss_t.backward()
        opt_t.step()
    return translator.eval()


def to_pixels(x):
    return ((x.flatten(1).detach().cpu().numpy() + 1) / 2).clip(0, 1)


def sharpness(pixels):
    images = pixels.reshape(-1, 8, 8)
    return float(np.abs(np.diff(images, axis=1)).mean() + np.abs(np.diff(images, axis=2)).mean())


def save_mixing_grid(rows, path):
    fig, axes = plt.subplots(len(rows), len(rows[0][1]), figsize=(len(rows[0][1]) * 0.9, len(rows) * 1.0))
    for r, (name, images) in enumerate(rows):
        for c, image in enumerate(images):
            axes[r, c].imshow(image.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=7, loc="left")
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
    print(f"device {device}")

    print("\n=== Step 1: AdaIN from scratch ===")
    x = torch.randn(4, 8, 5, 5)
    scale, bias = torch.randn(4, 8), torch.randn(4, 8)
    ours = adain(x, scale, bias)
    library = F.instance_norm(x) * scale[:, :, None, None] + bias[:, :, None, None]
    print(f"  max diff vs F.instance_norm * scale + bias: {(ours - library).abs().max():.2e}")
    print(f"  per-channel mean after AdaIN equals the style bias: {torch.allclose(ours.mean(dim=(2, 3)), bias, atol=1e-5)}")

    print("\n=== Step 2: a tiny style-based generator ===")
    generator_net = train_style_gan(images, steps=3000, device=device, seed=0)
    with torch.no_grad():
        samples = to_pixels(generator_net(torch.randn(1000, LATENT, device=device, generator=torch.Generator(device).manual_seed(1))))
    probabilities = classifier.predict_proba(samples)
    print(f"  confident samples {np.mean(probabilities.max(1) > 0.9):.2f} | class counts {np.bincount(probabilities.argmax(1), minlength=10).tolist()}")

    print("\n=== Step 3: style mixing (coarse layers from A, fine layer from B) ===")
    rng = torch.Generator(device).manual_seed(2)
    with torch.no_grad():
        w_a = generator_net.mapping(torch.randn(32, LATENT, device=device, generator=rng))
        w_b = generator_net.mapping(torch.randn(32, LATENT, device=device, generator=rng))
        torch.manual_seed(9)
        image_a = to_pixels(generator_net.synthesize([w_a, w_a, w_a]))
        torch.manual_seed(9)
        image_b = to_pixels(generator_net.synthesize([w_b, w_b, w_b]))
        torch.manual_seed(9)
        coarse_a = to_pixels(generator_net.synthesize([w_a, w_a, w_b]))
        torch.manual_seed(9)
        coarse_b = to_pixels(generator_net.synthesize([w_b, w_b, w_a]))
    class_a, class_b = classifier.predict(image_a), classifier.predict(image_b)
    differ = class_a != class_b
    print(f"  pairs where A and B are different digits: {int(differ.sum())}/32")
    print(f"  'coarse from A, fine from B' classified as A's digit: {np.mean(classifier.predict(coarse_a)[differ] == class_a[differ]):.2f}"
          f" | as B's digit: {np.mean(classifier.predict(coarse_a)[differ] == class_b[differ]):.2f}")
    print(f"  'coarse from B, fine from A' classified as B's digit: {np.mean(classifier.predict(coarse_b)[differ] == class_b[differ]):.2f}")
    save_mixing_grid([("A", image_a[:10]), ("B", image_b[:10]), ("coarse A + fine B", coarse_a[:10]), ("coarse B + fine A", coarse_b[:10])],
                     OUT_DIR / "style_mixing.png")

    print("\n=== Step 4: the truncation trick ===")
    with torch.no_grad():
        w_mean = generator_net.mapping(torch.randn(5000, LATENT, device=device, generator=torch.Generator(device).manual_seed(3))).mean(0)
        w = generator_net.mapping(torch.randn(1000, LATENT, device=device, generator=torch.Generator(device).manual_seed(4)))
        for psi in [1.0, 0.7, 0.4]:
            truncated = to_pixels(generator_net.synthesize([w_mean + psi * (w - w_mean)] * 3))
            confidence = classifier.predict_proba(truncated).max(1)
            print(f"  psi {psi:.1f} | confident samples {np.mean(confidence > 0.9):.2f} | per-pixel std across samples {truncated.std(0).mean():.3f} | "
                  f"distinct predicted classes {len(np.unique(classifier.predict(truncated)))}")

    print("\n=== Step 5: image-to-image, 4x4 -> 8x8 super-resolution (pix2pix-style) ===")
    low = F.avg_pool2d(images, 2)
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(0)).to(device)
    train_index, test_index = order[300:], order[:300]
    results = {}
    for name, weight in [("L1 only", 0.0), ("L1 + adversarial", 0.05)]:
        translator = train_translator(low[train_index], images[train_index], steps=800, device=device, seed=0, adversarial_weight=weight)
        with torch.no_grad():
            results[name] = translator(low[test_index])
    truth = to_pixels(images[test_index])
    nearest = to_pixels(F.interpolate(low[test_index], scale_factor=2, mode="nearest"))
    test_labels = digits.target[test_index.cpu().numpy()]
    print(f"  {'method':18s} | L1 to truth | sharpness | classifier accuracy")
    for name, output in [("nearest upsample", nearest), ("truth", truth)] + [(k, to_pixels(v)) for k, v in results.items()]:
        print(f"  {name:18s} | {np.abs(output - truth).mean():11.4f} | {sharpness(output):9.4f} | {np.mean(classifier.predict(output) == test_labels):.3f}")
    save_mixing_grid([("input 4x4 (upsampled)", nearest[:10]), ("L1 only", to_pixels(results["L1 only"])[:10]),
                      ("L1 + adversarial", to_pixels(results["L1 + adversarial"])[:10]), ("truth", truth[:10])],
                     OUT_DIR / "super_resolution.png")
    print("  saved style_mixing.png and super_resolution.png")


if __name__ == "__main__":
    main()
