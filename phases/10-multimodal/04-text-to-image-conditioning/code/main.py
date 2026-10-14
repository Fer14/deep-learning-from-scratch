import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = ["circle", "square", "triangle"]
COLOURS = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.3, 0.95), "yellow": (0.95, 0.85, 0.1)}
COLOUR_NAMES = list(COLOURS)
SIDES = ["left", "right"]
IMAGE_SIZE = 12
WORDS = ["<null>", "a", "on", "the"] + SHAPES + COLOUR_NAMES + SIDES
WORD_INDEX = {w: i for i, w in enumerate(WORDS)}


def render(shape, colour, side, rng):
    radius = rng.uniform(2.6, 3.2)
    cx = rng.uniform(3.0, 3.8) if side == "left" else rng.uniform(8.2, 9.0)
    cy = rng.uniform(4.0, 8.0)
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE] + 0.5
    if shape == "circle":
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    elif shape == "square":
        mask = (np.abs(x - cx) <= radius * 0.85) & (np.abs(y - cy) <= radius * 0.85)
    else:
        top = cy - radius
        mask = (y >= top) & (y <= cy + radius * 0.8) & (np.abs(x - cx) <= (y - top) * 0.6)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    image[mask] = COLOURS[colour]
    return image.transpose(2, 0, 1)


def make_dataset(num_images, rng):
    images, captions, labels = [], [], []
    for _ in range(num_images):
        shape, colour, side = rng.integers(3), rng.integers(4), rng.integers(2)
        images.append(render(SHAPES[shape], COLOUR_NAMES[colour], SIDES[side], rng))
        captions.append(f"a {COLOUR_NAMES[colour]} {SHAPES[shape]} on the {SIDES[side]}")
        labels.append((colour, shape, side))
    images = torch.tensor(np.array(images), dtype=torch.float32) * 2 - 1
    return images, captions, torch.tensor(labels)


def tokenize(captions):
    return torch.tensor([[WORD_INDEX[w] for w in c.split()] for c in captions])


class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(len(WORDS), embed_dim)
        self.position_embedding = nn.Embedding(6, embed_dim)
        self.mlp = nn.Sequential(nn.Linear(6 * embed_dim, 2 * embed_dim), nn.SiLU(), nn.Linear(2 * embed_dim, embed_dim))

    def forward(self, ids):
        tokens = self.word_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        return self.mlp(tokens.flatten(1)), tokens


def timestep_embedding(t, dim):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class FiLMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = nn.Linear(cond_dim, 2 * out_dim)

    def forward(self, h, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return F.silu(self.linear(h) * (1 + scale) + shift)


class Denoiser(nn.Module):
    def __init__(self, hidden=256, cond_dim=128):
        super().__init__()
        pixels = 3 * IMAGE_SIZE * IMAGE_SIZE
        self.text_encoder = TextEncoder(cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.layers = nn.ModuleList([FiLMLayer(pixels, hidden, cond_dim), FiLMLayer(hidden, hidden, cond_dim), FiLMLayer(hidden, hidden, cond_dim)])
        self.out = nn.Linear(hidden, pixels)
        self.skip = nn.Linear(pixels, pixels)
        self.cond_dim = cond_dim

    def forward(self, x, t, ids):
        text, _ = self.text_encoder(ids)
        cond = self.time_mlp(timestep_embedding(t, self.cond_dim)) + text
        h = x.flatten(1)
        for layer in self.layers:
            h = layer(h, cond)
        return self.out(h).view_as(x) + self.skip(x.flatten(1)).view_as(x)


class Diffusion:
    def __init__(self, steps, device):
        self.steps = steps
        self.betas = torch.linspace(1e-4 * 1000 / steps, 0.02 * 1000 / steps, steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])

    def add_noise(self, x0, t, noise):
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, ids, guidance, generator):
        x = torch.randn(len(ids), 3, IMAGE_SIZE, IMAGE_SIZE, device=ids.device, generator=generator)
        null_ids = torch.zeros_like(ids)
        for step in reversed(range(self.steps)):
            t = torch.full((len(ids),), step, device=ids.device)
            x0_cond = model(x, t, ids)
            x0_uncond = model(x, t, null_ids)
            x0 = (x0_uncond + guidance * (x0_cond - x0_uncond)).clamp(-1, 1)
            alpha, alpha_bar, alpha_bar_prev, beta = self.alphas[step], self.alpha_bars[step], self.alpha_bars_prev[step], self.betas[step]
            mean = (alpha_bar_prev.sqrt() * beta * x0 + alpha.sqrt() * (1 - alpha_bar_prev) * x) / (1 - alpha_bar)
            if step > 0:
                variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                x = mean + variance.sqrt() * torch.randn(x.shape, device=x.device, generator=generator)
            else:
                x = mean
        return x.clamp(-1, 1)


class AttributeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Linear(64 * 9, 128), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, 4), nn.Linear(128, 3), nn.Linear(128, 2)])

    def forward(self, x):
        h = self.net(x)
        return [head(h) for head in self.heads]


def train_classifier(images, labels, device):
    model = AttributeClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(800):
        batch = torch.randint(len(images), (128,))
        noisy = images[batch].to(device) + 0.1 * torch.randn(128, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        outputs = model(noisy)
        loss = sum(F.cross_entropy(o, labels[batch, i].to(device)) for i, o in enumerate(outputs))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def train_denoiser(model, diffusion, images, ids, steps, device, drop_probability):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps + 1):
        batch = torch.randint(len(images), (128,))
        x0, caption = images[batch].to(device), ids[batch].to(device)
        drop = torch.rand(len(batch), device=device) < drop_probability
        caption = caption.masked_fill(drop.unsqueeze(1), 0)
        t = torch.randint(0, diffusion.steps, (len(batch),), device=device)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(model(diffusion.add_noise(x0, t, noise), t, caption), x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"  step {step:4d} | clean-image mse {loss.item():.4f}")
    return model.eval()


@torch.no_grad()
def attribute_accuracy(classifier, images, labels):
    outputs = classifier(images)
    per_attribute = [(o.argmax(1).cpu() == labels[:, i]).float() for i, o in enumerate(outputs)]
    all_correct = per_attribute[0] * per_attribute[1] * per_attribute[2]
    return [p.mean().item() for p in per_attribute] + [all_correct.mean().item()]


def cross_attention(queries, context, w_q, w_k, w_v, w_o, num_heads):
    batch, length, dim = queries.shape
    head_dim = dim // num_heads
    q = (queries @ w_q.T).view(batch, length, num_heads, head_dim).transpose(1, 2)
    k = (context @ w_k.T).view(batch, context.shape[1], num_heads, head_dim).transpose(1, 2)
    v = (context @ w_v.T).view(batch, context.shape[1], num_heads, head_dim).transpose(1, 2)
    weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(head_dim), dim=-1)
    return ((weights @ v).transpose(1, 2).reshape(batch, length, dim)) @ w_o.T


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, captions, labels = make_dataset(6000, rng)
    ids = tokenize(captions)
    print(f"device {device} | {len(images)} images of {IMAGE_SIZE}x{IMAGE_SIZE} | {len(SHAPES) * len(COLOURS) * len(SIDES)} possible captions")
    print(f"example caption {captions[0]!r} -> ids {ids[0].tolist()}")

    print("\nstep 1: an attribute classifier to grade generated images")
    classifier = train_classifier(images, labels, device)
    real_accuracy = attribute_accuracy(classifier, images[-500:].to(device), labels[-500:])
    print(f"  on real images: colour {real_accuracy[0]:.3f} shape {real_accuracy[1]:.3f} side {real_accuracy[2]:.3f}")

    print("\nstep 2: train a text-conditioned denoiser (FiLM) with 10% caption dropout")
    diffusion = Diffusion(100, device)
    model = train_denoiser(Denoiser().to(device), diffusion, images, ids, steps=2000, device=device, drop_probability=0.1)
    print(f"  parameters {sum(p.numel() for p in model.parameters()):,}")

    print("\nstep 3: sample every caption with different guidance scales")
    all_captions = [f"a {c} {s} on the {side}" for c in COLOUR_NAMES for s in SHAPES for side in SIDES]
    all_labels = torch.tensor([(ci, si, di) for ci in range(4) for si in range(3) for di in range(2)])
    prompt_ids = tokenize(all_captions).repeat(8, 1).to(device)
    prompt_labels = all_labels.repeat(8, 1)
    samples = {}
    for guidance in [0.0, 1.0, 3.0, 5.0]:
        generator = torch.Generator(device=device).manual_seed(0)
        generated = diffusion.sample(model, prompt_ids, guidance, generator)
        samples[guidance] = generated
        colour, shape, side, everything = attribute_accuracy(classifier, generated, prompt_labels)
        spread = generated.view(8, 24, -1).std(0).mean().item()
        print(f"  guidance {guidance:3.1f} | colour {colour:.3f} shape {shape:.3f} side {side:.3f} | all three {everything:.3f} | pixel std across seeds {spread:.3f}")

    print("\nuse it: cross-attention conditioning, by hand and with nn.MultiheadAttention")
    _, word_tokens = model.text_encoder(prompt_ids[:4])
    image_tokens = torch.randn(4, 9, 128, device=device)
    attention = nn.MultiheadAttention(128, 4, batch_first=True, bias=False).to(device)
    w_q, w_k, w_v = attention.in_proj_weight.chunk(3)
    with torch.no_grad():
        ours = cross_attention(image_tokens, word_tokens, w_q, w_k, w_v, attention.out_proj.weight, 4)
        library, _ = attention(image_tokens, word_tokens, word_tokens)
    print(f"  9 image tokens attend to 6 word tokens: output {tuple(library.shape)} | max difference {(ours - library).abs().max().item():.2e}")

    fig, axes = plt.subplots(4, 12, figsize=(12, 4.6))
    for row, guidance in enumerate([0.0, 1.0, 3.0, 5.0]):
        for col in range(12):
            index = col * 2
            axes[row, col].imshow(((samples[guidance][index].cpu().permute(1, 2, 0) + 1) / 2).numpy())
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(all_captions[index].replace("a ", "").replace(" on the left", ""), fontsize=7)
        axes[row, 0].text(-8, 6, f"w={guidance}", fontsize=8, ha="right", va="center")
    fig.suptitle("Generated '... on the left' images by guidance scale")
    fig.tight_layout()
    out_path = Path(__file__).parent / "guidance_grid.png"
    fig.savefig(out_path, dpi=110)
    print(f"\nsaved {out_path.name}")


if __name__ == "__main__":
    main()
