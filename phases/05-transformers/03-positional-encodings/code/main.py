import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_encoding(length, d_model, base=10000.0):
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = base ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    angles = positions * frequencies
    encoding = torch.zeros(length, d_model)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding


def rope_angles(length, d_head, base=10000.0):
    frequencies = base ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
    return torch.arange(length, dtype=torch.float32)[:, None] * frequencies


def apply_rope(x, angles):
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = torch.cos(angles), torch.sin(angles)
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope, max_length):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.use_rope = use_rope
        self.register_buffer("angles", rope_angles(max_length, self.d_head))

    def forward(self, x):
        batch, length, d_model = x.shape
        q, k, v = (t.view(batch, length, self.num_heads, self.d_head).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        if self.use_rope:
            q, k = apply_rope(q, self.angles[:length]), apply_rope(k, self.angles[:length])
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1)
        return self.output((weights @ v).transpose(1, 2).reshape(batch, length, d_model))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, use_rope, max_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, use_rope, max_length)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, kind, vocab_size=10, length=12, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(vocab_size, d_model)
        if kind == "learned":
            self.position = nn.Parameter(torch.randn(length, d_model) * 0.02)
        elif kind == "sinusoidal":
            self.register_buffer("position", sinusoidal_encoding(length, d_model))
        self.blocks = nn.ModuleList(Block(d_model, num_heads, kind == "rope", length) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        if self.kind in ("learned", "sinusoidal"):
            x = x + self.position[:tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def train_reversal(kind, generator, device, steps=800, length=12):
    torch.manual_seed(0)
    model = Encoder(kind, length=length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        tokens = torch.randint(0, 10, (128, length), generator=generator).to(device)
        loss = F.cross_entropy(model(tokens).reshape(-1, 10), tokens.flip(1).reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tokens = torch.randint(0, 10, (2000, length), generator=generator).to(device)
    with torch.no_grad():
        return (model(tokens).argmax(dim=-1) == tokens.flip(1)).float().mean().item()


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(0)

    encoding = sinusoidal_encoding(64, 32)
    print(f"Sinusoidal encoding: {tuple(encoding.shape)}, values in [{encoding.min():.1f}, {encoding.max():.1f}]")
    print(f"Wavelengths range from 2*pi = {2 * math.pi:.1f} to 2*pi*10000^(30/32) = "
          f"{2 * math.pi * 10000 ** (30 / 32):,.0f} positions")
    dots = encoding @ encoding.T
    print("Dot product PE(p) . PE(p + k) depends only on k:")
    for k in [1, 4, 16]:
        values = [dots[p, p + k].item() for p in (0, 10, 30)]
        print(f"  k={k:>2}: p=0 -> {values[0]:.3f}, p=10 -> {values[1]:.3f}, p=30 -> {values[2]:.3f}")
    frequencies = 10000 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32)
    shift = 5
    rotation = torch.zeros(32, 32)
    for index, frequency in enumerate(frequencies):
        c, s = math.cos(shift * frequency), math.sin(shift * frequency)
        rotation[2 * index:2 * index + 2, 2 * index:2 * index + 2] = torch.tensor([[c, -s], [s, c]])
    shifted = encoding[:-shift] @ rotation
    print(f"PE(p + 5) = PE(p) @ R_5 for a fixed rotation matrix R_5: max error "
          f"{(shifted - encoding[shift:]).abs().max():.1e}")

    x = torch.randn(1, 6, 64, generator=generator)
    block = Block(64, 4, False, 12)
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    with torch.no_grad():
        equivariant = torch.allclose(block(x[:, permutation]), block(x)[:, permutation], atol=1e-5)
    print(f"\nWithout positions, a transformer block is permutation equivariant: {equivariant}")
    print("So it cannot tell 'abc' from 'cba'. A task that needs order: reverse a 12-token sequence.")

    print(f"\n  {'encoding':<11} {'token accuracy':>15} {'seconds':>8}")
    for kind in ["none", "sinusoidal", "learned", "rope"]:
        start = time.perf_counter()
        accuracy = train_reversal(kind, generator, device)
        print(f"  {kind:<11} {accuracy:>15.3f} {time.perf_counter() - start:>8.1f}")

    angles = rope_angles(32, 16)
    q = torch.randn(16, generator=generator)
    k = torch.randn(16, generator=generator)
    print("\nRoPE: score between q at position m and k at position n depends only on m - n:")
    for m, n in [(3, 1), (10, 8), (25, 23), (5, 9), (20, 24)]:
        score = apply_rope(q, angles[m]) @ apply_rope(k, angles[n])
        print(f"  m={m:>2}, n={n:>2}, m-n={m - n:>3}: score {score:.4f}")
    print(f"RoPE preserves vector norms: |q| = {q.norm():.4f}, |rope(q, 17)| = {apply_rope(q, angles[17]).norm():.4f}")

    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.imshow(encoding.T, aspect="auto", cmap="RdBu")
    axis.set_xlabel("position")
    axis.set_ylabel("dimension")
    axis.set_title("Sinusoidal positional encoding")
    figure.tight_layout()
    output_path = Path(__file__).parent / "sinusoidal.png"
    figure.savefig(output_path, dpi=100)
    print(f"\nSaved {output_path.name}")


if __name__ == "__main__":
    main()
