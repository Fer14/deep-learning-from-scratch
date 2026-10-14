import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit

SUBJECTS = ["the keeper", "the baker", "the boy", "a fisherman", "the old woman", "her daughter", "a sailor"]
VERBS = ["watched", "cleaned", "counted", "painted", "carried", "found", "followed"]
OBJECTS = ["the lamp", "the boats", "the stairs", "a notebook", "the bread", "the gulls", "the stars"]
PLACES = ["by the harbor", "on the hill", "at the top", "near the rocks", "in the town", "under the sky"]
TIMES = ["in the morning", "at night", "before the storm", "after the rain", "every day"]


def entropy_bits(probabilities):
    p = np.array(probabilities, dtype=float)
    return float(-(p * np.log(p)).sum())


def source_entropy_per_char():
    optional_place = entropy_bits([0.6, 0.4]) + 0.6 * math.log(len(PLACES))
    optional_time = entropy_bits([0.4, 0.6]) + 0.4 * math.log(len(TIMES))
    sentence_entropy = 3 * math.log(7) + optional_place + optional_time
    mean_length = (
        np.mean([len(s) for s in SUBJECTS]) + np.mean([len(v) for v in VERBS]) + np.mean([len(o) for o in OBJECTS]) + 2
        + 0.6 * (np.mean([len(p) for p in PLACES]) + 1) + 0.4 * (np.mean([len(t) for t in TIMES]) + 1) + 2
    )
    return sentence_entropy / mean_length


def generate_corpus(num_sentences, seed):
    rng = np.random.default_rng(seed)
    sentences = []
    for _ in range(num_sentences):
        parts = [rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(OBJECTS)]
        if rng.random() < 0.6:
            parts.append(rng.choice(PLACES))
        if rng.random() < 0.4:
            parts.append(rng.choice(TIMES))
        sentences.append(" ".join(parts) + ".")
    return " ".join(sentences)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x):
        batch, length, embed_dim = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(batch, length, embed_dim)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, block_size) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets):
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_final(x))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    def non_embedding_params(self):
        embedding = self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        return sum(p.numel() for p in self.parameters()) - embedding


def get_batch(data, block_size, batch_size, device, generator):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


def warmup_cosine_lr(step, max_lr, warmup_steps, total_steps, min_ratio=0.1):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, data, block_size, device):
    model.eval()
    generator = torch.Generator().manual_seed(123)
    losses = [model(*get_batch(data, block_size, 64, device, generator)).item() for _ in range(8)]
    model.train()
    return float(np.mean(losses))


def train(config, train_data, val_data, vocab_size, steps, block_size, batch_size, device):
    torch.manual_seed(0)
    model = GPT(vocab_size, block_size, config["embed_dim"], config["num_heads"], config["num_layers"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    generator = torch.Generator().manual_seed(1)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = warmup_cosine_lr(step, config["lr"], warmup_steps=steps // 10, total_steps=steps)
        loss = model(*get_batch(train_data, block_size, batch_size, device, generator))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, evaluate(model, val_data, block_size, device)


def power_law(n, scale, exponent, irreducible):
    return scale * n ** (-exponent) + irreducible


def gradient_accumulation_check(device):
    torch.manual_seed(0)
    model = GPT(30, 16, 32, 2, 1).to(device)
    ids = torch.randint(0, 30, (32, 17), device=device)
    x, y = ids[:, :-1], ids[:, 1:]
    model.zero_grad()
    model(x, y).backward()
    full_grad = model.head.weight.grad.clone()
    model.zero_grad()
    micro_batches = 4
    for xm, ym in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        (model(xm, ym) / micro_batches).backward()
    return (full_grad - model.head.weight.grad).abs().max().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = generate_corpus(40000, seed=0)
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
    cut = int(0.95 * len(data))
    train_data, val_data = data[:cut], data[cut:]
    print(f"device {device} | synthetic corpus {len(text):,} characters | vocab {len(chars)}")
    print(f"example: {text[:150]!r}")

    print("\nstep 1: warmup + cosine schedule")
    for step in [0, 49, 100, 250, 400, 499]:
        print(f"  step {step:3d}: lr {warmup_cosine_lr(step, 3e-3, 50, 500):.5f}")

    print("\nstep 2: gradient accumulation gives the same gradient as one big batch")
    print(f"  max |full batch grad - 4 accumulated micro-batch grads| = {gradient_accumulation_check(device):.2e}")

    print("\nstep 3: train a family of models with the same data budget")
    configs = [
        {"embed_dim": 8, "num_heads": 1, "num_layers": 1, "lr": 1e-2},
        {"embed_dim": 16, "num_heads": 2, "num_layers": 1, "lr": 1e-2},
        {"embed_dim": 24, "num_heads": 2, "num_layers": 2, "lr": 6e-3},
        {"embed_dim": 32, "num_heads": 2, "num_layers": 2, "lr": 5e-3},
        {"embed_dim": 48, "num_heads": 4, "num_layers": 3, "lr": 4e-3},
        {"embed_dim": 64, "num_heads": 4, "num_layers": 4, "lr": 3e-3},
    ]
    steps, block_size, batch_size = 400, 32, 32
    tokens_seen = steps * block_size * batch_size
    results = []
    for config in configs:
        start = time.time()
        model, val_loss = train(config, train_data, val_data, len(chars), steps, block_size, batch_size, device)
        params = model.non_embedding_params()
        flops = 6 * params * tokens_seen
        results.append((params, val_loss))
        print(f"  d={config['embed_dim']:2d} L={config['num_layers']} | N={params:7,} | tokens/param {tokens_seen / params:6.1f} | compute {flops:.2e} FLOPs | val loss {val_loss:.3f} | {time.time() - start:4.1f}s")

    print("\nstep 4: fit L(N) = a * N^(-alpha) + L_inf")
    n_values = np.array([r[0] for r in results], dtype=float)
    losses = np.array([r[1] for r in results])
    irreducible = source_entropy_per_char()
    print(f"  L_inf from the grammar itself: entropy {irreducible:.3f} nats per character")
    (scale, exponent), _ = curve_fit(lambda n, a, b: power_law(n, a, b, irreducible), n_values, losses, p0=(10.0, 0.3), maxfev=20000)
    print(f"  a = {scale:.2f}, alpha = {exponent:.3f}")
    free_fit, _ = curve_fit(power_law, n_values, losses, p0=(10.0, 0.3, irreducible), bounds=([0, 0, 0], [1e4, 3, 3]))
    print(f"  fitting L_inf too: a = {free_fit[0]:.2f}, alpha = {free_fit[1]:.3f}, L_inf = {free_fit[2]:.3f}")
    for n in [n_values[-1] * 4, n_values[-1] * 16]:
        print(f"  extrapolated loss at N={int(n):,}: {power_law(n, scale, exponent, irreducible):.3f}")

    print("\nuse it: PyTorch schedulers and autocast")
    probe = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_cosine_lr(s, 1.0, 50, 500))
    library_lrs = []
    for _ in range(500):
        library_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    ours = [warmup_cosine_lr(s, 3e-3, 50, 500) for s in range(500)]
    print(f"  LambdaLR vs hand schedule, max difference {max(abs(a - b) for a, b in zip(library_lrs, ours)):.2e}")
    model = GPT(len(chars), block_size, 32, 2, 2).to(device)
    x, y = get_batch(val_data, block_size, 8, device, torch.Generator().manual_seed(0))
    with torch.no_grad():
        full = model(x, y).item()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            low = model(x, y).item()
    print(f"  loss in float32 {full:.4f} vs bfloat16 autocast {low:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(n_values, losses - irreducible, "o", label="measured (L - L_inf)")
    grid = np.logspace(np.log10(n_values[0]), np.log10(n_values[-1] * 16), 50)
    ax.loglog(grid, power_law(grid, scale, exponent, 0.0), "--", label=f"fit alpha={exponent:.2f}")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("reducible validation loss")
    ax.set_title("A tiny scaling law")
    ax.legend()
    fig.tight_layout()
    out_path = Path(__file__).parent / "scaling_law.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved {out_path.name}")


if __name__ == "__main__":
    main()
