import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch, length, d_model = x.shape
        q, k, v = (t.view(batch, length, self.num_heads, self.d_head).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return self.output((weights @ v).transpose(1, 2).reshape(batch, length, d_model))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.dropout(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, dropout=0.0, pre_norm=True):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, mask=None):
        if self.pre_norm:
            x = x + self.dropout(self.attention(self.norm1(x), mask))
            return x + self.dropout(self.feed_forward(self.norm2(x)))
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class SortingTransformer(nn.Module):
    def __init__(self, num_layers, pre_norm, vocab_size=10, length=10, d_model=64, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position = nn.Parameter(torch.randn(length, d_model) * 0.02)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, num_heads, 4 * d_model, pre_norm=pre_norm)
                                    for _ in range(num_layers))
        self.final_norm = LayerNorm(d_model) if pre_norm else nn.Identity()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens) + self.position
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


def copy_into_library(block, library):
    with torch.no_grad():
        library.self_attn.in_proj_weight.copy_(block.attention.qkv.weight)
        library.self_attn.in_proj_bias.copy_(block.attention.qkv.bias)
        library.self_attn.out_proj.weight.copy_(block.attention.output.weight)
        library.self_attn.out_proj.bias.copy_(block.attention.output.bias)
        library.linear1.weight.copy_(block.feed_forward.up.weight)
        library.linear1.bias.copy_(block.feed_forward.up.bias)
        library.linear2.weight.copy_(block.feed_forward.down.weight)
        library.linear2.bias.copy_(block.feed_forward.down.bias)
        library.norm1.weight.copy_(block.norm1.weight)
        library.norm1.bias.copy_(block.norm1.bias)
        library.norm2.weight.copy_(block.norm2.weight)
        library.norm2.bias.copy_(block.norm2.bias)


def train_sorting(num_layers, pre_norm, generator, device, steps=1000, lr=1e-3):
    torch.manual_seed(0)
    model = SortingTransformer(num_layers, pre_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        tokens = torch.randint(0, 10, (128, 10), generator=generator).to(device)
        loss = F.cross_entropy(model(tokens).reshape(-1, 10), tokens.sort(dim=1).values.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    tokens = torch.randint(0, 10, (2000, 10), generator=generator).to(device)
    with torch.no_grad():
        predictions = model(tokens).argmax(dim=-1)
    target = tokens.sort(dim=1).values
    return (predictions == target).float().mean().item(), (predictions == target).all(dim=1).float().mean().item(), losses


def gradient_norms_by_layer(num_layers, pre_norm, generator):
    torch.manual_seed(0)
    model = SortingTransformer(num_layers, pre_norm)
    tokens = torch.randint(0, 10, (64, 10), generator=generator)
    F.cross_entropy(model(tokens).reshape(-1, 10), tokens.sort(dim=1).values.reshape(-1)).backward()
    return [block.attention.qkv.weight.grad.norm().item() for block in model.blocks]


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(0)
    d_model, num_heads, d_hidden = 64, 4, 256

    x = torch.randn(4, 10, d_model, generator=generator)
    norm = LayerNorm(d_model)
    print(f"Your LayerNorm vs nn.LayerNorm: {(norm(x) - nn.LayerNorm(d_model)(x)).abs().max():.1e}")
    normalized = norm(x)
    print(f"  after normalization: mean {normalized.mean(-1).abs().max():.1e}, std {normalized.std(-1, unbiased=False).mean():.3f}")

    block = TransformerBlock(d_model, num_heads, d_hidden)
    counts = {name: sum(p.numel() for p in module.parameters())
              for name, module in [("attention", block.attention), ("feed-forward", block.feed_forward),
                                   ("layer norms", nn.ModuleList([block.norm1, block.norm2]))]}
    total = sum(counts.values())
    print(f"\nOne block (d_model={d_model}, heads={num_heads}, d_ff={d_hidden}): {total:,} parameters")
    for name, count in counts.items():
        print(f"  {name:<13} {count:>7,} ({count / total:.0%})")

    for pre_norm in [True, False]:
        mine = TransformerBlock(d_model, num_heads, d_hidden, pre_norm=pre_norm).eval()
        library = nn.TransformerEncoderLayer(d_model, num_heads, d_hidden, dropout=0.0, activation="gelu",
                                             batch_first=True, norm_first=pre_norm).eval()
        copy_into_library(mine, library)
        with torch.no_grad():
            causal = torch.tril(torch.ones(10, 10, dtype=torch.bool))
            diff = (mine(x) - library(x)).abs().max()
            diff_masked = (mine(x, causal) - library(x, src_mask=~causal)).abs().max()
        label = "pre-norm " if pre_norm else "post-norm"
        print(f"{label} block vs nn.TransformerEncoderLayer(norm_first={pre_norm}): {diff:.1e}, with causal mask {diff_masked:.1e}")

    print("\nGradient norm of each block's attention weights at initialization (12 blocks):")
    pre = gradient_norms_by_layer(12, True, generator)
    post = gradient_norms_by_layer(12, False, generator)
    print(f"  {'block':>5} {'pre-norm':>10} {'post-norm':>10}")
    for index in [0, 3, 7, 11]:
        print(f"  {index + 1:>5} {pre[index]:>10.2e} {post[index]:>10.2e}")

    print("\nTask: sort 10 digits. Token accuracy / exact-match accuracy after 1000 steps:")
    print(f"  {'model':<22} {'token acc':>10} {'exact':>7} {'final loss':>11}")
    for name, num_layers, pre_norm in [("1 block, pre-norm", 1, True), ("2 blocks, pre-norm", 2, True),
                                       ("2 blocks, post-norm", 2, False)]:
        token_accuracy, exact, losses = train_sorting(num_layers, pre_norm, generator, device, steps=1000)
        print(f"  {name:<22} {token_accuracy:>10.3f} {exact:>7.3f} {sum(losses[-50:]) / 50:>11.4f}")


if __name__ == "__main__":
    main()
