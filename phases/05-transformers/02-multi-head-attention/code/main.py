import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x):
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch, length, self.num_heads * self.d_head)

    def forward(self, x, mask=None):
        queries, keys, values = self.qkv(x).chunk(3, dim=-1)
        queries, keys, values = self.split_heads(queries), self.split_heads(keys), self.split_heads(values)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.output(self.merge_heads(weights @ values)), weights


class NeighborModel(nn.Module):
    def __init__(self, vocab_size, length, d_model, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position = nn.Parameter(torch.randn(length, d_model) * 0.1)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, vocab_size))

    def forward(self, tokens):
        x = self.embed(tokens) + self.position
        attended, weights = self.attention(x)
        return self.mlp(x + attended), weights


def make_neighbor_batch(batch, length, generator):
    tokens = torch.randint(0, 10, (batch, length), generator=generator)
    targets = (tokens.roll(1, dims=1) - tokens.roll(-1, dims=1)) % 10
    return tokens, targets


def train_neighbor_model(num_heads, generator, steps=1500, length=10, d_model=32):
    torch.manual_seed(0)
    model = NeighborModel(10, length, d_model, num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        tokens, targets = make_neighbor_batch(128, length, generator)
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tokens, targets = make_neighbor_batch(2000, length, generator)
    with torch.no_grad():
        logits, _ = model(tokens)
    return model, (logits.argmax(dim=-1) == targets).float().mean().item()


def main():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    d_model, num_heads, length = 32, 4, 6
    x = torch.randn(2, length, d_model, generator=generator)

    attention = MultiHeadAttention(d_model, num_heads)
    output, weights = attention(x)
    print(f"d_model={d_model}, heads={num_heads}, d_head={attention.d_head}")
    print(f"Input {tuple(x.shape)} -> per-head Q {tuple(attention.split_heads(x).shape)} -> "
          f"weights {tuple(weights.shape)} -> output {tuple(output.shape)}")
    print(f"Parameters: {sum(p.numel() for p in attention.parameters()):,} "
          f"(4 * d_model^2 + 4 * d_model = {4 * d_model * d_model + 4 * d_model:,}, independent of the number of heads)")

    library = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    with torch.no_grad():
        library.in_proj_weight.copy_(attention.qkv.weight)
        library.in_proj_bias.copy_(attention.qkv.bias)
        library.out_proj.weight.copy_(attention.output.weight)
        library.out_proj.bias.copy_(attention.output.bias)
        library_output, library_weights = library(x, x, x, average_attn_weights=False)
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
        mine_causal, _ = attention(x, causal)
        library_causal, _ = library(x, x, x, attn_mask=~causal)
    print(f"\nYours vs nn.MultiheadAttention: output diff {(output - library_output).abs().max():.1e}, "
          f"weights diff {(weights - library_weights).abs().max():.1e}")
    print(f"With a causal mask: output diff {(mine_causal - library_causal).abs().max():.1e} "
          f"(PyTorch's boolean attn_mask marks positions to BLOCK, yours marks positions to KEEP)")

    with torch.no_grad():
        queries, keys, values = attention.qkv(x).chunk(3, dim=-1)
        separate = []
        for head in range(num_heads):
            columns = slice(head * attention.d_head, (head + 1) * attention.d_head)
            scores = queries[..., columns] @ keys[..., columns].transpose(-2, -1) / math.sqrt(attention.d_head)
            separate.append(torch.softmax(scores, dim=-1) @ values[..., columns])
        loop_output = attention.output(torch.cat(separate, dim=-1))
    print(f"Batched heads vs a Python loop over {num_heads} independent heads: diff {(loop_output - output).abs().max():.1e}")

    print("\nTask: output (x[i-1] - x[i+1]) mod 10 at every position (sequence wraps around).")
    tokens, targets = make_neighbor_batch(1, 10, generator)
    print(f"  tokens  {tokens[0].tolist()}\n  targets {targets[0].tolist()}")
    print(f"  {'heads':>5} {'d_head':>6} {'accuracy':>9} {'seconds':>8}")
    trained = {}
    for heads in [1, 2, 4]:
        start = time.perf_counter()
        model, accuracy = train_neighbor_model(heads, generator)
        trained[heads] = model
        print(f"  {heads:>5} {32 // heads:>6} {accuracy:>9.3f} {time.perf_counter() - start:>8.1f}")

    with torch.no_grad():
        _, single_weights = trained[1](tokens)
        _, weights = trained[4](tokens)
    print("\n  Where the 1-head model looks from position 5 (columns = positions 0..9):")
    print("  head 0: " + " ".join(f"{value:.2f}" for value in single_weights[0, 0, 5].tolist()))
    print("\n  Where each head of the 4-head model looks from position 5:")
    for head in range(4):
        row = weights[0, head, 5]
        print(f"  head {head}: " + " ".join(f"{value:.2f}" for value in row.tolist()) + f"   argmax {row.argmax().item()}")
    offsets = []
    for head in range(4):
        focus = weights[0, head].argmax(dim=-1)
        offset = ((focus - torch.arange(10)) % 10)
        offsets.append(torch.mode(offset).values.item())
    print(f"  Most common offset (key position - query position, mod 10) per head: {offsets}")


if __name__ == "__main__":
    main()
