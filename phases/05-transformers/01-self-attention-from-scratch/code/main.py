import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(queries, keys, values, mask=None):
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ values, weights


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.output = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x, mask=None):
        attended, weights = scaled_dot_product_attention(self.query(x), self.key(x), self.value(x), mask)
        return self.output(attended), weights


def causal_mask(length):
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


class CountingModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, use_attention):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attention = SelfAttention(d_model, d_model) if use_attention else None
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, num_classes))

    def forward(self, tokens):
        x = self.embed(tokens)
        if self.attention is not None:
            attended, _ = self.attention(x)
            x = x + attended
        return self.mlp(x)


def make_counting_batch(batch, length, vocab_size, generator):
    tokens = torch.randint(0, vocab_size, (batch, length), generator=generator)
    counts = (tokens[:, :, None] == tokens[:, None, :]).sum(dim=-1)
    return tokens, counts


def train_counting(use_attention, generator, steps=1500, vocab_size=6, length=12):
    torch.manual_seed(0)
    model = CountingModel(vocab_size, 32, length + 1, use_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        tokens, counts = make_counting_batch(128, length, vocab_size, generator)
        loss = F.cross_entropy(model(tokens).reshape(-1, length + 1), counts.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tokens, counts = make_counting_batch(2000, length, vocab_size, generator)
    with torch.no_grad():
        accuracy = (model(tokens).argmax(dim=-1) == counts).float().mean().item()
    return model, accuracy


def main():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)

    x = torch.randn(2, 5, 16, generator=generator)
    layer = SelfAttention(16, 8)
    output, weights = layer(x)
    print(f"Input {tuple(x.shape)} -> Q, K, V {tuple(layer.query(x).shape)} -> weights {tuple(weights.shape)} "
          f"-> output {tuple(output.shape)}")
    print(f"Each row of the weight matrix sums to 1: {torch.allclose(weights.sum(dim=-1), torch.ones(2, 5))}")

    queries, keys, values = layer.query(x), layer.key(x), layer.value(x)
    mine, _ = scaled_dot_product_attention(queries, keys, values)
    library = F.scaled_dot_product_attention(queries, keys, values)
    mine_causal, causal_weights = scaled_dot_product_attention(queries, keys, values, causal_mask(5))
    library_causal = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
    print(f"\nYours vs F.scaled_dot_product_attention: {(mine - library).abs().max():.1e}, "
          f"causal: {(mine_causal - library_causal).abs().max():.1e}")
    print("Causal attention weights for one sequence (row t can only see positions <= t):")
    for row in causal_weights[0].tolist():
        print("  " + " ".join(f"{value:.2f}" for value in row))

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted_output, _ = layer(x[:, permutation])
    print(f"\nPermutation equivariance: attention(permuted x) == permuted attention(x): "
          f"{torch.allclose(permuted_output, output[:, permutation], atol=1e-6)}")

    print("\nWhy divide by sqrt(d)? Random q, k with unit-variance entries, 16 keys:")
    print(f"  {'d':>5} {'std(q.k)':>10} {'max weight unscaled':>20} {'max weight scaled':>18}")
    for d in [4, 16, 64, 256, 1024]:
        q = torch.randn(1000, 1, d, generator=generator)
        k = torch.randn(1000, 16, d, generator=generator)
        raw = (q @ k.transpose(1, 2)).squeeze(1)
        unscaled = torch.softmax(raw, dim=-1).max(dim=-1).values.mean()
        scaled = torch.softmax(raw / math.sqrt(d), dim=-1).max(dim=-1).values.mean()
        print(f"  {d:>5} {raw.std():>10.2f} {unscaled:>20.3f} {scaled:>18.3f}")

    print("\nTask: for every token, output how many times it appears in the sequence (length 12, 6 symbols).")
    tokens, counts = make_counting_batch(1, 12, 6, generator)
    print(f"  tokens {tokens[0].tolist()}\n  counts {counts[0].tolist()}")
    _, baseline_accuracy = train_counting(False, generator)
    model, attention_accuracy = train_counting(True, generator)
    print(f"  per-token MLP, no attention: accuracy {baseline_accuracy:.3f}")
    print(f"  one self-attention layer:    accuracy {attention_accuracy:.3f}")

    with torch.no_grad():
        example = torch.tensor([[0, 1, 0, 2, 2, 2, 3, 0, 4, 2, 5, 2]])
        _, example_weights = model.attention(model.embed(example))
        same = example[0][:, None] == example[0][None, :]
        on_same = (example_weights[0] * same).sum(dim=-1)
    print(f"\n  Example {example[0].tolist()}")
    print("  Attention mass each position puts on positions holding the same token:")
    print("  " + " ".join(f"{value:.2f}" for value in on_same.tolist()))
    print(f"  Predicted counts {model(example).argmax(dim=-1)[0].tolist()}")


if __name__ == "__main__":
    main()
