import math
import time

import torch
import torch.nn.functional as F


def naive_attention(q, k, v, causal=False):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        length = q.shape[-2]
        mask = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def tiled_attention(q, k, v, block_size):
    scale = 1.0 / math.sqrt(q.shape[-1])
    output = torch.zeros_like(q)
    for q_start in range(0, q.shape[-2], block_size):
        q_block = q[..., q_start:q_start + block_size, :] * scale
        running_max = torch.full(q_block.shape[:-1], float("-inf"), device=q.device)
        running_sum = torch.zeros(q_block.shape[:-1], device=q.device)
        accumulator = torch.zeros_like(q_block)
        for k_start in range(0, k.shape[-2], block_size):
            k_block = k[..., k_start:k_start + block_size, :]
            v_block = v[..., k_start:k_start + block_size, :]
            scores = q_block @ k_block.transpose(-2, -1)
            new_max = torch.maximum(running_max, scores.amax(dim=-1))
            correction = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max[..., None])
            running_sum = running_sum * correction + probabilities.sum(dim=-1)
            accumulator = accumulator * correction[..., None] + probabilities @ v_block
            running_max = new_max
        output[..., q_start:q_start + block_size, :] = accumulator / running_sum[..., None]
    return output


def sliding_window_mask(length, window, device):
    positions = torch.arange(length, device=device)
    distance = positions[:, None] - positions[None, :]
    return (distance >= 0) & (distance < window)


def feature_map(x):
    return F.elu(x) + 1


def linear_attention(q, k, v):
    q, k = feature_map(q), feature_map(k)
    key_value = k.transpose(-2, -1) @ v
    normalizer = q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)
    return (q @ key_value) / normalizer


def linear_attention_quadratic_form(q, k, v):
    q, k = feature_map(q), feature_map(k)
    weights = q @ k.transpose(-2, -1)
    return (weights / weights.sum(dim=-1, keepdim=True)) @ v


def causal_linear_attention(q, k, v):
    q, k = feature_map(q), feature_map(k)
    key_value = torch.cumsum(k[..., :, :, None] * v[..., :, None, :], dim=-3)
    numerator = (q[..., :, :, None] * key_value).sum(dim=-2)
    normalizer = (q * torch.cumsum(k, dim=-2)).sum(dim=-1, keepdim=True)
    return numerator / normalizer


def benchmark(function, repeats):
    function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def peak_memory_mb(function, device):
    if device != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    function()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / 2 ** 20


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nMemory for ONE attention score matrix (float32, batch 1, 1 head):")
    for length in [1_024, 8_192, 32_768, 131_072]:
        print(f"  n = {length:>7,}: n^2 = {length ** 2:>17,} scores = {length ** 2 * 4 / 2 ** 30:>8.2f} GiB")

    q, k, v = (torch.randn(2, 4, 256, 32, device=device) for _ in range(3))
    reference = naive_attention(q, k, v)
    for block_size in [16, 64, 100]:
        print(f"Tiled attention with online softmax, block {block_size:>3}: max diff from naive "
              f"{(tiled_attention(q, k, v, block_size) - reference).abs().max():.1e}")

    window = 32
    mask = sliding_window_mask(256, window, device)
    windowed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    print(f"\nSliding window ({window}): each query sees {mask.sum(dim=-1).max().item()} keys instead of up to 256 "
          f"({mask.float().mean():.1%} of the full matrix)")
    print(f"  output differs from causal full attention by {(windowed - naive_attention(q, k, v, causal=True)).abs().mean():.3f} "
          f"on average (it is an approximation)")

    print(f"\nLinear attention, softmax(q.k) replaced by phi(q).phi(k) with phi = elu + 1:")
    print(f"  (phi(Q) phi(K)^T) V == phi(Q) (phi(K)^T V): max diff "
          f"{(linear_attention(q, k, v) - linear_attention_quadratic_form(q, k, v)).abs().max():.1e}")
    print(f"  differs from softmax attention by {(linear_attention(q, k, v) - reference).abs().mean():.3f} on average "
          f"(a different attention function, not an approximation of the same numbers)")
    causal = causal_linear_attention(q, k, v)
    causal_check = torch.stack([linear_attention(q[..., :t + 1, :], k[..., :t + 1, :], v[..., :t + 1, :])[..., -1, :]
                                for t in [0, 50, 255]], dim=-2)
    print(f"  causal version with running sums matches recomputing each prefix: "
          f"{(causal[..., [0, 50, 255], :] - causal_check).abs().max():.1e}")

    print(f"\nTime per call (ms) and peak extra memory (MB), batch 1, 8 heads, d_head 64:")
    print(f"  {'n':>6} {'naive ms':>9} {'naive MB':>9} {'tiled ms':>9} {'sdpa ms':>8} {'sdpa MB':>8} {'linear ms':>10} {'linear MB':>10}")
    lengths = [512, 1024, 2048, 4096] if device == "cuda" else [256, 512, 1024]
    for length in lengths:
        q, k, v = (torch.randn(1, 8, length, 64, device=device) for _ in range(3))
        repeats = 5 if device == "cuda" else 3
        row = [
            benchmark(lambda: naive_attention(q, k, v), repeats), peak_memory_mb(lambda: naive_attention(q, k, v), device),
            benchmark(lambda: tiled_attention(q, k, v, 256), 1),
            benchmark(lambda: F.scaled_dot_product_attention(q, k, v), repeats),
            peak_memory_mb(lambda: F.scaled_dot_product_attention(q, k, v), device),
            benchmark(lambda: linear_attention(q, k, v), repeats), peak_memory_mb(lambda: linear_attention(q, k, v), device),
        ]
        memory = [f"{value:.1f}" if device == "cuda" else "-" for value in (row[1], row[4], row[6])]
        print(f"  {length:>6} {row[0]:>9.2f} {memory[0]:>9} {row[2]:>9.2f} {row[3]:>8.2f} {memory[1]:>8} {row[5]:>10.2f} {memory[2]:>10}")

    num_heads, num_kv_heads, d_head, length = 8, 2, 64, 1024
    print(f"\nKV memory per token per layer: multi-head {2 * num_heads * d_head} values, "
          f"grouped-query with {num_kv_heads} KV heads {2 * num_kv_heads * d_head}, multi-query {2 * d_head}")
    q = torch.randn(1, num_heads, length, d_head, device=device)
    k, v = (torch.randn(1, num_kv_heads, length, d_head, device=device) for _ in range(2))
    expanded = naive_attention(q, k.repeat_interleave(num_heads // num_kv_heads, dim=1),
                               v.repeat_interleave(num_heads // num_kv_heads, dim=1))
    library = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    print(f"Grouped-query attention: repeated KV heads vs F.scaled_dot_product_attention(enable_gqa=True): "
          f"{(expanded - library).abs().max():.1e}")


if __name__ == "__main__":
    main()
