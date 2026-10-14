import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = """the old harbor town woke up before the sun. the baker lit the oven and the smell of warm bread drifted down to the water. the fishermen pulled their boats into the cold sea and the gulls followed them, crying for scraps.
on the hill above the town stood a lighthouse. the keeper was a quiet woman who wrote everything in a small blue notebook. she wrote the wind, the waves, the color of the sky and the names of the boats that came home.
every morning the keeper climbed the stairs of the lighthouse. there were one hundred and twelve stairs, and she counted them every time. at the top she cleaned the great lamp and looked out at the sea.
the baker had a son who wanted to be a sailor. he watched the boats leave the harbor and he dreamed of islands far away. his father wanted him to bake bread, but the boy wanted to read the stars.
one evening a storm came from the north. the wind pushed the waves over the wall of the harbor and the rain fell like stones. the keeper lit the great lamp and the light turned slowly over the dark water.
a small boat was still out at sea. the fishermen on the dock could not see it, but the keeper saw a tiny light between the waves. she wrote in her notebook: a boat in the storm, north of the rocks, one light.
the baker's son ran up the hill to the lighthouse. he climbed the one hundred and twelve stairs and stood next to the keeper. together they watched the tiny light move closer to the rocks.
the keeper turned the great lamp toward the boat. the light showed the safe path between the rocks, and the boat followed it slowly into the harbor. the fishermen cheered and the gulls flew in circles above the dock.
the next morning the sea was calm and the sky was clear. the baker made a large loaf of bread for the keeper and his son carried it up the hill. the keeper smiled and wrote in her notebook: calm sea, clear sky, warm bread.
after that day the boy came to the lighthouse every evening. the keeper taught him the names of the stars and the shapes of the clouds. she taught him how to read the wind and how to clean the great lamp.
the boy still baked bread in the morning, but in the evening he climbed the stairs and watched the sea. he wrote his own notes in a small green notebook: the wind from the west, the waves are low, three boats came home.
years later the keeper grew old and her hands began to shake. the boy, now a young man, became the new keeper of the lighthouse. he kept the blue notebook next to his green one and read it on quiet nights.
the town changed slowly. new houses were built on the hill and new boats came to the harbor. but the lighthouse stayed the same, and every night the light turned over the dark water.
the new keeper had a daughter who loved numbers. she counted the stairs, the boats, the gulls and the waves. she wrote the numbers in a small red notebook and drew lines between them.
one winter the daughter noticed that the storms came more often when the wind turned from the north. she showed her father the lines in her notebook, and he showed her the old blue notebook. the same pattern was there, written many years before.
the keeper and his daughter wrote a letter to the fishermen. when the wind turns north, they wrote, stay in the harbor and wait for the light. the fishermen read the letter and pinned it on the wall of the bakery.
the bakery still smells of warm bread in the morning. the gulls still cry above the dock. and on the hill the lighthouse still stands, with three small notebooks on the table at the top of the stairs.
"""


class CachedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, use_sdpa=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sdpa = use_sdpa
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x, cache=None):
        batch, length, embed_dim = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        past_length = 0
        if cache is not None:
            past_k, past_v = cache
            past_length = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)
        total_length = k.shape[2]
        if self.use_sdpa:
            mask = self.mask[past_length:total_length, :total_length]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~self.mask[past_length:total_length, :total_length], float("-inf"))
            out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(batch, length, embed_dim)
        return self.proj(out), new_cache


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CachedSelfAttention(embed_dim, num_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, x, cache=None):
        attended, new_cache = self.attn(self.ln1(x), cache)
        x = x + attended
        return x + self.mlp(self.ln2(x)), new_cache


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, block_size) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, ids, caches=None):
        past_length = 0 if caches is None else caches[0][0].shape[2]
        positions = torch.arange(past_length, past_length + ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if caches is None else caches[i])
            new_caches.append(cache)
        return self.head(self.ln_final(x)), new_caches


@torch.no_grad()
def generate_naive(model, prompt_ids, max_new_tokens):
    ids = prompt_ids
    all_logits = []
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        all_logits.append(logits[:, -1])
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return ids, torch.stack(all_logits, dim=1)


@torch.no_grad()
def generate_cached(model, prompt_ids, max_new_tokens):
    logits, caches = model(prompt_ids)
    ids = prompt_ids
    all_logits = []
    for step in range(max_new_tokens):
        all_logits.append(logits[:, -1])
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if step < max_new_tokens - 1:
            logits, caches = model(next_id, caches)
    return ids, torch.stack(all_logits, dim=1), caches


def train(model, data, device, steps, block_size):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        starts = torch.randint(len(data) - block_size - 1, (16,), generator=generator)
        x = torch.stack([data[s : s + block_size] for s in starts]).to(device)
        y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts]).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return loss.item()


def timed(function, device, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = function()
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best, result


def cache_bytes(num_layers, num_kv_heads, head_dim, length, bytes_per_value):
    return 2 * num_layers * num_kv_heads * head_dim * length * bytes_per_value


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chars = sorted(set(CORPUS))
    char_to_id = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([char_to_id[c] for c in CORPUS], dtype=torch.long)

    print(f"device {device}")
    print("step 1: train a small GPT with a 128-character context")
    model = GPT(len(chars), 128, 64, 4, 2).to(device)
    print(f"  final train loss {train(model, data, device, steps=300, block_size=128):.3f}")

    print("\nstep 2: generate 110 characters with and without the cache")
    prompt = torch.tensor([[char_to_id[c] for c in "the keeper"]], device=device)
    naive_ids, naive_logits = generate_naive(model, prompt, 110)
    cached_ids, cached_logits, caches = generate_cached(model, prompt, 110)
    print(f"  identical tokens: {torch.equal(naive_ids, cached_ids)} | max |logit difference| {(naive_logits - cached_logits).abs().max().item():.2e}")
    print(f"  text: {''.join(chars[i] for i in cached_ids[0].tolist())!r}")
    k, v = caches[0]
    print(f"  cache per layer: keys {tuple(k.shape)} + values {tuple(v.shape)} = (batch, heads, positions, head_dim)")

    print("\nstep 3: speed on a wider random model (batch of 8, context 512, 300 new tokens)")
    torch.manual_seed(1)
    big = GPT(64, 512, 128, 4, 4).to(device).eval()
    long_prompt = torch.randint(0, 64, (8, 16), device=device)
    naive_time, (naive_out, _) = timed(lambda: generate_naive(big, long_prompt, 300), device, repeats=1)
    cached_time, (cached_out, _, _) = timed(lambda: generate_cached(big, long_prompt, 300), device, repeats=1)
    print(f"  without cache {naive_time:6.2f}s ({8 * 300 / naive_time:7.1f} tokens/s)")
    print(f"  with cache    {cached_time:6.2f}s ({8 * 300 / cached_time:7.1f} tokens/s) | speedup {naive_time / cached_time:.1f}x | same tokens {torch.equal(naive_out, cached_out)}")
    prompt_length, new_tokens = 16, 300
    naive_positions = sum(prompt_length + t for t in range(new_tokens))
    cached_positions = prompt_length + new_tokens - 1
    print(f"  positions pushed through the MLPs per sequence: naive {naive_positions:,} vs cached {cached_positions:,}")

    print("\nstep 4: how big does the cache get for a real model?")
    gigabyte = 1024**3
    mha = cache_bytes(num_layers=32, num_kv_heads=32, head_dim=128, length=4096, bytes_per_value=2)
    gqa = cache_bytes(num_layers=32, num_kv_heads=8, head_dim=128, length=4096, bytes_per_value=2)
    print(f"  7B-class model, 32 layers, 32 heads x 128 dims, 4096 tokens, fp16: {mha / gigabyte:.2f} GB per sequence")
    print(f"  same model with grouped-query attention (8 KV heads):             {gqa / gigabyte:.2f} GB per sequence")
    print(f"  a batch of 32 such sequences with full multi-head attention:      {32 * mha / gigabyte:.0f} GB")

    print("\nuse it: scaled_dot_product_attention inside the cached step")
    for block in model.blocks:
        block.attn.use_sdpa = True
    sdpa_ids, sdpa_logits, _ = generate_cached(model, prompt, 110)
    print(f"  identical tokens: {torch.equal(sdpa_ids, cached_ids)} | max |logit difference| {(sdpa_logits - cached_logits).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
