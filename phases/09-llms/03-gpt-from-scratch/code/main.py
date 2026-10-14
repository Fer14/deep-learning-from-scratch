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


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_id[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        batch, length, embed_dim = x.shape
        q, k, v = self.qkv(x).split(embed_dim, dim=2)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(batch, length, embed_dim)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.out = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        return self.dropout(self.out(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature, generator):
        for _ in range(max_new_tokens):
            context = ids[:, -self.block_size :]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1, generator=generator)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def get_batch(data, block_size, batch_size, device, generator):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, device, generator, batches=10):
    model.eval()
    losses = [model(*get_batch(data, block_size, 32, device, generator))[1].item() for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CharTokenizer(CORPUS)
    data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
    cut = int(0.9 * len(data))
    train_data, val_data = data[:cut], data[cut:]
    block_size = 64

    model = GPT(len(tokenizer.chars), block_size, embed_dim=64, num_heads=4, num_layers=2, dropout=0.2).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | vocab {len(tokenizer.chars)} | parameters {num_params:,}")

    generator = torch.Generator().manual_seed(0)
    x, y = get_batch(train_data, block_size, 4, device, generator)
    print(f"step 1: one batch x {tuple(x.shape)}, targets are x shifted by one: {torch.equal(x[0, 1:], y[0, :-1])}")
    _, loss = model(x, y)
    print(f"step 2: initial loss {loss.item():.3f} (uniform guess = ln {len(tokenizer.chars)} = {math.log(len(tokenizer.chars)):.3f})")

    print("step 3: training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    steps = 600
    best_val, best_state = float("inf"), None
    start = time.time()
    for step in range(steps + 1):
        xb, yb = get_batch(train_data, block_size, 32, device, generator)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            train_loss = estimate_loss(model, train_data, block_size, device, generator)
            val_loss = estimate_loss(model, val_data, block_size, device, generator)
            if val_loss < best_val:
                best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  step {step:4d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} | {time.time() - start:5.1f}s")
    model.load_state_dict(best_state)
    model.eval()
    print(f"  restored the checkpoint with the best val loss {best_val:.3f}")

    print("step 4: generate")
    prompt = torch.tensor([tokenizer.encode("the keeper ")], device=device)
    sample_generator = torch.Generator(device=device).manual_seed(1)
    out = model.generate(prompt, 200, temperature=0.8, generator=sample_generator)
    print("  " + tokenizer.decode(out[0].tolist()).replace("\n", "\n  "))

    print("\nuse it: compare the hand-written attention with PyTorch kernels")
    attention = model.blocks[0].attn
    attention.eval()
    x = torch.randn(2, 16, 64, device=device)
    with torch.no_grad():
        ours = attention(x)
        q, k, v = attention.qkv(x).split(64, dim=2)
        q, k, v = attention.split_heads(q), attention.split_heads(k), attention.split_heads(v)
        fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        fused = attention.proj(fused.transpose(1, 2).reshape(2, 16, 64))
        mha = nn.MultiheadAttention(64, 4, batch_first=True).to(device)
        mha.in_proj_weight.copy_(attention.qkv.weight)
        mha.in_proj_bias.copy_(attention.qkv.bias)
        mha.out_proj.weight.copy_(attention.proj.weight)
        mha.out_proj.bias.copy_(attention.proj.bias)
        causal_mask = torch.triu(torch.ones(16, 16, dtype=torch.bool, device=device), diagonal=1)
        library, _ = mha(x, x, x, attn_mask=causal_mask)
    print(f"  max |ours - scaled_dot_product_attention| = {(ours - fused).abs().max().item():.2e}")
    print(f"  max |ours - nn.MultiheadAttention|        = {(ours - library).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
