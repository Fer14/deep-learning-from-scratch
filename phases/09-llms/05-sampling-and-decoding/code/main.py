import math

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
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, block_size) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, ids):
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.ln_final(self.blocks(x)))

    @torch.no_grad()
    def next_logits(self, ids):
        return self(ids[:, -self.block_size :])[:, -1, :]


def train_model(data, vocab_size, device, steps=500, block_size=64):
    torch.manual_seed(0)
    model = GPT(vocab_size, block_size, 64, 4, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        starts = torch.randint(len(data) - block_size - 1, (32,), generator=generator)
        x = torch.stack([data[s : s + block_size] for s in starts]).to(device)
        y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, loss.item()


def apply_temperature(logits, temperature):
    return logits / temperature


def top_k_filter(logits, k):
    kth_value = torch.sort(logits, descending=True).values[..., k - 1 : k]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def top_p_filter(logits, p):
    sorted_logits, sorted_index = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative - sorted_probs > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_index, sorted_logits)


def sample_from_probs(probs, generator):
    cumulative = probs.cumsum(dim=-1)
    u = torch.rand(probs.shape[0], 1, generator=generator, device=probs.device)
    return torch.searchsorted(cumulative, u * cumulative[:, -1:]).clamp(max=probs.shape[-1] - 1)


def generate(model, prompt_ids, length, strategy, generator):
    ids = prompt_ids
    for _ in range(length):
        logits = model.next_logits(ids)
        if strategy["name"] == "greedy":
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = apply_temperature(logits, strategy.get("temperature", 1.0))
            if "top_k" in strategy:
                logits = top_k_filter(logits, strategy["top_k"])
            if "top_p" in strategy:
                logits = top_p_filter(logits, strategy["top_p"])
            next_id = sample_from_probs(F.softmax(logits, dim=-1), generator)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def beam_search(model, prompt_ids, length, beam_width):
    beams = [(0.0, prompt_ids)]
    for _ in range(length):
        candidates = []
        for score, ids in beams:
            log_probs = F.log_softmax(model.next_logits(ids), dim=-1)[0]
            top_log_probs, top_ids = log_probs.topk(beam_width)
            for log_prob, token in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
                candidates.append((score + log_prob, new_ids))
        beams = sorted(candidates, key=lambda c: c[0], reverse=True)[:beam_width]
    return beams[0][1]


@torch.no_grad()
def mean_log_prob(model, ids, prompt_length):
    total, count = 0.0, 0
    for t in range(prompt_length, ids.shape[1]):
        log_probs = F.log_softmax(model.next_logits(ids[:, :t]), dim=-1)
        total += log_probs[0, ids[0, t]].item()
        count += 1
    return total / count


def distinct_ngrams(text, n):
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def known_word_fraction(text, known_words):
    words = [w.strip(".,:'\"") for w in text.split()]
    words = [w for w in words if w]
    return sum(w in known_words for w in words) / max(1, len(words))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chars = sorted(set(CORPUS))
    char_to_id = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([char_to_id[c] for c in CORPUS], dtype=torch.long)
    known_words = {w.strip(".,:'\"") for w in CORPUS.split()}

    model, final_loss = train_model(data, len(chars), device)
    print(f"device {device} | trained a tiny GPT, final train loss {final_loss:.3f}")

    print("\nstep 1: what the filters keep from one next-token distribution")
    prompt = "the keeper "
    prompt_ids = torch.tensor([[char_to_id[c] for c in prompt]], device=device)
    logits = model.next_logits(prompt_ids)
    for label, filtered in [
        ("raw", logits),
        ("temperature 0.5", apply_temperature(logits, 0.5)),
        ("temperature 1.5", apply_temperature(logits, 1.5)),
        ("top-k 3", top_k_filter(logits, 3)),
        ("top-p 0.9", top_p_filter(logits, 0.9)),
    ]:
        probs = F.softmax(filtered, dim=-1)[0]
        top = probs.topk(4)
        shown = "  ".join(f"{chars[i]!r}:{p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  {label:16s} nonzero {int((probs > 0).sum()):2d}/{len(chars)} | {shown}")

    print("\nstep 2: decode 4 continuations of 100 characters with each strategy")
    strategies = [
        {"name": "greedy"},
        {"name": "temperature 0.5", "temperature": 0.5},
        {"name": "temperature 1.0", "temperature": 1.0},
        {"name": "temperature 1.5", "temperature": 1.5},
        {"name": "top-k 5", "top_k": 5},
        {"name": "top-p 0.9", "top_p": 0.9},
        {"name": "top-p 0.9 t=0.8", "top_p": 0.9, "temperature": 0.8},
    ]
    generator = torch.Generator(device=device).manual_seed(0)
    batch_prompt = prompt_ids.repeat(4, 1)
    print(f"  {'strategy':18s} {'logp/char':>9s} {'distinct-4':>10s} {'real words':>10s} {'unique':>6s}")
    examples = {}
    for strategy in strategies:
        out = generate(model, batch_prompt, 100, strategy, generator)
        texts = ["".join(chars[i] for i in row.tolist()) for row in out]
        log_prob = sum(mean_log_prob(model, out[i : i + 1], len(prompt)) for i in range(4)) / 4
        diversity = sum(distinct_ngrams(t[len(prompt) :], 4) for t in texts) / 4
        realness = sum(known_word_fraction(t, known_words) for t in texts) / 4
        print(f"  {strategy['name']:18s} {log_prob:9.3f} {diversity:10.3f} {realness:10.3f} {len(set(texts)):6d}")
        examples[strategy["name"]] = texts[0]

    beam = beam_search(model, prompt_ids, 60, beam_width=5)
    beam_text = "".join(chars[i] for i in beam[0].tolist())
    print(f"  {'beam search (5)':18s} {mean_log_prob(model, beam, len(prompt)):9.3f} {distinct_ngrams(beam_text[len(prompt):], 4):10.3f} {known_word_fraction(beam_text, known_words):10.3f} {1:6d}")
    examples["beam search (5)"] = beam_text

    print("\nstep 3: samples")
    for name in ["greedy", "beam search (5)", "temperature 0.5", "top-p 0.9 t=0.8", "temperature 1.5"]:
        print(f"  {name:16s} {examples[name][:90]!r}")

    print("\nuse it: torch.multinomial samples from the same distribution as the hand sampler")
    probs = F.softmax(logits, dim=-1).repeat(20000, 1)
    ours = sample_from_probs(probs, torch.Generator(device=device).manual_seed(1)).flatten()
    library = torch.multinomial(probs, 1, generator=torch.Generator(device=device).manual_seed(1)).flatten()
    ours_freq = torch.bincount(ours, minlength=len(chars)).float() / len(ours)
    library_freq = torch.bincount(library, minlength=len(chars)).float() / len(library)
    print(f"  max |frequency difference| over 20000 draws: {(ours_freq - library_freq).abs().max().item():.4f}")
    print(f"  max |frequency - true probability|:          {(library_freq - probs[0]).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
