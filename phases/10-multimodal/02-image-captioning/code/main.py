import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = ["circle", "square", "triangle"]
COLOURS = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.3, 0.95), "yellow": (0.95, 0.85, 0.1)}
SIZES = ["small", "big"]
SIDES = ["left", "right"]
HELD_OUT = {("green", "triangle"), ("blue", "square")}
IMAGE_SIZE = 16


def render(shape, colour, size, side, rng):
    radius = rng.uniform(2.0, 2.8) if size == "small" else rng.uniform(4.0, 4.8)
    cx = rng.uniform(3.5, 5.0) if side == "left" else rng.uniform(11.0, 12.5)
    cy = rng.uniform(5.0, 11.0)
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE] + 0.5
    if shape == "circle":
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    elif shape == "square":
        mask = (np.abs(x - cx) <= radius * 0.85) & (np.abs(y - cy) <= radius * 0.85)
    else:
        top = cy - radius
        mask = (y >= top) & (y <= cy + radius * 0.8) & (np.abs(x - cx) <= (y - top) * 0.6)
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0.05) + rng.normal(0, 0.03, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image[mask] = np.array(COLOURS[colour]) + rng.normal(0, 0.05, 3)
    return np.clip(image, 0, 1).transpose(2, 0, 1)


def make_dataset(num_images, rng, allow):
    images, captions = [], []
    while len(images) < num_images:
        shape, colour = rng.choice(SHAPES), rng.choice(list(COLOURS))
        size, side = rng.choice(SIZES), rng.choice(SIDES)
        if not allow((colour, shape)):
            continue
        images.append(render(shape, colour, size, side, rng))
        captions.append(f"a {size} {colour} {shape} on the {side}")
    return torch.tensor(np.array(images), dtype=torch.float32), captions


class Vocabulary:
    def __init__(self):
        words = sorted({"a", "on", "the"} | set(SHAPES) | set(COLOURS) | set(SIZES) | set(SIDES))
        self.words = ["<pad>", "<bos>", "<eos>"] + words
        self.index = {w: i for i, w in enumerate(self.words)}

    def encode(self, captions):
        rows = [[1] + [self.index[w] for w in c.split()] + [2] for c in captions]
        return torch.tensor(rows)

    def decode(self, ids):
        words = []
        for i in ids:
            if i == 2:
                break
            if i > 2:
                words.append(self.words[i])
        return " ".join(words)


class ConvEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
        )
        self.position = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)

    def forward(self, images):
        features = self.net(images)
        return features.flatten(2).transpose(1, 2) + self.position


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.last_weights = None

    def forward(self, x, context, causal):
        batch, length, embed_dim = x.shape
        q = self.q(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).view(batch, context.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.ones(length, context.shape[1], dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        return self.proj((weights @ v).transpose(1, 2).reshape(batch, length, embed_dim))


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = Attention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = Attention(embed_dim, num_heads)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, x, image_tokens):
        h = self.ln1(x)
        x = x + self.self_attn(h, h, causal=True)
        x = x + self.cross_attn(self.ln2(x), image_tokens, causal=False)
        return x + self.mlp(self.ln3(x))


class Captioner(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_length=10):
        super().__init__()
        self.encoder = ConvEncoder(embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def decode_step(self, ids, image_tokens):
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x, image_tokens)
        return self.head(self.ln_final(x))

    def forward(self, images, ids):
        return self.decode_step(ids, self.encoder(images))


class LibraryCaptioner(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_length=10):
        super().__init__()
        self.encoder = ConvEncoder(embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        layer = nn.TransformerDecoderLayer(embed_dim, num_heads, 4 * embed_dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(embed_dim))
        self.head = nn.Linear(embed_dim, vocab_size)

    def decode_step(self, ids, image_tokens):
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        mask = nn.Transformer.generate_square_subsequent_mask(ids.shape[1], device=ids.device)
        return self.head(self.decoder(x, image_tokens, tgt_mask=mask, tgt_is_causal=True))

    def forward(self, images, ids):
        return self.decode_step(ids, self.encoder(images))


def train(model, images, tokens, steps, device, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(steps + 1):
        batch = torch.randint(len(images), (64,))
        x, ids = images[batch].to(device), tokens[batch].to(device)
        logits = model(x, ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"  step {step:4d} | loss {loss.item():.3f}")
    model.eval()
    return model


@torch.no_grad()
def greedy_caption(model, images, max_length=9):
    image_tokens = model.encoder(images)
    ids = torch.ones(len(images), 1, dtype=torch.long, device=images.device)
    for _ in range(max_length):
        next_id = model.decode_step(ids, image_tokens)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[:, 1:]


def score(predicted, reference):
    exact = np.mean([p == r for p, r in zip(predicted, reference)])
    slots = {"size": 1, "colour": 2, "shape": 3, "side": 6}
    slot_accuracy = {}
    for name, position in slots.items():
        hits = [len(p.split()) > position and p.split()[position] == r.split()[position] for p, r in zip(predicted, reference)]
        slot_accuracy[name] = float(np.mean(hits))
    return exact, slot_accuracy


def evaluate(model, vocab, images, captions, device):
    predicted = [vocab.decode(row.tolist()) for row in greedy_caption(model, images.to(device))]
    return predicted, *score(predicted, captions)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    vocab = Vocabulary()
    train_images, train_captions = make_dataset(5000, rng, allow=lambda combo: combo not in HELD_OUT)
    test_images, test_captions = make_dataset(500, rng, allow=lambda combo: combo not in HELD_OUT)
    novel_images, novel_captions = make_dataset(300, rng, allow=lambda combo: combo in HELD_OUT)
    train_tokens = vocab.encode(train_captions)
    print(f"device {device} | vocab {len(vocab.words)} words | {len(train_captions)} training pairs")
    print(f"example: {train_captions[0]!r} -> {train_tokens[0].tolist()}")
    print(f"held-out colour/shape pairs never seen in training: {sorted(HELD_OUT)}")

    print("\nstep 1: train the hand-written encoder-decoder captioner with teacher forcing")
    model = train(Captioner(len(vocab.words)).to(device), train_images, train_tokens, steps=800, device=device, seed=0)
    print(f"  parameters {sum(p.numel() for p in model.parameters()):,}")

    print("\nstep 2: greedy decoding")
    predicted, exact, slots = evaluate(model, vocab, test_images, test_captions, device)
    for p, r in list(zip(predicted, test_captions))[:4]:
        print(f"  predicted {p!r:36s} reference {r!r}")
    print(f"  test exact match {exact:.3f} | per word {', '.join(f'{k} {v:.3f}' for k, v in slots.items())}")

    print("\nstep 3: compositional generalization to unseen colour/shape pairs")
    predicted, exact, slots = evaluate(model, vocab, novel_images, novel_captions, device)
    for p, r in list(zip(predicted, novel_captions))[:3]:
        print(f"  predicted {p!r:36s} reference {r!r}")
    print(f"  novel exact match {exact:.3f} | per word {', '.join(f'{k} {v:.3f}' for k, v in slots.items())}")

    print("\nstep 4: where does cross-attention look while predicting the colour word?")
    with torch.no_grad():
        batch = test_images[:200].to(device)
        ids = vocab.encode(test_captions[:200]).to(device)
        model(batch, ids[:, :-1])
        is_left = torch.tensor(["left" in c for c in test_captions[:200]], device=device)
        for layer_index, block in enumerate(model.blocks):
            weights = block.cross_attn.last_weights.mean(1)
            grid = weights.view(weights.shape[0], weights.shape[1], 4, 4)
            left_mass = grid[..., :2].sum((-2, -1))
            for position, word in [(2, "colour")]:
                print(f"  layer {layer_index} predicting the {word} word: attention mass on the left half {left_mass[is_left, position].mean().item():.2f} (shape on the left) vs {left_mass[~is_left, position].mean().item():.2f} (shape on the right)")

    print("\nuse it: the same model with nn.TransformerDecoder")
    library = train(LibraryCaptioner(len(vocab.words)).to(device), train_images, train_tokens, steps=800, device=device, seed=0)
    _, exact, _ = evaluate(library, vocab, test_images, test_captions, device)
    _, novel_exact, _ = evaluate(library, vocab, novel_images, novel_captions, device)
    print(f"  test exact match {exact:.3f} | novel pairs exact match {novel_exact:.3f}")


if __name__ == "__main__":
    main()
