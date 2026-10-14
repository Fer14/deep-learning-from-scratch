import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = ["circle", "square", "triangle"]
COLOURS = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.3, 0.95), "yellow": (0.95, 0.85, 0.1)}
COLOUR_NAMES = list(COLOURS)
SIDES = ["left", "right"]
IMAGE_SIZE = 16


def draw_shape(image, shape, colour, side, rng):
    radius = rng.uniform(2.3, 3.2)
    cx = rng.uniform(3.5, 4.5) if side == "left" else rng.uniform(11.5, 12.5)
    cy = rng.uniform(4.0, 12.0)
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE] + 0.5
    if shape == "circle":
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    elif shape == "square":
        mask = (np.abs(x - cx) <= radius * 0.85) & (np.abs(y - cy) <= radius * 0.85)
    else:
        top = cy - radius
        mask = (y >= top) & (y <= cy + radius * 0.8) & (np.abs(x - cx) <= (y - top) * 0.6)
    image[mask] = np.array(COLOURS[colour]) + rng.normal(0, 0.05, 3)


def make_scene(rng):
    while True:
        slots = {side: (rng.choice(SHAPES), rng.choice(COLOUR_NAMES)) if rng.random() < 0.7 else None for side in SIDES}
        if any(slots.values()):
            break
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0.05) + rng.normal(0, 0.03, (IMAGE_SIZE, IMAGE_SIZE, 3))
    for side, content in slots.items():
        if content:
            draw_shape(image, content[0], content[1], side, rng)
    return np.clip(image, 0, 1).transpose(2, 0, 1), slots


def ask(slots, rng):
    kind = rng.choice(["count", "colour", "shape", "exists"])
    if kind == "count":
        return kind, "how many shapes are there ?", ["zero", "one", "two"][sum(v is not None for v in slots.values())]
    if kind == "colour":
        side = rng.choice(SIDES)
        return kind, f"what colour is the {side} shape ?", slots[side][1] if slots[side] else "none"
    if kind == "shape":
        side = rng.choice(SIDES)
        return kind, f"what shape is on the {side} ?", slots[side][0] if slots[side] else "none"
    colour = rng.choice(COLOUR_NAMES)
    present = any(v is not None and v[1] == colour for v in slots.values())
    return kind, f"is there a {colour} shape ?", "yes" if present else "no"


def slot_labels(slots):
    labels = []
    for side in SIDES:
        content = slots[side]
        labels.append(SHAPES.index(content[0]) + 1 if content else 0)
        labels.append(COLOUR_NAMES.index(content[1]) + 1 if content else 0)
    return labels


def make_dataset(num_examples, rng):
    images, questions, answers, kinds, labels = [], [], [], [], []
    for _ in range(num_examples):
        image, slots = make_scene(rng)
        kind, question, answer = ask(slots, rng)
        images.append(image)
        questions.append(question)
        answers.append(answer)
        kinds.append(kind)
        labels.append(slot_labels(slots))
    return torch.tensor(np.array(images), dtype=torch.float32), questions, answers, kinds, torch.tensor(labels)


class Vocabulary:
    def __init__(self, sentences):
        words = sorted({w for s in sentences for w in s.split()})
        self.words = ["<pad>", "<bos>", "<sep>", "<eos>"] + words
        self.index = {w: i for i, w in enumerate(self.words)}

    def build(self, questions, answers, length):
        ids = torch.zeros(len(questions), length, dtype=torch.long)
        answer_position = torch.zeros(len(questions), dtype=torch.long)
        for row, (q, a) in enumerate(zip(questions, answers)):
            tokens = [1] + [self.index[w] for w in q.split()] + [2, self.index[a], 3]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            answer_position[row] = len(tokens) - 2
        return ids, answer_position


class VisionEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, images):
        return self.net(images).flatten(2).transpose(1, 2)


class VisionPretrainer(nn.Module):
    def __init__(self, encoder, width=64):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([nn.Linear(width, n) for n in [4, 5, 4, 5]])

    def forward(self, images):
        grid = self.encoder(images).view(images.shape[0], 4, 4, -1)
        left, right = grid[:, :, :2].mean((1, 2)), grid[:, :, 2:].mean((1, 2))
        return [self.heads[0](left), self.heads[1](left), self.heads[2](right), self.heads[3](right)]


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, use_sdpa=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.use_sdpa = use_sdpa

    def forward(self, x):
        batch, length, embed_dim = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            out = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(batch, length, embed_dim))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyVLM(nn.Module):
    def __init__(self, vision_encoder, vocab_size, vision_width=64, embed_dim=64, num_heads=4, num_layers=2, max_length=32):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = nn.Sequential(nn.Linear(vision_width, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, images, ids, blind=False):
        with torch.no_grad():
            visual = self.vision_encoder(images)
        visual_tokens = self.projector(visual)
        if blind:
            visual_tokens = torch.zeros_like(visual_tokens)
        x = torch.cat([visual_tokens, self.token_embedding(ids)], dim=1)
        x = x + self.position_embedding(torch.arange(x.shape[1], device=x.device))
        logits = self.head(self.ln_final(self.blocks(x)))
        return logits[:, visual_tokens.shape[1] :]


def pretrain_vision(encoder, images, labels, device, steps=400):
    model = VisionPretrainer(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(steps):
        batch = torch.randint(len(images) - 1000, (128,))
        outputs = model(images[batch].to(device))
        targets = labels[batch].to(device)
        loss = sum(F.cross_entropy(o, targets[:, i]) for i, o in enumerate(outputs))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model(images[-1000:].to(device))
        accuracy = [(o.argmax(1).cpu() == labels[-1000:, i]).float().mean().item() for i, o in enumerate(outputs)]
    return dict(zip(["left shape", "left colour", "right shape", "right colour"], accuracy))


def answer_loss(logits, ids, answer_position):
    rows = torch.arange(len(ids), device=ids.device)
    answer_logits = logits[rows, answer_position - 1]
    eos_logits = logits[rows, answer_position]
    return (F.cross_entropy(answer_logits, ids[rows, answer_position]) + F.cross_entropy(eos_logits, ids[rows, answer_position + 1])) / 2


def train_vlm(model, images, ids, answer_position, device, steps, blind=False):
    parameters = [p for name, p in model.named_parameters() if not name.startswith("vision_encoder")]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3)
    for step in range(steps + 1):
        batch = torch.randint(len(images), (64,))
        x, b_ids, b_pos = images[batch].to(device), ids[batch].to(device), answer_position[batch].to(device)
        loss = answer_loss(model(x, b_ids, blind=blind), b_ids, b_pos)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not blind and step % 200 == 0:
            print(f"  step {step:4d} | answer loss {loss.item():.3f}")
    model.eval()
    return model


@torch.no_grad()
def predict(model, images, ids, answer_position, device, blind=False):
    logits = model(images.to(device), ids.to(device), blind=blind)
    rows = torch.arange(len(ids), device=device)
    return logits[rows, answer_position.to(device) - 1].argmax(-1).cpu()


def accuracy_by_kind(predictions, ids, answer_position, kinds):
    correct = predictions == ids[torch.arange(len(ids)), answer_position]
    kinds = np.array(kinds)
    return {k: correct[torch.tensor(kinds == k)].float().mean().item() for k in ["count", "colour", "shape", "exists"]}, correct.float().mean().item()


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_images, train_q, train_a, train_kinds, train_labels = make_dataset(20000, rng)
    test_images, test_q, test_a, test_kinds, _ = make_dataset(1000, rng)
    vocab = Vocabulary(train_q + train_a + test_q + test_a)
    train_ids, train_pos = vocab.build(train_q, train_a, 14)
    test_ids, test_pos = vocab.build(test_q, test_a, 14)
    print(f"device {device} | vocab {len(vocab.words)} | 20000 training (image, question, answer) triples")
    for q, a in list(zip(train_q, train_a))[:4]:
        print(f"  Q: {q:34s} A: {a}")

    print("\nstep 1: pretrain the vision encoder on slot attributes, then freeze it")
    vision_encoder = VisionEncoder().to(device)
    accuracy = pretrain_vision(vision_encoder, train_images, train_labels, device)
    print("  held-out attribute accuracy: " + ", ".join(f"{k} {v:.3f}" for k, v in accuracy.items()))
    for parameter in vision_encoder.parameters():
        parameter.requires_grad_(False)
    vision_encoder.eval()

    print("\nstep 2: train projector + language model on visual question answering")
    torch.manual_seed(1)
    model = train_vlm(TinyVLM(vision_encoder, len(vocab.words)).to(device), train_images, train_ids, train_pos, device, steps=800)
    trainable = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("vision_encoder"))
    print(f"  trainable parameters {trainable:,} (vision encoder frozen: {sum(p.numel() for p in vision_encoder.parameters()):,})")

    print("\nstep 3: evaluate on new scenes")
    predictions = predict(model, test_images, test_ids, test_pos, device)
    per_kind, overall = accuracy_by_kind(predictions, test_ids, test_pos, test_kinds)
    print(f"  overall {overall:.3f} | " + " | ".join(f"{k} {v:.3f}" for k, v in per_kind.items()))
    for i in range(5):
        print(f"  Q: {test_q[i]:34s} predicted {vocab.words[predictions[i]]:8s} truth {test_a[i]}")

    print("\nstep 4: a blind baseline that sees only the question")
    torch.manual_seed(1)
    blind_model = train_vlm(TinyVLM(vision_encoder, len(vocab.words)).to(device), train_images, train_ids, train_pos, device, steps=400, blind=True)
    blind_predictions = predict(blind_model, test_images, test_ids, test_pos, device, blind=True)
    per_kind, overall = accuracy_by_kind(blind_predictions, test_ids, test_pos, test_kinds)
    print(f"  overall {overall:.3f} | " + " | ".join(f"{k} {v:.3f}" for k, v in per_kind.items()))

    print("\nuse it: swap the hand-written attention for scaled_dot_product_attention")
    with torch.no_grad():
        reference = model(test_images[:64].to(device), test_ids[:64].to(device))
        for block in model.blocks:
            block.attn.use_sdpa = True
        fused = model(test_images[:64].to(device), test_ids[:64].to(device))
    print(f"  max |logit difference| {(reference - fused).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
