import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEMPLATES = ["a handwritten {}", "the digit {}", "number {}"]
SPECIAL = ["<pad>", "<t2i>", "<i2t>", "<img>", "<txt>", "<eos>"]
CODEBOOK_SIZE = 48
PATCH = 2


def image_to_patches(images):
    batch = images.shape[0]
    patches = images.view(batch, 8 // PATCH, PATCH, 8 // PATCH, PATCH).permute(0, 1, 3, 2, 4)
    return patches.reshape(batch, -1, PATCH * PATCH)


def patches_to_image(patches):
    batch = patches.shape[0]
    grid = patches.view(batch, 8 // PATCH, 8 // PATCH, PATCH, PATCH).permute(0, 1, 3, 2, 4)
    return grid.reshape(batch, 8, 8)


def kmeans(points, k, iterations, generator):
    centroids = points[torch.randperm(len(points), generator=generator)[:k]].clone()
    for _ in range(iterations):
        assignment = torch.cdist(points, centroids).argmin(1)
        for j in range(k):
            members = points[assignment == j]
            if len(members) > 0:
                centroids[j] = members.mean(0)
    inertia = ((points - centroids[torch.cdist(points, centroids).argmin(1)]) ** 2).sum().item()
    return centroids, inertia


class ImageTokenizer:
    def __init__(self, centroids):
        self.centroids = centroids

    def encode(self, images):
        patches = image_to_patches(images)
        return torch.cdist(patches, self.centroids.unsqueeze(0).expand(len(patches), -1, -1)).argmin(-1)

    def decode(self, codes):
        return patches_to_image(self.centroids[codes])


class Vocabulary:
    def __init__(self):
        words = sorted({w for t in TEMPLATES for w in t.format("").split()} | set(DIGIT_NAMES))
        self.tokens = SPECIAL + words + [f"<code{i}>" for i in range(CODEBOOK_SIZE)]
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.code_offset = len(SPECIAL) + len(words)

    def text_ids(self, caption):
        return [self.index[w] for w in caption.split()]


def build_sequence(vocab, task, caption, codes):
    text = vocab.text_ids(caption)
    image = [vocab.code_offset + c for c in codes]
    if task == "t2i":
        prefix = [vocab.index["<t2i>"]] + text + [vocab.index["<img>"]]
        target = image + [vocab.index["<eos>"]]
    else:
        prefix = [vocab.index["<i2t>"]] + image + [vocab.index["<txt>"]]
        target = text + [vocab.index["<eos>"]]
    return prefix, target


def make_batch(vocab, codes, labels, rng, length):
    ids = torch.zeros(len(labels), length, dtype=torch.long)
    targets = torch.full((len(labels), length), -100, dtype=torch.long)
    for row, (image_codes, label) in enumerate(zip(codes.tolist(), labels.tolist())):
        task = "t2i" if rng.random() < 0.5 else "i2t"
        caption = TEMPLATES[rng.integers(len(TEMPLATES))].format(DIGIT_NAMES[label])
        prefix, target = build_sequence(vocab, task, caption, image_codes)
        sequence = prefix + target
        ids[row, : len(sequence)] = torch.tensor(sequence)
        targets[row, len(prefix) - 1 : len(sequence) - 1] = torch.tensor(target)
    return ids, targets


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, length, embed_dim = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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


class AnyToAnyGPT(nn.Module):
    def __init__(self, vocab_size, max_length=24, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, ids):
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.ln_final(self.blocks(x)))


@torch.no_grad()
def generate(model, prefix, steps, allowed, generator, temperature=1.0):
    ids = prefix
    for _ in range(steps):
        logits = model(ids)[:, -1]
        logits[:, ~allowed] = float("-inf")
        if temperature == 0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            next_id = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1, generator=generator)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[:, prefix.shape[1] :]


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32)
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels))
    train_index, test_index = order[:1500], order[1500:]

    print(f"device {device}")
    print("step 1: an image tokenizer: 2x2 patches -> nearest of 48 k-means codes")
    patches = image_to_patches(images[train_index]).reshape(-1, PATCH * PATCH)
    centroids, inertia = kmeans(patches, CODEBOOK_SIZE, 25, torch.Generator().manual_seed(0))
    tokenizer = ImageTokenizer(centroids)
    codes = tokenizer.encode(images)
    reconstruction = tokenizer.decode(codes)
    print(f"  one 8x8 image = {codes.shape[1]} tokens | reconstruction mse {F.mse_loss(reconstruction, images).item():.4f} (pixel variance {images.var().item():.4f})")
    probe = LogisticRegression(max_iter=5000).fit(images[train_index].flatten(1).numpy(), labels[train_index].numpy())
    print(f"  a digit classifier trained on real pixels scores {probe.score(reconstruction[test_index].flatten(1).numpy(), labels[test_index].numpy()):.3f} on tokenized-then-decoded test images")

    print("\nstep 2: one vocabulary, two tasks, one sequence format")
    vocab = Vocabulary()
    prefix, target = build_sequence(vocab, "t2i", "a handwritten seven", codes[0].tolist())
    print(f"  vocab {len(vocab.tokens)} = {len(SPECIAL)} special + {vocab.code_offset - len(SPECIAL)} words + {CODEBOOK_SIZE} image codes")
    print(f"  text->image: {' '.join(vocab.tokens[i] for i in prefix)} | {' '.join(vocab.tokens[i] for i in target[:4])} ... <eos>")
    prefix, target = build_sequence(vocab, "i2t", "a handwritten seven", codes[0].tolist())
    print(f"  image->text: {' '.join(vocab.tokens[i] for i in prefix[:4])} ... <txt> | {' '.join(vocab.tokens[i] for i in target)}")

    print("\nstep 3: train one decoder-only transformer on both directions")
    model = AnyToAnyGPT(len(vocab.tokens)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    train_codes, train_labels = codes[train_index], labels[train_index]
    for step in range(1001):
        batch = torch.randint(len(train_index), (64,))
        ids, targets = make_batch(vocab, train_codes[batch], train_labels[batch], rng, 24)
        logits = model(ids.to(device))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(device).reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            print(f"  step {step:4d} | loss {loss.item():.3f}")
    model.eval()
    print(f"  parameters {sum(p.numel() for p in model.parameters()):,}")

    print("\nstep 4: image -> text on test digits")
    generator = torch.Generator(device=device).manual_seed(0)
    word_allowed = torch.zeros(len(vocab.tokens), dtype=torch.bool, device=device)
    word_allowed[len(SPECIAL) : vocab.code_offset] = True
    word_allowed[vocab.index["<eos>"]] = True
    test_codes, test_labels = codes[test_index], labels[test_index]
    prefix = torch.cat([
        torch.full((len(test_index), 1), vocab.index["<i2t>"]),
        test_codes + vocab.code_offset,
        torch.full((len(test_index), 1), vocab.index["<txt>"]),
    ], dim=1).to(device)
    captions = generate(model, prefix, 4, word_allowed, generator, temperature=0)
    predicted_words = [[vocab.tokens[i] for i in row.tolist()] for row in captions]
    correct = [DIGIT_NAMES[label] in words for words, label in zip(predicted_words, test_labels.tolist())]
    print(f"  caption names the right digit for {np.mean(correct):.3f} of {len(test_index)} test images")
    for words, label in list(zip(predicted_words, test_labels.tolist()))[:4]:
        print(f"  true {label}: {' '.join(w for w in words if w != '<eos>')}")

    print("\nstep 5: text -> image, graded by the pixel classifier")
    code_allowed = torch.zeros(len(vocab.tokens), dtype=torch.bool, device=device)
    code_allowed[vocab.code_offset :] = True
    requested = torch.arange(10).repeat_interleave(20)
    text_prefix = torch.tensor([[vocab.index["<t2i>"]] + vocab.text_ids(f"a handwritten {DIGIT_NAMES[d]}") + [vocab.index["<img>"]] for d in requested.tolist()], device=device)
    generated_images = {}
    for temperature in [0.7, 1.0]:
        generated_codes = generate(model, text_prefix, 16, code_allowed, generator, temperature=temperature) - vocab.code_offset
        generated_images[temperature] = tokenizer.decode(generated_codes.cpu())
        accuracy = probe.score(generated_images[temperature].flatten(1).numpy(), requested.numpy())
        print(f"  temperature {temperature}: classifier agrees with the requested digit on {accuracy:.3f} of 200 generated images")

    print("\nuse it: compare the hand k-means with scikit-learn")
    library = KMeans(n_clusters=CODEBOOK_SIZE, n_init=1, random_state=0).fit(patches.numpy())
    print(f"  inertia: ours {inertia:.1f} | sklearn KMeans {library.inertia_:.1f}")

    fig, axes = plt.subplots(4, 10, figsize=(10, 4.4))
    for d in range(10):
        axes[0, d].imshow(images[test_index][test_labels == d][0], cmap="gray_r")
        axes[0, d].set_title(DIGIT_NAMES[d], fontsize=8)
        axes[1, d].imshow(reconstruction[test_index][test_labels == d][0], cmap="gray_r")
        for row in range(2):
            axes[row + 2, d].imshow(generated_images[0.7][d * 20 + row], cmap="gray_r")
    for ax in axes.flat:
        ax.axis("off")
    for row, label in enumerate(["real", "tokenized", "generated", "generated"]):
        axes[row, 0].text(-3, 3.5, label, fontsize=8, ha="right", va="center")
    fig.tight_layout()
    out_path = Path(__file__).parent / "any_to_any.png"
    fig.savefig(out_path, dpi=110)
    print(f"\nsaved {out_path.name}")


if __name__ == "__main__":
    main()
