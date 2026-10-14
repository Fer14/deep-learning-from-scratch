import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TRAIN_TEMPLATES = [
    "a handwritten {}",
    "the digit {}",
    "a scan of the number {}",
    "a small picture of a {}",
    "someone wrote a {}",
]
NEW_TEMPLATE = "a blurry drawing of the number {}"


class WordTokenizer:
    def __init__(self, sentences):
        words = sorted({w for s in sentences for w in s.split()})
        self.vocab = ["<pad>", "<unk>"] + words
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}

    def encode(self, sentences, length):
        ids = torch.zeros(len(sentences), length, dtype=torch.long)
        for row, sentence in enumerate(sentences):
            tokens = [self.word_to_id.get(w, 1) for w in sentence.split()][:length]
            ids[row, : len(tokens)] = torch.tensor(tokens)
        return ids


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, embed_dim))

    def forward(self, images):
        return self.net(images.flatten(1))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.net = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, embed_dim))

    def forward(self, ids):
        mask = (ids != 0).float().unsqueeze(-1)
        pooled = (self.embedding(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.net(pooled)


class CLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, ids):
        return F.normalize(self.text_encoder(ids), dim=-1)

    def forward(self, images, ids):
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * self.encode_image(images) @ self.encode_text(ids).T


def symmetric_info_nce(logits):
    n = logits.shape[0]
    diagonal = logits.diagonal()
    image_to_text = (torch.logsumexp(logits, dim=1) - diagonal).mean()
    text_to_image = (torch.logsumexp(logits, dim=0) - diagonal).mean()
    return (image_to_text + text_to_image) / 2


def make_captions(labels, rng):
    templates = rng.integers(0, len(TRAIN_TEMPLATES), size=len(labels))
    return [TRAIN_TEMPLATES[t].format(DIGIT_NAMES[y]) for t, y in zip(templates, labels)]


@torch.no_grad()
def class_text_embeddings(model, tokenizer, templates, device):
    embeddings = []
    for name in DIGIT_NAMES:
        ids = tokenizer.encode([t.format(name) for t in templates], 8).to(device)
        embeddings.append(F.normalize(model.encode_text(ids).mean(0), dim=-1))
    return torch.stack(embeddings)


@torch.no_grad()
def zero_shot_accuracy(model, images, labels, text_embeddings):
    predictions = (model.encode_image(images) @ text_embeddings.T).argmax(1)
    return (predictions == labels).float().mean().item()


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32)
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels))
    train_index, test_index = order[:1400], order[1400:]

    all_sentences = [t.format(n) for t in TRAIN_TEMPLATES for n in DIGIT_NAMES]
    tokenizer = WordTokenizer(all_sentences)
    print(f"device {device} | {len(train_index)} train / {len(test_index)} test images | text vocab {len(tokenizer.vocab)}")
    example_captions = make_captions(labels[train_index[:3]].numpy(), rng)
    print(f"example pairs: {list(zip(labels[train_index[:3]].tolist(), example_captions))}")

    print("\nstep 1: train with the symmetric contrastive loss")
    model = CLIP(len(tokenizer.vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    train_images, train_labels = images[train_index].to(device), labels[train_index]
    test_images, test_labels = images[test_index].to(device), labels[test_index].to(device)
    batch_size = 128
    for step in range(601):
        batch = torch.randint(len(train_index), (batch_size,))
        captions = make_captions(train_labels[batch].numpy(), rng)
        ids = tokenizer.encode(captions, 8).to(device)
        logits = model(train_images[batch], ids)
        loss = symmetric_info_nce(logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 150 == 0:
            with torch.no_grad():
                same_class = (train_labels[batch].unsqueeze(0) == train_labels[batch].unsqueeze(1)).float().sum(1).mean()
            print(f"  step {step:3d} | loss {loss.item():.3f} (chance ln {batch_size} = {math.log(batch_size):.2f}) | temperature {1 / model.logit_scale.exp().item():.3f} | captions per batch sharing the image's class {same_class:.1f}")

    print("\nstep 2: zero-shot classification by comparing images with caption embeddings")
    model.eval()
    single = class_text_embeddings(model, tokenizer, ["a handwritten {}"], device)
    ensemble = class_text_embeddings(model, tokenizer, TRAIN_TEMPLATES, device)
    unseen = class_text_embeddings(model, tokenizer, [NEW_TEMPLATE], device)
    print(f"  prompt 'a handwritten {{}}'        test accuracy {zero_shot_accuracy(model, test_images, test_labels, single):.3f}")
    print(f"  ensemble of 5 training prompts  test accuracy {zero_shot_accuracy(model, test_images, test_labels, ensemble):.3f}")
    print(f"  unseen prompt {NEW_TEMPLATE!r}: {zero_shot_accuracy(model, test_images, test_labels, unseen):.3f}")
    unknown = [w for w in NEW_TEMPLATE.split() if w not in tokenizer.word_to_id and w != "{}"]
    print(f"  words of the unseen prompt mapped to <unk>: {unknown}")

    print("\nstep 3: text-to-image retrieval")
    with torch.no_grad():
        image_embeddings = model.encode_image(test_images)
        for query in ["a handwritten seven", "the digit zero", "someone wrote a four"]:
            text = model.encode_text(tokenizer.encode([query], 8).to(device))
            top = (image_embeddings @ text.T).squeeze(1).topk(10).indices
            print(f"  {query!r:24s} top-10 labels {test_labels[top].tolist()}")

    print("\nstep 4: the embedding space")
    with torch.no_grad():
        similarity = ensemble @ ensemble.T
    off_diagonal = similarity[~torch.eye(10, dtype=torch.bool, device=device)]
    print(f"  cosine between class prompts: mean {off_diagonal.mean().item():.3f}, max {off_diagonal.max().item():.3f} (1.0 on the diagonal)")
    with torch.no_grad():
        matched = (image_embeddings * ensemble[test_labels]).sum(1).mean().item()
        mismatched = (image_embeddings * ensemble[(test_labels + 1) % 10]).sum(1).mean().item()
    print(f"  cosine image vs its own class prompt {matched:.3f} | vs another class prompt {mismatched:.3f}")

    print("\nuse it: library loss and a linear probe")
    with torch.no_grad():
        ids = tokenizer.encode(make_captions(train_labels[:256].numpy(), rng), 8).to(device)
        logits = model(train_images[:256], ids)
        targets = torch.arange(256, device=device)
        library = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    print(f"  hand InfoNCE {symmetric_info_nce(logits).item():.5f} | F.cross_entropy both directions {library.item():.5f}")
    with torch.no_grad():
        train_features = model.encode_image(train_images).cpu().numpy()
        test_features = image_embeddings.cpu().numpy()
    probe = LogisticRegression(max_iter=2000).fit(train_features, train_labels.numpy())
    print(f"  linear probe on frozen image embeddings: test accuracy {probe.score(test_features, test_labels.cpu().numpy()):.3f}")
    raw = LogisticRegression(max_iter=5000).fit(images[train_index].flatten(1).numpy(), train_labels.numpy())
    print(f"  logistic regression on raw pixels:       test accuracy {raw.score(images[test_index].flatten(1).numpy(), test_labels.cpu().numpy()):.3f}")


if __name__ == "__main__":
    main()
