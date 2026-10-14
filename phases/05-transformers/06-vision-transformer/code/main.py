import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torchvision.models import VisionTransformer


def load_digit_tensors(device):
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
    images = (images - images.mean()) / images.std()
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    split = int(0.8 * len(labels))
    train_idx, test_idx = order[:split], order[split:]
    return (images[train_idx].to(device), labels[train_idx].to(device),
            images[test_idx].to(device), labels[test_idx].to(device))


def patchify(images, patch_size):
    batch, channels, height, width = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(batch, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size)


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, length, d_model = x.shape
        q, k, v = (t.view(batch, length, self.num_heads, self.d_head).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_head), dim=-1)
        self.last_weights = weights.detach()
        return self.output((weights @ v).transpose(1, 2).reshape(batch, length, d_model))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, mlp_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_size), nn.GELU(), nn.Linear(mlp_size, d_model))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ScratchViT(nn.Module):
    def __init__(self, image_size=8, patch_size=2, channels=1, d_model=48, num_heads=4, num_layers=4, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(channels * patch_size * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, num_heads, 2 * d_model) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, images):
        tokens = self.patch_embed(patchify(images, self.patch_size))
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 2 * 2, 10),
        )

    def forward(self, images):
        return self.layers(images)


class LibraryViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VisionTransformer(image_size=8, patch_size=2, num_layers=4, num_heads=4, hidden_dim=48,
                                       mlp_dim=96, num_classes=10)

    def forward(self, images):
        return self.model(images.repeat(1, 3, 1, 1))


def accuracy(model, images, labels):
    model.eval()
    with torch.no_grad():
        return (model(images).argmax(dim=1) == labels).float().mean().item()


def train(model, images, labels, epochs, generator, batch_size=64, lr=3e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_steps = epochs * math.ceil(len(labels) / batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(labels), generator=generator).to(labels.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            shift = torch.randint(-1, 2, (2,), generator=generator).tolist()
            batch_images = torch.roll(images[batch], shifts=shift, dims=(2, 3))
            loss = F.cross_entropy(model(batch_images), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(0)
    train_images, train_labels, test_images, test_labels = load_digit_tensors(device)

    patches = patchify(train_images[:1], 2)
    print(f"An 8x8 digit -> {patches.shape[1]} patches of 2x2 = {patches.shape[2]} values each")
    print(f"Patch at row 0, column 1 of the first digit, flattened: {[round(v, 1) for v in patches[0, 1].tolist()]}")

    torch.manual_seed(0)
    model = ScratchViT().to(device)
    conv = nn.Conv2d(1, 48, kernel_size=2, stride=2).to(device)
    with torch.no_grad():
        conv.weight.copy_(model.patch_embed.weight.view(48, 1, 2, 2))
        conv.bias.copy_(model.patch_embed.bias)
        via_conv = conv(train_images[:4]).flatten(2).transpose(1, 2)
        via_linear = model.patch_embed(patchify(train_images[:4], 2))
    print(f"Linear patch embedding == Conv2d(kernel=stride=patch): max diff {(via_conv - via_linear).abs().max():.1e}")

    sequence_length = patches.shape[1] + 1
    print(f"Sequence entering the encoder: {sequence_length} tokens (1 [CLS] + {patches.shape[1]} patches) of width 48")

    results = []
    for name, build in [("ViT (yours)", ScratchViT), ("ViT (torchvision)", LibraryViT), ("small CNN", SmallCNN)]:
        torch.manual_seed(0)
        model = build().to(device)
        start = time.perf_counter()
        train(model, train_images, train_labels, 40, generator)
        results.append((name, model, accuracy(model, test_images, test_labels), time.perf_counter() - start))

    print(f"\n{'model':<19} {'params':>8} {'test acc':>9} {'seconds':>8}")
    for name, model, test_accuracy, seconds in results:
        print(f"{name:<19} {sum(p.numel() for p in model.parameters()):>8,} {test_accuracy:>9.3f} {seconds:>8.1f}")

    vit = results[0][1]
    vit.eval()
    with torch.no_grad():
        vit(test_images[:256])
        cls_attention = vit.blocks[-1].attention.last_weights[:, :, 0, 1:].mean(dim=(0, 1)).view(4, 4)
    print("\nLast block: average attention from [CLS] to each of the 4x4 patches:")
    for row in cls_attention.tolist():
        print("  " + " ".join(f"{value:.3f}" for value in row))

    position = vit.position[0, 1:].detach()
    similarity = F.normalize(position, dim=1) @ F.normalize(position, dim=1).T
    print("\nCosine similarity of learned position embeddings to patch (1, 1):")
    for row in similarity[5].view(4, 4).tolist():
        print("  " + " ".join(f"{value:>5.2f}" for value in row))


if __name__ == "__main__":
    main()
