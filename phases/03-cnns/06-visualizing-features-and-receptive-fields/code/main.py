from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def load_digits_16(device):
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
    images = F.interpolate(images, size=(16, 16), mode="bilinear", align_corners=False)
    images = (images - images.mean()) / images.std()
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    split = int(0.8 * len(labels))
    train_idx, test_idx = order[:split], order[split:]
    return (images[train_idx].to(device), labels[train_idx].to(device),
            images[test_idx].to(device), labels[test_idx].to(device))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.head = nn.Linear(32, 10)

    def features(self, images):
        x1 = torch.relu(self.conv1(images))
        x2 = torch.relu(self.conv2(F.max_pool2d(x1, 2)))
        x3 = torch.relu(self.conv3(F.max_pool2d(x2, 2)))
        return x1, x2, x3

    def forward(self, images):
        return self.head(self.features(images)[2].mean(dim=(2, 3)))


def theoretical_receptive_fields(layers):
    field, jump = 1, 1
    fields = []
    for name, kernel, stride in layers:
        field += (kernel - 1) * jump
        jump *= stride
        fields.append((name, field, jump))
    return fields


def effective_receptive_field(model, layer_index, size, trials=16):
    total = torch.zeros(size, size)
    device = next(model.parameters()).device
    for _ in range(trials):
        image = torch.randn(1, 1, size, size, device=device, requires_grad=True)
        activation = model.features(image)[layer_index]
        center = activation.shape[-1] // 2
        activation[0, :, center, center].sum().backward()
        total += image.grad[0, 0].abs().cpu()
    return total / total.max()


def saliency_map(model, image, label):
    image = image.clone().unsqueeze(0).requires_grad_(True)
    model(image)[0, label].backward()
    return image.grad[0, 0].abs()


def occlusion_map(model, image, label, patch=4, stride=2):
    size = image.shape[-1]
    baseline = torch.softmax(model(image.unsqueeze(0)), dim=1)[0, label].item()
    positions = range(0, size - patch + 1, stride)
    heat = torch.zeros(len(positions), len(positions))
    with torch.no_grad():
        for i, row in enumerate(positions):
            for j, col in enumerate(positions):
                occluded = image.clone()
                occluded[:, row:row + patch, col:col + patch] = image.min()
                prob = torch.softmax(model(occluded.unsqueeze(0)), dim=1)[0, label].item()
                heat[i, j] = baseline - prob
    return heat


def maximize_class(model, label, size, steps=300, lr=0.05):
    device = next(model.parameters()).device
    image = torch.zeros(1, 1, size, size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=lr)
    for _ in range(steps):
        score = model(image)[0, label] - 0.05 * image.pow(2).sum()
        optimizer.zero_grad()
        (-score).backward()
        optimizer.step()
    return image.detach()[0, 0]


def train(model, images, labels, epochs=40, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(1)
    for _ in range(epochs):
        order = torch.randperm(len(labels), generator=generator).to(labels.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = F.cross_entropy(model(images[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_images, train_labels, test_images, test_labels = load_digits_16(device)

    layers = [("conv1 3x3", 3, 1), ("pool1 2x2", 2, 2), ("conv2 3x3", 3, 1), ("pool2 2x2", 2, 2), ("conv3 3x3", 3, 1)]
    print("Theoretical receptive field r and jump j (input pixels between neighboring units):")
    for name, field, jump in theoretical_receptive_fields(layers):
        print(f"  {name:<10} r={field:>2}  j={jump}")

    model = SmallCNN().to(device)
    untrained_erf = effective_receptive_field(model, 2, 32)
    train(model, train_images, train_labels)
    model.eval()
    with torch.no_grad():
        test_accuracy = (model(test_images).argmax(dim=1) == test_labels).float().mean().item()
    print(f"\nTrained SmallCNN test accuracy: {test_accuracy:.3f}")
    trained_erf = effective_receptive_field(model, 2, 32)

    for name, erf in [("untrained", untrained_erf), ("trained", trained_erf)]:
        nonzero = (erf > 0).nonzero()
        width = (nonzero[:, 0].max() - nonzero[:, 0].min() + 1).item()
        mass_center = erf[12:20, 12:20].sum() / erf.sum()
        print(f"Effective RF of conv3 ({name}): nonzero width {width}, "
              f"{mass_center:.0%} of gradient mass in its central 8x8 (uniform would be {64 / width ** 2:.0%})")

    image, label = test_images[0], test_labels[0].item()
    x1, x2, x3 = model.features(image.unsqueeze(0))
    print(f"\nFeature map shapes for one digit: conv1 {tuple(x1.shape)}, conv2 {tuple(x2.shape)}, conv3 {tuple(x3.shape)}")
    print(f"Fraction of zero activations after ReLU: conv1 {(x1 == 0).float().mean():.0%}, "
          f"conv2 {(x2 == 0).float().mean():.0%}, conv3 {(x3 == 0).float().mean():.0%}")

    saliency = saliency_map(model, image, label)
    occlusion = occlusion_map(model, image, label)
    row, col = divmod(occlusion.argmax().item(), occlusion.shape[1])
    print(f"Digit {label}: occluding the 4x4 patch at ({row * 2}, {col * 2}) drops p(label) by {occlusion.max():.2f}")

    dreams = [maximize_class(model, digit, 16).cpu() for digit in range(10)]
    with torch.no_grad():
        dream_preds = [model(d.to(device)[None, None]).argmax().item() for d in dreams]
    print(f"Class-maximizing inputs are classified as: {dream_preds}")

    figure, axes = plt.subplots(4, 10, figsize=(15, 6.5))
    for axis in axes.flat:
        axis.axis("off")
    for index in range(8):
        axes[0, index].imshow(model.conv1.weight[index, 0].detach().cpu(), cmap="RdBu")
        axes[0, index].set_title(f"filter {index}", fontsize=8)
        axes[1, index].imshow(x1[0, index].detach().cpu(), cmap="viridis")
        axes[1, index].set_title(f"map {index}", fontsize=8)
    axes[2, 0].imshow(image[0].cpu(), cmap="gray")
    axes[2, 0].set_title(f"input ({label})", fontsize=8)
    axes[2, 1].imshow(saliency.cpu(), cmap="hot")
    axes[2, 1].set_title("saliency", fontsize=8)
    axes[2, 2].imshow(occlusion, cmap="hot")
    axes[2, 2].set_title("occlusion", fontsize=8)
    axes[2, 3].imshow(untrained_erf, cmap="magma")
    axes[2, 3].set_title("ERF untrained", fontsize=8)
    axes[2, 4].imshow(trained_erf, cmap="magma")
    axes[2, 4].set_title("ERF trained", fontsize=8)
    for digit in range(10):
        axes[3, digit].imshow(dreams[digit], cmap="gray")
        axes[3, digit].set_title(f"max class {digit}", fontsize=8)
    figure.tight_layout()
    output_path = Path(__file__).parent / "features.png"
    figure.savefig(output_path, dpi=90)
    print(f"\nSaved {output_path.name}")


if __name__ == "__main__":
    main()
