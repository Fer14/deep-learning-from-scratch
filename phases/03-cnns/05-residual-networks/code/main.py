import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torchvision.models import resnet18


def load_digit_tensors(device, size=8):
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
    if size != 8:
        images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    images = (images - images.mean()) / images.std()
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    split = int(0.8 * len(labels))
    train_idx, test_idx = order[:split], order[split:]
    return (images[train_idx].to(device), labels[train_idx].to(device),
            images[test_idx].to(device), labels[test_idx].to(device))


class PlainNet(nn.Module):
    def __init__(self, num_layers, width=16, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.layers = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(num_layers))
        self.head = nn.Linear(width, num_classes)

    def forward(self, images):
        x = torch.relu(self.stem(images))
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x.mean(dim=(2, 3)))


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class ResidualNet(nn.Module):
    def __init__(self, num_layers, width=16, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(num_layers // 2))
        self.head = nn.Linear(width, num_classes)
        self.layers = [conv for block in self.blocks for conv in (block.conv1, block.conv2)]

    def forward(self, images):
        x = torch.relu(self.stem(images))
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=(2, 3)))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


def gradient_norms(model, images, labels):
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    return [layer.weight.grad.norm().item() for layer in model.layers]


def activation_stds(model, images):
    stds = []
    with torch.no_grad():
        x = torch.relu(model.stem(images))
        if isinstance(model, PlainNet):
            for layer in model.layers:
                x = torch.relu(layer(x))
                stds.append(x.std().item())
        else:
            for block in model.blocks:
                x = block(x)
                stds.append(x.std().item())
                stds.append(x.std().item())
    return stds


def accuracy(model, images, labels):
    model.eval()
    with torch.no_grad():
        return (model(images).argmax(dim=1) == labels).float().mean().item()


def train(model, train_images, train_labels, epochs, batch_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator).to(train_labels.device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = F.cross_entropy(model(train_images[batch]), train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(order))
    return losses


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_images, train_labels, test_images, test_labels = load_digit_tensors(device)
    probe_images, probe_labels = train_images[:256], train_labels[:256]

    print("Gradient norm of each conv layer's weights at initialization (depth 30):")
    torch.manual_seed(0)
    plain = PlainNet(30).to(device)
    torch.manual_seed(0)
    residual = ResidualNet(30).to(device)
    plain_grads = gradient_norms(plain, probe_images, probe_labels)
    residual_grads = gradient_norms(residual, probe_images, probe_labels)
    plain_stds = activation_stds(plain, probe_images)
    residual_stds = activation_stds(residual, probe_images)
    print(f"{'layer':>6} {'plain grad':>12} {'resid grad':>12} {'plain act std':>14} {'resid act std':>14}")
    for index in [0, 4, 9, 14, 19, 24, 29]:
        print(f"{index + 1:>6} {plain_grads[index]:>12.2e} {residual_grads[index]:>12.2e} "
              f"{plain_stds[index]:>14.2e} {residual_stds[index]:>14.2e}")

    print("\nTraining plain vs residual networks of increasing depth (12 epochs each):")
    print(f"{'depth':>6} {'plain loss':>11} {'plain acc':>10} {'resid loss':>11} {'resid acc':>10}")
    for depth in [6, 16, 30]:
        row = []
        for network in (PlainNet, ResidualNet):
            torch.manual_seed(0)
            model = network(depth).to(device)
            losses = train(model, train_images, train_labels, epochs=12)
            row += [losses[-1], accuracy(model, test_images, test_labels)]
        print(f"{depth:>6} {row[0]:>11.4f} {row[1]:>10.3f} {row[2]:>11.4f} {row[3]:>10.3f}")

    block = BasicBlock(16, 32, stride=2)
    print(f"\nBasic block with projection shortcut: {(4, 16, 8, 8)} -> "
          f"{tuple(block(torch.randn(4, 16, 8, 8)).shape)}")

    torch.manual_seed(0)
    train_images_32, train_labels_32, test_images_32, test_labels_32 = load_digit_tensors(device, size=32)
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"torchvision resnet18(weights=None) with a 1-channel 3x3 stem: {params:,} params")
    if device == "cuda":
        losses = train(model, train_images_32, train_labels_32, epochs=3, batch_size=128)
        print(f"  3 epochs on 32x32 digits: loss {losses[-1]:.3f}, "
              f"test acc {accuracy(model, test_images_32, test_labels_32):.3f}")
    else:
        with torch.no_grad():
            logits = model.eval()(test_images_32[:8])
        print(f"  no GPU, forward pass only: {tuple(test_images_32[:8].shape)} -> {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
