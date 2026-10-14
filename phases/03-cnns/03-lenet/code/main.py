import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def load_digits_32(device):
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
    images = F.interpolate(images, size=(28, 28), mode="bilinear", align_corners=False)
    images = F.pad(images, (2, 2, 2, 2))
    images = (images - images.mean()) / images.std()
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    split = int(0.8 * len(labels))
    train_idx, test_idx = order[:split], order[split:]
    return (images[train_idx].to(device), labels[train_idx].to(device),
            images[test_idx].to(device), labels[test_idx].to(device))


class ScratchConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        bound = (1.0 / fan_in) ** 0.5
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_channels).uniform_(-bound, bound))

    def forward(self, images):
        batch, channels, height, width = images.shape
        k = self.kernel_size
        out_height, out_width = height - k + 1, width - k + 1
        patches = images.unfold(2, k, 1).unfold(3, k, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch, out_height, out_width, -1)
        output = patches @ self.weight.reshape(self.weight.shape[0], -1).T + self.bias
        return output.permute(0, 3, 1, 2)


def scratch_max_pool(images, size=2):
    batch, channels, height, width = images.shape
    windows = images.reshape(batch, channels, height // size, size, width // size, size)
    return windows.amax(dim=(3, 5))


class ScratchLeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = ScratchConv2d(1, 6, 5)
        self.conv2 = ScratchConv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, images):
        x = scratch_max_pool(torch.relu(self.conv1(images)))
        x = scratch_max_pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def shapes(self, images):
        trace = [("input", images.shape)]
        x = self.conv1(images)
        trace.append(("conv1 5x5 x6", x.shape))
        x = scratch_max_pool(torch.relu(x))
        trace.append(("pool 2x2", x.shape))
        x = self.conv2(x)
        trace.append(("conv2 5x5 x16", x.shape))
        x = scratch_max_pool(torch.relu(x))
        trace.append(("pool 2x2", x.shape))
        x = x.flatten(1)
        trace.append(("flatten", x.shape))
        x = self.fc1(x)
        trace.append(("fc 120", x.shape))
        x = self.fc2(torch.relu(x))
        trace.append(("fc 84", x.shape))
        trace.append(("fc 10", self.fc3(torch.relu(x)).shape))
        return trace


def torch_lenet5(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, num_classes),
    )


def mlp_baseline(num_classes=10):
    return nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 64), nn.ReLU(), nn.Linear(64, num_classes))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def accuracy(model, images, labels):
    model.eval()
    with torch.no_grad():
        return (model(images).argmax(dim=1) == labels).float().mean().item()


def train(model, train_images, train_labels, test_images, test_labels, epochs=15, batch_size=64, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator).to(train_labels.device)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = F.cross_entropy(model(train_images[batch]), train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        if epoch % 5 == 4:
            print(f"    epoch {epoch + 1:>2}  loss {total_loss / len(order):.4f}  "
                  f"test acc {accuracy(model, test_images, test_labels):.3f}")
    return accuracy(model, test_images, test_labels)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_images, train_labels, test_images, test_labels = load_digits_32(device)
    print(f"Device: {device}. Train {tuple(train_images.shape)}, test {tuple(test_images.shape)}")

    scratch_model = ScratchLeNet5().to(device)
    print("\nLeNet-5 shapes:")
    for name, shape in scratch_model.shapes(train_images[:1]):
        print(f"  {name:<14} {tuple(shape)}")

    reference_conv = nn.Conv2d(1, 6, 5).to(device)
    with torch.no_grad():
        reference_conv.weight.copy_(scratch_model.conv1.weight)
        reference_conv.bias.copy_(scratch_model.conv1.bias)
        diff = (reference_conv(train_images[:8]) - scratch_model.conv1(train_images[:8])).abs().max()
        pool_diff = (F.max_pool2d(train_images[:8], 2) - scratch_max_pool(train_images[:8])).abs().max()
    print(f"\nScratchConv2d vs nn.Conv2d max diff: {diff:.1e}, scratch pool vs F.max_pool2d: {pool_diff:.1e}")

    results = []
    for name, model in [("LeNet-5 (scratch conv)", scratch_model),
                        ("LeNet-5 (nn.Conv2d)", torch_lenet5().to(device)),
                        ("MLP baseline", mlp_baseline().to(device))]:
        torch.manual_seed(0)
        print(f"\nTraining {name} ({count_parameters(model):,} parameters)")
        start = time.perf_counter()
        test_accuracy = train(model, train_images, train_labels, test_images, test_labels)
        results.append((name, model, test_accuracy, time.perf_counter() - start))

    shifted_test = torch.roll(test_images, shifts=(3, 3), dims=(2, 3))
    print(f"\n{'model':<24} {'params':>8} {'test acc':>9} {'shifted 3px':>12} {'seconds':>8}")
    for name, model, test_accuracy, seconds in results:
        shifted_accuracy = accuracy(model, shifted_test, test_labels)
        print(f"{name:<24} {count_parameters(model):>8,} {test_accuracy:>9.3f} {shifted_accuracy:>12.3f} {seconds:>8.1f}")


if __name__ == "__main__":
    main()
