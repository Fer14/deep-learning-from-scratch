import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torchvision.models import vgg11

VGG11_CONFIG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
TINY_VGG_CONFIG = [16, 16, "M", 32, 32, "M", 64, 64, "M"]


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


def make_features(config, in_channels, kernel_size=3):
    layers = []
    channels = in_channels
    for item in config:
        if item == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers += [nn.Conv2d(channels, item, kernel_size=kernel_size, padding=kernel_size // 2), nn.ReLU(inplace=True)]
            channels = item
    return nn.Sequential(*layers), channels


class VGG(nn.Module):
    def __init__(self, config, in_channels, num_classes, pooled_size, hidden, kernel_size=3):
        super().__init__()
        self.features, channels = make_features(config, in_channels, kernel_size)
        self.pool = nn.AdaptiveAvgPool2d(pooled_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * pooled_size * pooled_size, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))


class WideShallowNet(nn.Module):
    def __init__(self, channels=48, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size=7, padding=3)
        self.classifier = nn.Linear(channels * 4 * 4, num_classes)

    def forward(self, images):
        return self.classifier(F.max_pool2d(torch.relu(self.conv(images)), 4).flatten(1))


def receptive_field(config):
    field, jump = 1, 1
    for item in config:
        if item == "M":
            field += (2 - 1) * jump
            jump *= 2
        else:
            field += (3 - 1) * jump
    return field


def empirical_receptive_field(model, size):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.abs_()
    image = torch.rand(1, 1, size, size, requires_grad=True)
    output = model.features(image)
    center = output.shape[-1] // 2
    output[0, :, center, center].sum().backward()
    rows = image.grad[0, 0].abs().sum(dim=1).nonzero()
    return (rows.max() - rows.min() + 1).item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def accuracy(model, images, labels):
    model.eval()
    with torch.no_grad():
        return (model(images).argmax(dim=1) == labels).float().mean().item()


def train(model, train_images, train_labels, test_images, test_labels, epochs=20, batch_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator).to(train_labels.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = F.cross_entropy(model(train_images[batch]), train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 4:
            print(f"    epoch {epoch + 1:>2}  test acc {accuracy(model, test_images, test_labels):.3f}")
    return accuracy(model, test_images, test_labels)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    channels = 64
    print("Same receptive field, different cost (C=64 channels in and out):")
    for name, kernels in [("one 5x5", [5]), ("two 3x3", [3, 3]), ("one 7x7", [7]), ("three 3x3", [3, 3, 3])]:
        params = sum(k * k * channels * channels for k in kernels)
        field = 1 + sum(k - 1 for k in kernels)
        print(f"  {name:<10} receptive field {field}x{field}  weights {params:>7,}  nonlinearities {len(kernels)}")

    print("\nReceptive field of one output unit, from the formula vs from gradients:")
    for depth in [1, 2, 3, 4]:
        config = [8] * depth
        model = VGG(config, 1, 10, 1, 16)
        print(f"  {depth} stacked 3x3 convs: formula {receptive_field(config)}, "
              f"gradient {empirical_receptive_field(model, 32)}")
    print(f"  tiny VGG {TINY_VGG_CONFIG}: formula {receptive_field(TINY_VGG_CONFIG)} "
          f"(pooling doubles the step of every later layer)")

    train_images, train_labels, test_images, test_labels = load_digits_16(device)
    print(f"\nDigits upsampled to {tuple(train_images.shape[1:])} on {device}")

    results = []
    for name, build in [("tiny VGG, 3x3 kernels", lambda: VGG(TINY_VGG_CONFIG, 1, 10, 2, 64)),
                        ("same layout, 5x5 kernels", lambda: VGG(TINY_VGG_CONFIG, 1, 10, 2, 64, kernel_size=5)),
                        ("wide shallow 7x7 net", WideShallowNet)]:
        torch.manual_seed(0)
        model = build().to(device)
        print(f"\nTraining {name}, {count_parameters(model):,} parameters")
        results.append((name, count_parameters(model), train(model, train_images, train_labels, test_images, test_labels)))
    print(f"\n{'model':<26} {'params':>8} {'test acc':>9}")
    for name, params, test_accuracy in results:
        print(f"{name:<26} {params:>8,} {test_accuracy:>9.3f}")

    my_vgg11 = VGG(VGG11_CONFIG, 3, 1000, 7, 4096)
    library_vgg11 = vgg11(weights=None)
    feature_params = count_parameters(my_vgg11.features)
    print(f"\nVGG-11: mine {count_parameters(my_vgg11):,} params, torchvision {count_parameters(library_vgg11):,} params")
    print(f"  convolutional features: {feature_params:,} ({feature_params / count_parameters(my_vgg11):.0%})")
    print(f"  fully connected head:   {count_parameters(my_vgg11.classifier):,} "
          f"({count_parameters(my_vgg11.classifier) / count_parameters(my_vgg11):.0%})")
    with torch.no_grad():
        logits = library_vgg11.eval()(torch.randn(1, 3, 64, 64))
    print(f"  torchvision vgg11 on a 64x64 image -> logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
