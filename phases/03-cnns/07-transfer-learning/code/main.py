import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torchvision.models import resnet18


def load_digits_16():
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
    images = F.interpolate(images, size=(16, 16), mode="bilinear", align_corners=False)
    images = (images - images.mean()) / images.std()
    return images, torch.tensor(digits.target)


def random_affine(images, generator, max_angle=0.35, max_scale=0.15, max_shift=0.2):
    count = len(images)
    angle = (torch.rand(count, generator=generator) * 2 - 1) * max_angle
    scale = 1 + (torch.rand(count, generator=generator) * 2 - 1) * max_scale
    shift = (torch.rand(count, 2, generator=generator) * 2 - 1) * max_shift
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = torch.cos(angle) / scale
    theta[:, 0, 1] = -torch.sin(angle) / scale
    theta[:, 1, 0] = torch.sin(angle) / scale
    theta[:, 1, 1] = torch.cos(angle) / scale
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta.to(images.device), list(images.shape), align_corners=False)
    background = images.min()
    return F.grid_sample(images - background, grid, align_corners=False) + background


def pretrain(images, labels, epochs, generator, batch_size=64):
    model = Classifier(Backbone(), 5).to(images.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        model.train()
        augmented = random_affine(images, generator)
        order = torch.randperm(len(labels), generator=generator).to(labels.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = F.cross_entropy(model(augmented[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.out_features = 256

    def forward(self, images):
        return self.layers(images)


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_features, num_classes)

    def forward(self, images):
        return self.head(self.backbone(images))


def train(model, images, labels, epochs, lr, parameters=None, batch_size=32, seed=1):
    optimizer = torch.optim.Adam(parameters if parameters is not None else model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(labels), generator=generator).to(labels.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = F.cross_entropy(model(images[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def accuracy(model, images, labels):
    model.eval()
    with torch.no_grad():
        return (model(images).argmax(dim=1) == labels).float().mean().item()


def few_shot_split(images, labels, shots, seed):
    generator = torch.Generator().manual_seed(seed)
    train_idx, test_idx = [], []
    for label in labels.unique():
        members = (labels == label).nonzero().flatten()
        members = members[torch.randperm(len(members), generator=generator)]
        train_idx.append(members[:shots])
        test_idx.append(members[60:])
    train_idx, test_idx = torch.cat(train_idx), torch.cat(test_idx)
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]


def set_frozen(module, frozen):
    for parameter in module.parameters():
        parameter.requires_grad = not frozen
    if frozen:
        module.eval()


def linear_probe(pretrained, images, labels, epochs):
    model = Classifier(copy.deepcopy(pretrained), 5).to(images.device)
    set_frozen(model.backbone, True)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)
    for _ in range(epochs):
        model.head.train()
        loss = F.cross_entropy(model.head(model.backbone(images)), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def fine_tune(pretrained, images, labels, epochs):
    model = Classifier(copy.deepcopy(pretrained), 5).to(images.device)
    parameter_groups = [{"params": model.backbone.parameters(), "lr": 3e-4},
                        {"params": model.head.parameters(), "lr": 3e-3}]
    train(model, images, labels, epochs, lr=3e-4, parameters=parameter_groups)
    return model


def from_scratch(images, labels, epochs):
    model = Classifier(Backbone(), 5).to(images.device)
    train(model, images, labels, epochs, lr=3e-3)
    return model


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_digits_16()
    images, labels = images.to(device), labels.to(device)

    generator = torch.Generator().manual_seed(0)
    source_mask = labels < 5
    source_images, source_labels = images[source_mask], labels[source_mask]
    target_images = random_affine(images[~source_mask], generator)
    target_labels = labels[~source_mask] - 5
    print(f"Source task: digits 0-4, {len(source_labels)} images, trained with random rotations/scales/shifts")
    print(f"Target task: digits 5-9, {len(target_labels)} randomly transformed images, only a few labels per class")

    torch.manual_seed(0)
    source_model = pretrain(source_images, source_labels, 30, generator)
    print(f"Source model accuracy on transformed 0-4: "
          f"{accuracy(source_model, random_affine(source_images, generator), source_labels):.3f}")
    pretrained = source_model.backbone
    torch.manual_seed(0)
    random_backbone = Backbone().to(device)

    print("\nTarget test accuracy on digits 5-9 (mean over 3 random label subsets):")
    print(f"{'labels/class':>13} {'scratch':>9} {'random probe':>13} {'linear probe':>13} {'fine-tune':>10}")
    for shots in [1, 3, 10, 30]:
        scores = {"scratch": [], "random": [], "probe": [], "finetune": []}
        for seed in range(3):
            train_x, train_y, test_x, test_y = few_shot_split(target_images, target_labels, shots, seed)
            epochs = max(20, 600 // len(train_y))
            torch.manual_seed(seed)
            scores["scratch"].append(accuracy(from_scratch(train_x, train_y, epochs), test_x, test_y))
            torch.manual_seed(seed)
            scores["random"].append(accuracy(linear_probe(random_backbone, train_x, train_y, 200), test_x, test_y))
            torch.manual_seed(seed)
            scores["probe"].append(accuracy(linear_probe(pretrained, train_x, train_y, 200), test_x, test_y))
            torch.manual_seed(seed)
            scores["finetune"].append(accuracy(fine_tune(pretrained, train_x, train_y, epochs), test_x, test_y))
        means = {key: sum(values) / len(values) for key, values in scores.items()}
        print(f"{shots:>13} {means['scratch']:>9.3f} {means['random']:>13.3f} "
              f"{means['probe']:>13.3f} {means['finetune']:>10.3f}")

    frozen_model = linear_probe(pretrained, target_images[:10], target_labels[:10], 1)
    trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in frozen_model.parameters())
    print(f"\nLinear probe trains {trainable:,} of {total:,} parameters")

    library_model = resnet18(weights=None)
    for parameter in library_model.parameters():
        parameter.requires_grad = False
    library_model.fc = nn.Linear(library_model.fc.in_features, 5)
    library_model = library_model.to(device)
    trainable = sum(p.numel() for p in library_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in library_model.parameters())
    print(f"torchvision resnet18 with frozen backbone and new 5-way fc: {trainable:,} of {total:,} trainable")
    with torch.no_grad():
        logits = library_model.eval()(target_images[:4].repeat(1, 3, 1, 1))
    print(f"  digits repeated to 3 channels {tuple(target_images[:4].repeat(1, 3, 1, 1).shape)} -> {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
