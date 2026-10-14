import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.datasets import load_digits
from torchvision.transforms import v2


class ArrayDataset:
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def collate(samples):
    images, labels = zip(*samples)
    return np.stack(images), np.array(labels)


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, drop_last=False, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return -(-len(self.dataset) // self.batch_size)

    def __iter__(self):
        order = self.rng.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        for batch_index in range(len(self)):
            indices = order[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            yield collate([self.dataset[i] for i in indices])


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomApply:
    def __init__(self, transform, p, rng):
        self.transform = transform
        self.p = p
        self.rng = rng

    def __call__(self, image):
        return self.transform(image) if self.rng.random() < self.p else image


class RandomShift:
    def __init__(self, max_shift, rng):
        self.max_shift = max_shift
        self.rng = rng

    def __call__(self, image):
        padded = np.pad(image, self.max_shift)
        dy, dx = self.rng.integers(0, 2 * self.max_shift + 1, size=2)
        return padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]


class RandomRotation:
    def __init__(self, max_degrees, rng):
        self.max_degrees = max_degrees
        self.rng = rng

    def __call__(self, image):
        angle = self.rng.uniform(-self.max_degrees, self.max_degrees)
        return rotate(image, angle, reshape=False, order=1, mode="constant")


class AddNoise:
    def __init__(self, std, rng):
        self.std = std
        self.rng = rng

    def __call__(self, image):
        return image + self.rng.normal(0.0, self.std, size=image.shape)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


def init_mlp(sizes, rng):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
        params[f"b{i}"] = np.zeros(fan_out)
    return params


def forward(params, X):
    n_layers = len(params) // 2
    activations = [X]
    hidden = X
    for i in range(n_layers):
        z = hidden @ params[f"W{i}"] + params[f"b{i}"]
        hidden = np.maximum(z, 0.0) if i < n_layers - 1 else z
        activations.append(hidden)
    return activations


def loss_and_grads(params, X, y):
    n_layers = len(params) // 2
    activations = forward(params, X)
    logits = activations[-1] - activations[-1].max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    grads = {}
    for i in reversed(range(n_layers)):
        grads[f"W{i}"] = activations[i].T @ grad
        grads[f"b{i}"] = grad.sum(axis=0)
        if i > 0:
            grad = (grad @ params[f"W{i}"].T) * (activations[i] > 0)
    return loss, grads


def train_numpy(loader, epochs, seed, lr=0.05, beta=0.9):
    params = init_mlp([64, 128, 10], np.random.default_rng(seed))
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    for _ in range(epochs):
        for images, labels in loader:
            _, grads = loss_and_grads(params, images.reshape(len(images), -1), labels)
            for name in params:
                velocity[name] = beta * velocity[name] + grads[name]
                params[name] -= lr * velocity[name]
    return params


def accuracy_numpy(params, loader):
    correct, total = 0, 0
    for images, labels in loader:
        correct += np.sum(forward(params, images.reshape(len(images), -1))[-1].argmax(axis=1) == labels)
        total += len(labels)
    return correct / total


def load_split(seed, n_train):
    digits = load_digits()
    images = digits.images / 16.0
    order = np.random.default_rng(seed).permutation(len(images))
    images, labels = images[order], digits.target[order]
    return images[:n_train], labels[:n_train], images[n_train:], labels[n_train:]


def plot_augmentations(image, transform, path):
    fig, axes = plt.subplots(1, 8, figsize=(10, 1.6))
    axes[0].imshow(image, cmap="gray_r")
    axes[0].set_title("original", fontsize=8)
    for ax in axes[1:]:
        ax.imshow(transform(image), cmap="gray_r")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


class TorchDigits(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def train_torch(train_set, test_sets, epochs, seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=generator)
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    accuracies = []
    with torch.no_grad():
        for test_set in test_sets:
            correct = 0
            for images, labels in torch.utils.data.DataLoader(test_set, batch_size=256):
                correct += (model(images).argmax(dim=1) == labels).sum().item()
            accuracies.append(correct / len(test_set))
    return accuracies


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    train_images, train_labels, test_images, test_labels = load_split(seed=0, n_train=300)
    print(f"digits: {len(train_labels)} training images, {len(test_labels)} test images, each {train_images.shape[1:]}")

    print("\n=== Step 1: a Dataset and a DataLoader ===")
    dataset = ArrayDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, seed=0)
    images, labels = next(iter(loader))
    print(f"  len(dataset) = {len(dataset)}, len(loader) = {len(loader)} batches")
    print(f"  first batch: images {images.shape}, labels {labels[:10]}")
    print(f"  with drop_last=True: {len(DataLoader(dataset, 32, True, drop_last=True))} batches (last partial batch of {len(dataset) % 32} dropped)")
    seen = np.concatenate([batch_labels for _, batch_labels in DataLoader(dataset, 32, True, seed=1)])
    print(f"  one epoch visits every example exactly once: {len(seen) == len(dataset)} and class counts match: {np.array_equal(np.bincount(seen), np.bincount(train_labels))}")

    print("\n=== Step 2: normalize with training statistics only ===")
    mean, std = train_images.mean(), train_images.std()
    print(f"  pixel mean {mean:.4f}, std {std:.4f} computed on the 300 training images; the test set reuses them")

    print("\n=== Step 3: augmentations ===")
    aug_rng = np.random.default_rng(42)
    augment = Compose([RandomApply(RandomShift(1, aug_rng), 0.5, aug_rng), AddNoise(0.1, aug_rng), Normalize(mean, std)])
    too_strong = Compose([RandomRotation(15, aug_rng), Normalize(mean, std)])
    plain = Normalize(mean, std)
    sample = train_images[0]
    variants = np.stack([augment(sample) for _ in range(5)])
    print(f"  five augmented copies of one image differ from the plain one by mean |diff| {np.abs(variants - plain(sample)).mean(axis=(1, 2)).round(3)}")
    out_path = Path(__file__).parent / "augmentations.png"
    plot_augmentations(sample, Compose([RandomApply(RandomShift(1, aug_rng), 0.5, aug_rng), AddNoise(0.1, aug_rng)]), out_path)
    print(f"  saved examples to {out_path.name}")

    print("\n=== Step 4: train with and without augmentation (NumPy MLP, 150 epochs) ===")
    shifted_test = np.stack([np.roll(image, 1, axis=1) for image in test_images])
    test_loader = DataLoader(ArrayDataset(test_images, test_labels, plain), batch_size=256, shuffle=False)
    shifted_loader = DataLoader(ArrayDataset(shifted_test, test_labels, plain), batch_size=256, shuffle=False)
    clean_train_loader = DataLoader(ArrayDataset(train_images, train_labels, plain), batch_size=256, shuffle=False)
    for name, transform in (("no augmentation", plain), ("shift + noise", augment), ("rotation 15 deg", too_strong)):
        train_loader = DataLoader(ArrayDataset(train_images, train_labels, transform), batch_size=32, shuffle=True, seed=0)
        start = time.perf_counter()
        params = train_numpy(train_loader, epochs=150, seed=1)
        elapsed = time.perf_counter() - start
        print(
            f"  {name:16s} | train acc {accuracy_numpy(params, clean_train_loader):.3f}"
            f" | test acc {accuracy_numpy(params, test_loader):.3f}"
            f" | test acc, images shifted 1 px {accuracy_numpy(params, shifted_loader):.3f} | {elapsed:.1f} s"
        )

    print("\n=== Use It: torch.utils.data and torchvision.transforms.v2 ===")
    torch_augment = v2.Compose([
        v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.125, 0.125))], p=0.5),
        v2.Lambda(lambda image: image + 0.1 * torch.randn_like(image)),
        v2.Normalize([mean], [std]),
    ])
    torch_plain = v2.Normalize([mean], [std])
    torch_test = TorchDigits(test_images, test_labels, torch_plain)
    torch_shifted = TorchDigits(shifted_test, test_labels, torch_plain)
    loader = torch.utils.data.DataLoader(TorchDigits(train_images, train_labels, torch_augment), batch_size=32, shuffle=True)
    images, labels = next(iter(loader))
    print(f"  DataLoader batch: images {tuple(images.shape)} {images.dtype}, labels {tuple(labels.shape)}")
    for name, transform in (("no augmentation", torch_plain), ("shift + noise", torch_augment)):
        test_acc, shifted_acc = train_torch(TorchDigits(train_images, train_labels, transform), [torch_test, torch_shifted], epochs=150, seed=1)
        print(f"  torch {name:16s} | test acc {test_acc:.3f} | shifted test acc {shifted_acc:.3f}")


if __name__ == "__main__":
    main()
