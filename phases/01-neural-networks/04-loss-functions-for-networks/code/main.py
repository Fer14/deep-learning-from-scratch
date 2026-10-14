import numpy as np
import torch
import torch.nn.functional as F


def mse(prediction, target):
    return np.mean((prediction - target) ** 2)


def mse_grad(prediction, target):
    return 2 * (prediction - target) / prediction.size


def mae(prediction, target):
    return np.mean(np.abs(prediction - target))


def huber(prediction, target, delta=1.0):
    error = prediction - target
    small = np.abs(error) <= delta
    return np.mean(np.where(small, 0.5 * error ** 2, delta * (np.abs(error) - 0.5 * delta)))


def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def bce_with_logits(logits, target):
    return -np.mean(target * log_sigmoid(logits) + (1 - target) * log_sigmoid(-logits))


def bce_with_logits_grad(logits, target):
    return (1.0 / (1.0 + np.exp(-logits)) - target) / logits.size


def log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def cross_entropy(logits, labels):
    return -np.mean(log_softmax(logits)[np.arange(len(labels)), labels])


def cross_entropy_grad(logits, labels):
    probs = np.exp(log_softmax(logits))
    probs[np.arange(len(labels)), labels] -= 1
    return probs / len(labels)


def numerical_grad(loss_fn, x, eps=1e-6):
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + eps
        plus = loss_fn(x)
        x[index] = original - eps
        minus = loss_fn(x)
        x[index] = original
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def main():
    rng = np.random.default_rng(0)

    print("=== Step 1: regression losses and outliers ===")
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prediction = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    with_outlier = prediction.copy()
    with_outlier[-1] = 15.0
    for name, fn in [("MSE", mse), ("MAE", mae), ("Huber", huber)]:
        print(f"  {name:5s} clean={fn(prediction, target):.4f}  with outlier={fn(with_outlier, target):.4f}")

    print("\n=== Step 2: binary cross-entropy from logits ===")
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    for logits in [np.array([3.0, -3.0, 2.0, -2.0]), np.array([0.0, 0.0, 0.0, 0.0]), np.array([-3.0, 3.0, -2.0, 2.0])]:
        print(f"  logits={logits} -> BCE={bce_with_logits(logits, labels):.4f}")
    huge = np.array([200.0, -200.0])
    naive_p = 1 / (1 + np.exp(-huge))
    with np.errstate(divide="ignore", invalid="ignore"):
        naive = -np.mean(np.array([0.0, 1.0]) * np.log(naive_p) + (1 - np.array([0.0, 1.0])) * np.log(1 - naive_p))
    print(f"  confidently wrong logits {huge}: naive BCE={naive}, logit BCE={bce_with_logits(huge, np.array([0.0, 1.0])):.1f}")

    print("\n=== Step 3: softmax cross-entropy for 4 classes ===")
    logits = rng.normal(size=(5, 4))
    class_labels = np.array([0, 3, 1, 2, 3])
    print(f"  random logits loss = {cross_entropy(logits, class_labels):.4f} (log 4 = {np.log(4):.4f})")
    perfect = np.full((5, 4), -5.0)
    perfect[np.arange(5), class_labels] = 5.0
    print(f"  confident correct loss = {cross_entropy(perfect, class_labels):.4f}")

    print("\n=== Step 4: the gradient is (prediction - target) ===")
    analytic = cross_entropy_grad(logits, class_labels)
    numeric = numerical_grad(lambda x: cross_entropy(x, class_labels), logits.copy())
    print(f"  softmax CE max |analytic - numeric| = {np.max(np.abs(analytic - numeric)):.2e}")
    binary_logits = rng.normal(size=6)
    binary_labels = rng.integers(0, 2, size=6).astype(float)
    analytic = bce_with_logits_grad(binary_logits, binary_labels)
    numeric = numerical_grad(lambda x: bce_with_logits(x, binary_labels), binary_logits.copy())
    print(f"  BCE        max |analytic - numeric| = {np.max(np.abs(analytic - numeric)):.2e}")
    analytic = mse_grad(prediction, target)
    numeric = numerical_grad(lambda x: mse(x, target), prediction.copy())
    print(f"  MSE        max |analytic - numeric| = {np.max(np.abs(analytic - numeric)):.2e}")

    print("\n=== Use It: torch losses ===")
    t_logits = torch.tensor(logits)
    t_labels = torch.tensor(class_labels)
    print(f"  F.cross_entropy            = {F.cross_entropy(t_logits, t_labels).item():.6f} vs scratch {cross_entropy(logits, class_labels):.6f}")
    print(f"  F.binary_cross_entropy_with_logits = {F.binary_cross_entropy_with_logits(torch.tensor(binary_logits), torch.tensor(binary_labels)).item():.6f} vs scratch {bce_with_logits(binary_logits, binary_labels):.6f}")
    print(f"  F.mse_loss                 = {F.mse_loss(torch.tensor(with_outlier), torch.tensor(target)).item():.6f} vs scratch {mse(with_outlier, target):.6f}")
    print(f"  F.huber_loss               = {F.huber_loss(torch.tensor(with_outlier), torch.tensor(target)).item():.6f} vs scratch {huber(with_outlier, target):.6f}")
    t_logits.requires_grad_(True)
    F.cross_entropy(t_logits, t_labels).backward()
    print(f"  autograd vs analytic CE gradient: {np.max(np.abs(t_logits.grad.numpy() - cross_entropy_grad(logits, class_labels))):.2e}")


if __name__ == "__main__":
    main()
