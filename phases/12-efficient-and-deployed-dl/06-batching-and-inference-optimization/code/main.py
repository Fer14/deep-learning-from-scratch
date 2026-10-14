import copy
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torch.nn.utils.fusion import fuse_conv_bn_eval


def load_data():
    digits = load_digits()
    images = torch.tensor(digits.images, dtype=torch.float32)[:, None] / 16.0
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    return images[order], labels[order]


class ConvBNNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width)
        self.conv2, self.bn2 = nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width)
        self.head = nn.Linear(2 * width * 16, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        return self.head(x.flatten(1))


class FoldedNet(nn.Module):
    def __init__(self, conv1, conv2, head):
        super().__init__()
        self.conv1, self.conv2, self.head = conv1, conv2, head

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.head(x.flatten(1))


def fold_bn(conv, bn):
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    folded = copy.deepcopy(conv)
    with torch.no_grad():
        folded.weight.copy_(conv.weight * factor[:, None, None, None])
        folded.bias.copy_((conv.bias - bn.running_mean) * factor + bn.bias)
    return folded


def train(model, images, labels, device, epochs=6):
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    x, y = images.to(device), labels.to(device)
    for _ in range(epochs):
        for i in range(0, 1400, 64):
            loss = F.cross_entropy(model(x[i:i + 64]), y[i:i + 64])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure(fn, device, repeats=30):
    for _ in range(5):
        fn()
    synchronize(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def simulate_server(arrivals, service_time, max_batch, max_wait):
    queue_start, server_free, latencies, batch_sizes = 0, 0.0, [], []
    n = len(arrivals)
    while queue_start < n:
        oldest = arrivals[queue_start]
        full_at = arrivals[queue_start + max_batch - 1] if queue_start + max_batch - 1 < n else np.inf
        start = max(server_free, min(oldest + max_wait, full_at))
        end_index = queue_start
        while end_index < n and end_index - queue_start < max_batch and arrivals[end_index] <= start:
            end_index += 1
        size = end_index - queue_start
        finish = start + service_time(size)
        latencies.extend(finish - arrivals[queue_start:end_index])
        batch_sizes.append(size)
        server_free, queue_start = finish, end_index
    latencies = np.array(latencies)
    return latencies, np.array(batch_sizes), n / (server_free - arrivals[0])


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cpu":
        torch.set_num_threads(2)
    images, labels = load_data()
    model = train(ConvBNNet(), images, labels, device)
    x_test, y_test = images[1400:].to(device), labels[1400:].to(device)
    with torch.no_grad():
        reference = model(x_test)
    print(f"Device: {device}. ConvBNNet test accuracy {(reference.argmax(1) == y_test).float().mean():.4f}\n")

    print("1) Turn off autograd bookkeeping at inference time")
    batch = x_test[:256]
    grad_time = measure(lambda: model(batch), device)
    with torch.no_grad():
        no_grad_time = measure(lambda: model(batch), device)
    with torch.inference_mode():
        inference_time = measure(lambda: model(batch), device)
    print(f"  grad enabled {grad_time * 1e3:.3f} ms | no_grad {no_grad_time * 1e3:.3f} ms | inference_mode {inference_time * 1e3:.3f} ms")

    print("\n2) Fold BatchNorm into the preceding convolution")
    folded = FoldedNet(fold_bn(model.conv1, model.bn1), fold_bn(model.conv2, model.bn2), model.head).eval()
    with torch.inference_mode():
        diff = (folded(x_test) - reference).abs().max().item()
        bn_time = measure(lambda: model(batch), device)
        folded_time = measure(lambda: folded(batch), device)
    print(f"  max |output difference| {diff:.2e}")
    print(f"  conv+bn {bn_time * 1e3:.3f} ms -> folded conv {folded_time * 1e3:.3f} ms per batch of 256 "
          f"({len(list(model.modules())) - 1} modules -> {len(list(folded.modules())) - 1})")

    print("\n3) Measure the service time of one batch for each batch size")
    sizes = [1, 2, 4, 8, 16, 32, 64]
    table = {}
    with torch.inference_mode():
        for size in sizes:
            x = x_test[:size]
            table[size] = measure(lambda: folded(x), device)
            print(f"  batch {size:3d}: {table[size] * 1e3:7.3f} ms  ->  {size / table[size]:9.0f} images/s")

    def service_time(size):
        bucket = next(s for s in sizes if s >= size)
        return table[bucket]

    print("\n4) Dynamic batching: simulate a server with Poisson request arrivals")
    rng = np.random.default_rng(0)
    capacity_single = 1 / table[1]
    for load in [0.5, 2.0]:
        rate = load * capacity_single
        arrivals = np.cumsum(rng.exponential(1 / rate, size=4000))
        print(f"  arrival rate {rate:8.0f} req/s ({load:.1f}x what batch-1 serving can handle)")
        print(f"    {'policy':30s} {'p50 ms':>8s} {'p99 ms':>8s} {'mean batch':>11s} {'served/s':>9s}")
        for name, max_batch, max_wait in [("no batching", 1, 0.0), ("max 16, wait <= 1 ms", 16, 1e-3),
                                          ("max 64, wait <= 2 ms", 64, 2e-3)]:
            latencies, batch_sizes, throughput = simulate_server(arrivals, service_time, max_batch, max_wait)
            print(f"    {name:30s} {np.percentile(latencies, 50) * 1e3:8.3f} {np.percentile(latencies, 99) * 1e3:8.2f} "
                  f"{batch_sizes.mean():11.1f} {throughput:9.0f}")

    print("\n5) Use It: torch.nn.utils.fusion and reduced precision")
    library_conv = fuse_conv_bn_eval(model.conv1, model.bn1)
    print(f"  fuse_conv_bn_eval matches our fold: max |weight diff| "
          f"{(library_conv.weight - fold_bn(model.conv1, model.bn1).weight).abs().max():.2e}")
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
        low = folded(x_test).float()
        low_time = measure(lambda: folded(x_test), device)
    with torch.inference_mode():
        full_time = measure(lambda: folded(x_test), device)
    agreement = (low.argmax(1) == reference.argmax(1)).float().mean()
    print(f"  autocast {str(dtype).split('.')[-1]}: same prediction on {agreement:.1%} of test images, "
          f"{full_time * 1e3:.3f} ms -> {low_time * 1e3:.3f} ms for {len(x_test)} images")


if __name__ == "__main__":
    main()
