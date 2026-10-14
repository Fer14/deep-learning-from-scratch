import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils import benchmark


def load_data():
    digits = load_digits()
    images = torch.tensor(digits.images, dtype=torch.float32)[:, None] / 16.0
    labels = torch.tensor(digits.target)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))
    return images[order], labels[order]


class SmallCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, 2 * width, 3, padding=1)
        self.fc1 = nn.Linear(2 * width * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def train(model, images, labels, device, epochs=8):
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    images, labels = images.to(device), labels.to(device)
    for _ in range(epochs):
        for i in range(0, 1500, 64):
            loss = F.cross_entropy(model(images[i:i + 64]), labels[i:i + 64])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        accuracy = (model(images[1500:]).argmax(1) == labels[1500:]).float().mean().item()
    return accuracy


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure(fn, device, warmup=5, repeats=30):
    for _ in range(warmup):
        fn()
    synchronize(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def count_flops(model, example):
    flops, rows = 0, []

    def hook(module, inputs, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            per_output = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
            layer_flops = 2 * per_output * output.numel()
        elif isinstance(module, nn.Linear):
            layer_flops = 2 * module.in_features * output.numel()
        else:
            return
        flops += layer_flops
        rows.append((module.__class__.__name__, tuple(output.shape), sum(p.numel() for p in module.parameters()),
                     layer_flops, output.numel() * output.element_size()))

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(example)
    for handle in handles:
        handle.remove()
    return flops, rows


def time_layers(model, example, device):
    timings = {}

    def make_hooks(name):
        def pre(module, inputs):
            synchronize(device)
            timings.setdefault(name, []).append(-time.perf_counter())

        def post(module, inputs, output):
            synchronize(device)
            timings[name][-1] += time.perf_counter()
        return pre, post

    handles = []
    for name, module in model.named_children():
        pre, post = make_hooks(name)
        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    with torch.no_grad():
        for _ in range(20):
            model(example)
    for handle in handles:
        handle.remove()
    return {name: statistics.median(values[5:]) for name, values in timings.items()}


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = load_data()
    model = SmallCNN()
    accuracy = train(model, images, labels, device)
    print(f"Device: {device}. SmallCNN trained on 8x8 digits, test accuracy {accuracy:.3f}\n")

    print("1) Count parameters, FLOPs and activation memory by hand (batch of 1)")
    example = images[:1].to(device)
    total_flops, rows = count_flops(model, example)
    print(f"  {'layer':8s} {'output shape':>18s} {'params':>8s} {'FLOPs':>10s} {'act bytes':>10s}")
    for name, shape, params, flops, act_bytes in rows:
        print(f"  {name:8s} {str(shape):>18s} {params:8d} {flops:10d} {act_bytes:10d}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total: {n_params} parameters ({n_params * 4 / 1024:.0f} KiB in fp32), {total_flops / 1e6:.2f} MFLOPs per image")

    print("\n2) Timing pitfalls")
    big = images[:512].to(device)
    with torch.no_grad():
        start = time.perf_counter()
        model(big)
        first_call = time.perf_counter() - start
        start = time.perf_counter()
        model(big)
        no_sync = time.perf_counter() - start
        synchronize(device)
        median_time = measure(lambda: model(big), device)
    print(f"  first call (includes warm-up / kernel selection): {first_call * 1e3:8.3f} ms")
    print(f"  second call, no synchronize:                       {no_sync * 1e3:8.3f} ms")
    print(f"  median of 30 after warm-up, synchronized:          {median_time * 1e3:8.3f} ms")
    if device == "cuda":
        print("  on GPU, kernels run asynchronously: without synchronize you time the launch, not the work")

    print("\n3) Throughput versus batch size")
    with torch.no_grad():
        for batch_size in [1, 8, 64, 512]:
            x = images[:batch_size].to(device)
            seconds = measure(lambda: model(x), device)
            gflops = total_flops * batch_size / seconds / 1e9
            print(f"  batch {batch_size:4d}: {seconds * 1e3:7.3f} ms/batch | {batch_size / seconds:10.0f} images/s | {gflops:7.2f} GFLOP/s")

    print("\n4) Where does the time go? Per-layer timing with hooks (batch 512)")
    layer_times = time_layers(model, big, device)
    total = sum(layer_times.values())
    for name, seconds in layer_times.items():
        print(f"  {name:6s} {seconds * 1e3:7.3f} ms  {100 * seconds / total:5.1f}%")

    print("\n5) Use It: torch.utils.benchmark and torch.profiler")
    with torch.no_grad():
        timer = benchmark.Timer(stmt="model(x)", globals={"model": model, "x": big})
        result = timer.blocked_autorange(min_run_time=0.5)
        print(f"  benchmark.Timer median: {result.median * 1e3:.3f} ms (hand-written: {median_time * 1e3:.3f} ms)")
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
        with profile(activities=activities, with_flops=True) as prof:
            for _ in range(5):
                with record_function("forward_pass"):
                    model(big)
            synchronize(device)
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=8, max_name_column_width=28))
    profiler_flops = sum(event.flops for event in prof.key_averages() if event.flops) / 5
    print(f"  profiler FLOPs per forward pass: {profiler_flops / 1e6:.1f} M | hand count x 512: {total_flops * 512 / 1e6:.1f} M")


if __name__ == "__main__":
    main()
