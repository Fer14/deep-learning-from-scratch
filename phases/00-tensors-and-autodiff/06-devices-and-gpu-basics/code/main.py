import time

import torch
import torch.utils.benchmark as benchmark


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_it(fn, device, repeats=10, warmup=3):
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / repeats


def matmul_gflops(size, seconds):
    return 2 * size ** 3 / seconds / 1e9


def train_tiny_regression(device, generator_seed):
    torch.manual_seed(generator_seed)
    X = torch.randn(2048, 32, device=device)
    true_w = torch.randn(32, 1, device=device)
    y = X @ true_w + 0.01 * torch.randn(2048, 1, device=device)
    w = torch.zeros(32, 1, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        loss = ((X @ w - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    return loss.item(), (w - true_w).abs().max().item()


def main():
    torch.manual_seed(0)
    device = pick_device()
    cpu = torch.device("cpu")

    print("=== Step 1: what hardware do you have? ===")
    print(f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, selected device: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"  GPU: {properties.name}, {properties.total_memory / 1e9:.1f} GB, {properties.multi_processor_count} SMs")
    print(f"  CPU threads used by torch: {torch.get_num_threads()}")

    print("\n=== Step 2: tensors live on a device ===")
    a = torch.randn(3, 3)
    b = a.to(device)
    print(f"  a.device = {a.device}, b = a.to(device) -> b.device = {b.device}")
    print(f"  a.to('cpu') is a (no copy when already there): {a.to('cpu') is a}")
    if device.type != "cpu":
        try:
            a + b
        except RuntimeError as error:
            print(f"  mixing devices raises: {str(error).splitlines()[0][:80]}")
        print(f"  move back with .cpu(): {b.cpu().device}, to NumPy needs CPU: {b.cpu().numpy().shape}")
    else:
        print("  only a CPU is available, so every tensor is already on the same device")

    print("\n=== Step 3: timing correctly (warm-up and synchronize) ===")
    size = 1024
    x_device = torch.randn(size, size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = time.perf_counter()
        x_device @ x_device
        naive = time.perf_counter() - start
        torch.cuda.synchronize()
        start = time.perf_counter()
        x_device @ x_device
        torch.cuda.synchronize()
        synced = time.perf_counter() - start
        print(f"  first call, no synchronize: {naive * 1000:.3f} ms (includes startup, or measures only the launch)")
        print(f"  second call with synchronize: {synced * 1000:.3f} ms (the real kernel time)")
    seconds = time_it(lambda: x_device @ x_device, device)
    print(f"  time_it: {size}x{size} matmul on {device}: {seconds * 1000:.3f} ms = {matmul_gflops(size, seconds):.1f} GFLOP/s")

    print("\n=== Step 4: CPU versus accelerator across sizes ===")
    print(f"  {'size':>6} | {'cpu ms':>9} | {'cpu GFLOP/s':>11} | {str(device) + ' ms':>9} | {'speedup':>7}")
    for size in [64, 256, 1024, 2048]:
        repeats = 5 if size < 2048 else 2
        x_cpu = torch.randn(size, size)
        cpu_seconds = time_it(lambda: x_cpu @ x_cpu, cpu, repeats=repeats, warmup=1)
        if device.type == "cpu":
            print(f"  {size:>6} | {cpu_seconds * 1000:9.3f} | {matmul_gflops(size, cpu_seconds):11.1f} | {'-':>9} | {'-':>7}")
            continue
        x_dev = x_cpu.to(device)
        dev_seconds = time_it(lambda: x_dev @ x_dev, device, repeats=repeats, warmup=1)
        print(f"  {size:>6} | {cpu_seconds * 1000:9.3f} | {matmul_gflops(size, cpu_seconds):11.1f} | {dev_seconds * 1000:9.3f} | {cpu_seconds / dev_seconds:6.1f}x")

    print("\n=== Step 5: moving data costs time too ===")
    host = torch.randn(4096, 1024)
    megabytes = host.numel() * host.element_size() / 1e6
    transfer_seconds = time_it(lambda: host.to(device), device, repeats=5)
    print(f"  copy {megabytes:.1f} MB host -> {device}: {transfer_seconds * 1000:.3f} ms")
    if device.type == "cuda":
        pinned = host.pin_memory()
        pinned_seconds = time_it(lambda: pinned.to(device, non_blocking=True), device, repeats=5)
        print(f"  from pinned memory: {pinned_seconds * 1000:.3f} ms ({megabytes / 1e3 / pinned_seconds:.1f} GB/s)")
        compute_seconds = time_it(lambda: torch.relu(host.to(device)).sum().item(), device, repeats=5)
        resident = host.to(device)
        resident_seconds = time_it(lambda: torch.relu(resident).sum().item(), device, repeats=5)
        print(f"  relu+sum including the copy each time: {compute_seconds * 1000:.3f} ms, data already on GPU: {resident_seconds * 1000:.3f} ms")

    print("\n=== Step 6: dtype changes speed and memory ===")
    for dtype in [torch.float64, torch.float32] + ([torch.float16, torch.bfloat16] if device.type == "cuda" else [torch.bfloat16]):
        x_typed = torch.randn(1024, 1024, device=device).to(dtype)
        seconds = time_it(lambda: x_typed @ x_typed, device, repeats=3, warmup=1)
        print(f"  {str(dtype):<15} {x_typed.element_size()} bytes/element | {seconds * 1000:8.3f} ms | {matmul_gflops(1024, seconds):9.1f} GFLOP/s")

    print("\n=== Step 7: GPU memory accounting ===")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        big = torch.empty(64, 1024, 1024, device=device)
        after = torch.cuda.memory_allocated()
        print(f"  allocating 64x1024x1024 float32: +{(after - before) / 1e6:.0f} MB allocated")
        del big
        print(f"  after del: allocated {torch.cuda.memory_allocated() / 1e6:.0f} MB, reserved by the caching allocator {torch.cuda.memory_reserved() / 1e6:.0f} MB")
        torch.cuda.empty_cache()
        print(f"  after empty_cache: reserved {torch.cuda.memory_reserved() / 1e6:.0f} MB")
    else:
        print("  no CUDA device: memory_allocated and memory_reserved only apply to GPUs")

    print("\n=== Step 8: device-agnostic training code ===")
    if device.type != "cpu":
        loss, error = train_tiny_regression(device, generator_seed=0)
        print(f"  on {device}: final loss {loss:.6f}, max weight error {error:.4f}")
    loss_cpu, error_cpu = train_tiny_regression(cpu, generator_seed=0)
    print(f"  on cpu : final loss {loss_cpu:.6f}, max weight error {error_cpu:.4f}")
    print("  the same code runs on any device; the same seed gives different random numbers on each device, so results match in quality, not bit for bit")

    print("\n=== Use It: torch.utils.benchmark ===")
    x_bench = torch.randn(1024, 1024, device=device)
    timer = benchmark.Timer(stmt="x @ x", globals={"x": x_bench})
    measurement = timer.blocked_autorange(min_run_time=0.5)
    print(f"  benchmark.Timer median: {measurement.median * 1000:.3f} ms over {len(measurement.times)} blocks (handles warm-up and CUDA sync for you)")


if __name__ == "__main__":
    main()
