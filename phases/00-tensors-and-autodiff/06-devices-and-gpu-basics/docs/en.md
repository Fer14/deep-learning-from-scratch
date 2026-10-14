# Devices and GPU Basics

> A GPU is a factory far away from your data. It is enormously fast once the work arrives, so the art is to send big jobs and keep the data there.

**Type:** Learn
**Languages:** Python
**Prerequisites:** PyTorch Tensors and Autograd
**Time:** ~45 minutes

## Learning Objectives

- Select a device in a portable way and move tensors and models between CPU and GPU
- Explain why GPU operations are asynchronous and time them correctly with warm-up and synchronization
- Measure matmul throughput in GFLOP/s and see where the GPU wins and where it does not
- Account for host-to-device transfer cost and use pinned memory
- Compare dtypes (float64, float32, float16, bfloat16) by speed and memory
- Read GPU memory with `memory_allocated` and `memory_reserved`, and write device-agnostic training code

## The Problem

You move your model to the GPU, run it, and it is slower than on the CPU. Or your timing says a matmul takes 0.02 ms, which would be impossible for its size. Or training crashes with "Expected all tensors to be on the same device". Or it runs out of GPU memory even though you deleted your big tensors.

All four come from not knowing how a GPU actually works with PyTorch. The GPU is a separate processor with its own memory, connected to the CPU by a comparatively slow bus. PyTorch hands it work asynchronously and caches its memory. Once you see those three facts, every one of these puzzles has an obvious answer, and you will know when a GPU helps at all: the models in this course are tiny, and many of them run perfectly well on a laptop CPU.

## The Concept

### Two processors, two memories

```
   CPU + RAM                         GPU + VRAM
 +-------------+    PCIe bus     +------------------+
 | Python      |  ~10-25 GB/s    | thousands of     |
 | your data   | <-------------> | simple cores     |
 | few fast    |                 | ~300-1000 GB/s   |
 | cores       |                 | to its own VRAM  |
 +-------------+                 +------------------+
```

A tensor lives on exactly one device. An operation runs on the device where its inputs are, and all inputs must be on the same device. Moving data across the bus is far slower than the GPU's access to its own memory, so you want to move data once and keep it there.

### Asynchronous execution

When Python calls `x @ x` on a CUDA tensor, PyTorch puts a kernel in a queue and returns immediately. The CPU keeps running Python while the GPU works. This is great for throughput and terrible for naive timing:

```
time.perf_counter()        CPU
x @ x            -------> queue kernel, return at once
time.perf_counter()        CPU: measured only the launch
                           GPU: still computing
```

To time GPU code you must call `torch.cuda.synchronize()` before reading the clock, and do a few warm-up runs first, because the first call pays one-time costs (CUDA context creation, kernel selection, memory allocation). Anything that needs the value on the CPU, such as `.item()`, `.cpu()` or printing, also forces a synchronization.

### Throughput and arithmetic intensity

A matmul of two `n x n` matrices does `2 n^3` floating point operations on `3 n^2` numbers. Large matmuls have lots of work per byte, so they keep thousands of GPU cores busy. Small ones are dominated by fixed launch overhead (a few microseconds per kernel), so for tiny tensors the GPU's advantage shrinks or disappears. Deep learning is fast on GPUs because it is mostly large matmuls.

### Dtypes

| dtype | Bytes | Typical use |
|-------|-------|-------------|
| float64 | 8 | Scientific computing, gradient checks; very slow on consumer GPUs |
| float32 | 4 | Default for training |
| float16 | 2 | Mixed precision on GPUs with tensor cores; narrow range |
| bfloat16 | 2 | Mixed precision with float32's range; fast on recent GPUs and some CPUs |

Halving the bytes halves memory and bandwidth, and on GPUs with tensor cores the 16-bit formats are several times faster. Phase 02 (Training Deep Nets) shows how to train safely in mixed precision.

### The caching allocator

Allocating GPU memory is slow, so PyTorch keeps freed blocks in a cache and reuses them. `torch.cuda.memory_allocated()` reports what live tensors use; `torch.cuda.memory_reserved()` reports what PyTorch is holding from the driver, which is what tools like `nvidia-smi` show. `del` lowers the first; `torch.cuda.empty_cache()` returns cached blocks and lowers the second.

## Build It

### Step 1: Pick a device

```python
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

`mps` is Apple Silicon's GPU backend. Every script in this course falls back to the CPU, so it runs anywhere.

### Step 2: Move tensors

```python
a = torch.randn(3, 3)
b = a.to(device)
a + b
```

On a GPU machine the last line raises "Expected all tensors to be on the same device". `.to()` returns the same object if the tensor is already on the target device, and `.numpy()` only works on CPU tensors, so you write `b.cpu().numpy()`.

### Step 3: A correct timer

```python
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
```

The script first shows what goes wrong: the very first CUDA matmul, timed without care, includes tens of milliseconds of setup, while a synchronized second call takes a fraction of a millisecond.

### Step 4: CPU versus GPU across sizes

For each size from 64 to 2048, the script times a square matmul on the CPU and on the accelerator and prints the speedup. On an RTX 3060 a 1024x1024 float32 matmul runs at several TFLOP/s, and the 2048 case is still around a hundred times faster than a desktop CPU. Your numbers will differ with your hardware, and a busy machine can make small multithreaded CPU matmuls look much slower than they are. Look at the trend, not the exact values: GFLOP/s grows with size on both devices because bigger problems use the hardware better.

### Step 5: Transfers

```python
host = torch.randn(4096, 1024)
transfer_seconds = time_it(lambda: host.to(device), device, repeats=5)
pinned = host.pin_memory()
pinned_seconds = time_it(lambda: pinned.to(device, non_blocking=True), device, repeats=5)
```

Copying 16.8 MB takes milliseconds, which is longer than many kernels. **Pinned** (page-locked) host memory lets the GPU copy directly by DMA and is several times faster. The last line of this step is the important one: computing `relu(x).sum()` with a fresh copy every time is dozens of times slower than computing it on data that already lives on the GPU. This is why `DataLoader(pin_memory=True)` exists and why you move the model once, outside the training loop.

### Step 6: Dtypes

The script times the same 1024x1024 matmul in each dtype. On a consumer GPU float64 is dozens of times slower than float32, and float16 and bfloat16 are several times faster than float32. On a CPU without native bfloat16 support, bfloat16 matmuls can be much slower than float32: the speed of a dtype depends on the hardware, not only on its size.

### Step 7: Memory accounting

```python
before = torch.cuda.memory_allocated()
big = torch.empty(64, 1024, 1024, device=device)
after = torch.cuda.memory_allocated()
del big
torch.cuda.memory_reserved()
torch.cuda.empty_cache()
```

64 x 1024 x 1024 float32 numbers are 268 MB. After `del`, allocated memory drops but reserved memory stays high until `empty_cache()`.

### Step 8: Device-agnostic code

```python
def train_tiny_regression(device, generator_seed):
    torch.manual_seed(generator_seed)
    X = torch.randn(2048, 32, device=device)
    true_w = torch.randn(32, 1, device=device)
    y = X @ true_w + 0.01 * torch.randn(2048, 1, device=device)
    w = torch.zeros(32, 1, device=device, requires_grad=True)
```

Create tensors directly on `device` instead of creating them on the CPU and moving them. The same function trains on the GPU and on the CPU to the same quality. The random numbers differ, because the CPU and CUDA generators are separate streams even with the same seed.

## Use It

Writing your own timer teaches you the pitfalls; in practice use `torch.utils.benchmark`:

```python
timer = benchmark.Timer(stmt="x @ x", globals={"x": x_bench})
measurement = timer.blocked_autorange(min_run_time=0.5)
measurement.median
```

It warms up, synchronizes CUDA, chooses the number of repetitions and reports robust statistics. For finding where time goes in a full model you will use `torch.profiler` in Phase 12 (Efficient and Deployed DL).

## Exercises

1. **Easy.** Remove the `synchronize` calls from `time_it` and rerun Step 4 on a GPU. Explain the new, impossibly fast numbers.
2. **Medium.** Time an elementwise operation (`x * 2 + 1`) on tensors of increasing size and compute its effective bandwidth in GB/s (bytes read plus bytes written, divided by time). Compare it with the matmul GFLOP/s and explain why elementwise operations are called memory-bound.
3. **Hard.** Time a loop of 1000 tiny `(16, 16)` matmuls on the GPU, then the same work as a single batched `(1000, 16, 16)` matmul. Explain the difference in terms of kernel launch overhead, and relate it to why training uses minibatches.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Device | "CPU or GPU" | The processor and memory where a tensor lives and its operations run |
| Host / device | "CPU / GPU" | CUDA vocabulary: host is the CPU side, device is the GPU side |
| Asynchronous execution | "The GPU is lazy" | Kernels are queued and the CPU continues without waiting for them to finish |
| Synchronize | "Wait for the GPU" | Block the CPU until all queued GPU work is done; needed for correct timing |
| Warm-up | "Ignore the first run" | Running code a few times before timing to exclude one-time setup costs |
| Pinned memory | "Faster copies" | Page-locked host memory the GPU can copy from directly by DMA |
| GFLOP/s | "Speed" | Billions of floating point operations per second actually achieved |
| Caching allocator | "PyTorch keeps my memory" | PyTorch reuses freed GPU blocks instead of returning them, so reserved exceeds allocated |

## Further Reading

- [PyTorch: CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) - asynchronous execution, streams, the caching allocator and pinned memory
- [PyTorch: torch.utils.benchmark tutorial](https://pytorch.org/tutorials/recipes/recipes/benchmark.html) - why naive timing lies and how the benchmark module fixes it
- [Horace He, Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) - compute-bound versus memory-bound versus overhead-bound, explained clearly
- [Dive into Deep Learning: GPUs](https://d2l.ai/chapter_builders-guide/use-gpu.html) - devices, copying and placing models in PyTorch
