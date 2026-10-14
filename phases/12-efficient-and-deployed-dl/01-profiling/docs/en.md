# Profiling

> Measure before you optimize: count the work, time it correctly, and find the one layer that actually costs you.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 00 (Tensors and Autodiff: Devices and GPU Basics), Phase 03 (CNNs: Convolution from Scratch)
**Time:** ~60 minutes

## Learning Objectives

- Count parameters, FLOPs and activation memory of convolutional and linear layers by hand
- Write a correct timer: warm-up, repeats, median, and device synchronization
- Explain why throughput grows with batch size and where it saturates
- Find the most expensive layer with forward hooks
- Use `torch.utils.benchmark.Timer` and `torch.profiler` and check them against your own numbers

## The Problem

Your model is too slow, or too big, for where it has to run. Before you quantize, prune, distill or rewrite anything (the rest of this phase), you need to know where the time and memory go. Intuition is a poor guide. The layer with the most parameters is often not the slowest one; the first call is much slower than the rest; and on a GPU, a naive `time.time()` around a forward pass can report a number that has nothing to do with the work done, because the GPU has not finished yet.

Profiling is the discipline of measuring this properly. It is also how you set expectations: if you know a model needs 2.7 MFLOPs per image and your hardware does 100 GFLOP/s, you know roughly what throughput to expect and when you are far from it.

## The Concept

### Counting the work

A multiply-add counts as 2 FLOPs. For the two layers that dominate most networks:

```
Linear(in, out), batch B:          FLOPs = 2 * in * out * B
Conv2d(C_in, C_out, k x k),
output H x W, batch B:             FLOPs = 2 * C_in * k * k * C_out * H * W * B
```

Parameters and FLOPs are different things. A linear layer uses each weight once per example. A convolution reuses each weight at every output position, so a small conv layer can cost far more compute than a big linear layer:

| Layer in this lesson | Params | FLOPs per image |
|-------|--------|-------|
| `conv2`: 32 to 64 channels, 3x3, 8x8 output | 18,496 | 2.36 M |
| `fc1`: 1024 to 128 | 131,200 | 0.26 M |

`fc1` has 7 times the parameters and one ninth of the compute.

Activation memory is the size of each layer's output. During training all of it is kept for the backward pass, which is why batch size is limited by activations, not weights.

### Timing correctly

```
for warm-up runs:   call fn                (caches, kernel selection, lazy init)
synchronize
for each repeat:
    start = clock
    call fn
    synchronize                            (wait for the GPU to finish)
    record clock - start
report the median                          (robust to background noise)
```

CUDA calls are asynchronous: PyTorch puts kernels in a queue and returns immediately. Without `torch.cuda.synchronize()` you measure how long it took to enqueue the work.

### Throughput and batch size

```
time per batch
  ^
  |                                  /
  |                               /     compute-bound: time grows with batch
  |____________________________/
  |   overhead-bound: time ~ constant
  +------------------------------------> batch size
```

At batch 1 the time is dominated by fixed costs: Python, kernel launches, memory traffic for weights. Larger batches amortize those costs, so images per second rise until the hardware is busy. Past that point, larger batches only add latency.

## Build It

### Step 1: A model worth profiling

`SmallCNN` is two 3x3 convolutions, a max pool and two linear layers, trained for a few epochs on scikit-learn's 8x8 digits (about 97% test accuracy). The script uses the GPU if there is one.

### Step 2: Count FLOPs with forward hooks

```python
def hook(module, inputs, output):
    nonlocal flops
    if isinstance(module, nn.Conv2d):
        per_output = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        layer_flops = 2 * per_output * output.numel()
    elif isinstance(module, nn.Linear):
        layer_flops = 2 * module.in_features * output.numel()
```

A forward hook sees each module's actual output, so the count uses real shapes instead of a formula you might get wrong. The table shows 2.66 MFLOPs per image, 89% of it in `conv2`, and 151k parameters, 87% of them in `fc1`.

### Step 3: Timing pitfalls

The script times one forward pass of 512 images three ways: the very first call, a second call without synchronization, and the median of 30 synchronized calls after warm-up. On a GPU the unsynchronized number is about half the real one, and the first call is slower again because cuDNN picks kernels. On a CPU there is no asynchrony, so the three numbers are close.

```python
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
```

### Step 4: Batch size sweep

On a GPU, batches 1, 8 and 64 all take about 0.17 ms: the time is pure overhead, and throughput rises almost linearly. At 512 the GPU finally has enough work and reaches a few TFLOP/s. On a CPU the peak comes earlier (around batch 64) and larger batches can even get slower because the activations no longer fit in cache.

### Step 5: Per-layer timing

`time_layers` registers a pre-hook and a post-hook on each child module, synchronizes in both, and records the elapsed time. `conv2` takes 60-75% of the forward pass, matching the FLOP count. This is the layer to optimize first; shrinking `fc1` would save memory but almost no time.

## Use It

```python
timer = benchmark.Timer(stmt="model(x)", globals={"model": model, "x": big})
result = timer.blocked_autorange(min_run_time=0.5)

with profile(activities=activities, with_flops=True) as prof:
    for _ in range(5):
        with record_function("forward_pass"):
            model(big)
print(prof.key_averages().table(sort_by=sort_key, row_limit=8))
```

`torch.utils.benchmark.Timer` does warm-up, synchronization and repeat selection for you; its median is within a few tens of percent of the hand-written timer (the gap is measurement noise on a shared machine). `torch.profiler` records every operator, down to the cuDNN kernel names on GPU, and `with_flops=True` estimates FLOPs for convolutions and matrix multiplications: its total matches your hand count exactly. `record_function` adds your own labels to the trace, and `prof.export_chrome_trace("trace.json")` writes a timeline you can open in Perfetto or `chrome://tracing`.

## Exercises

1. **Easy.** Double the width of `SmallCNN` (`width=64`). Predict the new FLOP count of `conv2` from the formula before running. How does the measured time scale?
2. **Medium.** Profile a training step (forward, backward, optimizer step) instead of inference. What fraction of the time is the backward pass? Is it close to the common rule of thumb of 2x the forward pass?
3. **Hard.** Add `profile_memory=True` to the profiler, profile a training step at batch sizes 64 and 512, and compare the peak memory with your hand-counted activation bytes. Which tensors account for the difference?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| FLOPs | "How big the model is" | Floating-point operations per forward pass; a multiply-add counts as 2 |
| FLOP/s | "Speed" | Operations per second achieved; compare with the hardware's peak to see how efficient you are |
| Latency | "How fast it responds" | Time for one request or batch from input to output |
| Throughput | "How much it handles" | Items processed per second; usually improves with batch size at the cost of latency |
| Warm-up | "Throw away the first run" | Initial calls that pay for caches, lazy initialization and kernel selection |
| Synchronize | "Wait for the GPU" | `torch.cuda.synchronize()`, which blocks until all queued GPU work is finished |
| Forward hook | "Spy on a layer" | A callback registered on a module that runs with its inputs and outputs on every forward pass |
| Profiler | "Where the time goes" | A tool that records the time, shapes and memory of every operator during a run |

## Further Reading

- [PyTorch docs: Profiling your PyTorch module](https://pytorch.org/tutorials/beginner/profiler.html) - `torch.profiler` usage, tables and traces
- [PyTorch docs: torch.utils.benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html) - why naive timing goes wrong and how `Timer` fixes it
- [Horace He, Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) - compute-bound, memory-bound and overhead-bound regimes explained
- [Williams, Waterman and Patterson, Roofline: An Insightful Visual Performance Model (2009)](https://doi.org/10.1145/1498765.1498785) - the model behind "how close am I to peak"
