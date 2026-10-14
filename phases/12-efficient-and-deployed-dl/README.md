# Phase 12: Efficient and Deployed DL

> A model is only as useful as the place it can run: measure it, shrink it, freeze it, and serve it.

This phase takes trained networks out of the notebook and into production. You start by measuring correctly: FLOPs, memory, synchronized timing and per-layer profiles. Then you make models smaller and cheaper in three complementary ways: quantization (fewer bits per weight, with real int8 matrix multiplication), knowledge distillation (a 125x smaller student that learns even a digit it never saw), and pruning (unstructured masks versus removing whole neurons). Finally you deploy: export a model to a framework-free artifact and to `torch.export`, serve it over HTTP with validation and latency percentiles, and simulate a server with dynamic batching to see how it survives traffic that would sink a one-request-at-a-time design. Every model is small and trained inside its own script on scikit-learn's digits.

**Prerequisites:** Phase 01 (Neural Networks), Phase 02 (Training Deep Nets), Phase 03 (CNNs)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Profiling](01-profiling/) | Hand-counted FLOPs and activation memory, a synchronized timer, a batch-size sweep, per-layer hooks, and torch.profiler |
| 02 | [Quantization](02-quantization/) | Affine and symmetric quantizers, per-channel scales, a bit-width sweep, and an int8 linear layer with int32 accumulation |
| 03 | [Knowledge Distillation](03-knowledge-distillation/) | Temperature softmax, the T^2-scaled distillation loss, distillation on unlabeled data, and Hinton's missing-digit experiment |
| 04 | [Pruning](04-pruning/) | Layer-wise and global magnitude masks, iterative pruning with a cubic schedule, and structured neuron removal that actually runs faster |
| 05 | [Export and Serving](05-export-and-serving/) | A JSON + npz export with a NumPy runtime, an HTTP prediction server, and torch.export with a dynamic batch dimension |
| 06 | [Batching and Inference Optimization](06-batching-and-inference-optimization/) | inference_mode, BatchNorm folding, a measured service-time table, and a dynamic-batching server simulation |

Run the first lesson:

```bash
uv run python phases/12-efficient-and-deployed-dl/01-profiling/code/main.py
```
