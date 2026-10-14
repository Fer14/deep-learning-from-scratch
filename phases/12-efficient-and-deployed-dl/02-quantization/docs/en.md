# Quantization

> A trained network does not need 32 bits per weight: map each number to a small integer and a shared scale, and the model gets four times smaller with almost no loss.

**Type:** Build
**Languages:** Python
**Prerequisites:** Profiling (this phase), Phase 01 (Neural Networks: PyTorch nn.Module), Phase 02 (Training Deep Nets: Mixed Precision and Checkpointing)
**Time:** ~70 minutes

## Learning Objectives

- Implement affine (asymmetric) and symmetric quantization and reason about their rounding error
- Explain why per-channel scales beat per-tensor scales, especially at low bit widths
- Measure how accuracy degrades as weights go from 8 to 2 bits
- Build an int8 linear layer with integer matrix multiplication and int32 accumulation
- Compare your implementation with PyTorch's dynamic quantization

## The Problem

A 7-billion-parameter language model in fp32 needs 28 GB just for its weights. In fp16 it needs 14 GB, more than most laptop GPUs have. Memory is also a speed problem: when you generate one token at a time, the bottleneck is reading every weight from memory, not the arithmetic. Fewer bytes per weight means faster inference.

Quantization stores numbers in low-precision integers (int8, int4) plus a few floating-point scale factors. Done well, an 8-bit model is four times smaller than fp32 and almost indistinguishable in accuracy; 4-bit weight quantization is how most large models run on consumer hardware today. Done badly, a single outlier weight destroys the precision of everything around it. The difference comes down to a few details you will implement here.

## The Concept

### Mapping floats to integers

Affine quantization maps a real range `[x_min, x_max]` onto the integers `[0, 2^b - 1]`:

```
scale      = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)
q          = clamp(round(x / scale) + zero_point, 0, 2^b - 1)
x_hat      = (q - zero_point) * scale
```

```
real line   x_min ------------------ 0 ------------------------- x_max
                |    |    |    |    |    |    |    |    |    |    |
integers        0    1    2   ...  zp  ...                      255
```

Symmetric quantization fixes the zero point at 0 and uses `[-(2^(b-1) - 1), 2^(b-1) - 1]`:

```
scale = max|x| / (2^(b-1) - 1)          q = clamp(round(x / scale), -127, 127)
```

Rounding error is at most `scale / 2`. Symmetric wastes a little range if the data is not centered, but makes integer arithmetic simpler because there are no zero-point terms. Weights are usually close to symmetric, so symmetric is the common choice for them.

### Granularity: one scale or many

The scale is set by the largest value it has to cover. With one scale for a whole matrix (per-tensor), a single row with large weights forces a coarse grid on every other row. With one scale per output row (per-channel), each row gets a grid fitted to its own range.

```
row 0  max|w| = 0.06   |.|.|.|.|.|      per-channel: fine grid
row 1  max|w| = 0.34   |   |   |   |    per-tensor: everyone uses this coarse grid
```

At lower bit widths you go further: group-wise quantization (one scale per block of 32-128 weights) is what GPTQ, AWQ and the GGUF formats use for 4-bit language models.

### Integer matrix multiplication

With symmetric quantization of both activations `X ~ s_x X_q` and weights `W ~ s_w W_q`:

```
X W^T  ~  (s_x X_q)(s_w W_q)^T  =  s_x * s_w * (X_q W_q^T)
```

The expensive part, `X_q W_q^T`, is a matrix product of int8 values. Products of int8 numbers fit in int16, and sums of many of them need int32, so hardware accumulates in int32 and rescales once at the end. With per-channel weights, `s_w` is a vector and the rescale is a broadcasted multiply.

### Kinds of quantization

| Method | What is quantized | Calibration | Typical use |
|--------|------------------|-------------|-------------|
| Weight-only | weights; activations stay float | none | LLMs on GPU (memory-bound) |
| Dynamic | weights ahead of time, activations per batch at runtime | none | CPU inference of linear and LSTM layers |
| Static | weights and activations with fixed scales | a calibration set | edge devices, CNNs |
| Quantization-aware training | simulated during training (fake quant) | training data | when post-training quantization loses too much |

## Build It

### Step 1: The baseline

An MLP with two hidden layers of 256 units, trained on scikit-learn's digits: 97.5% test accuracy and 85,002 parameters, about 340 KB of fp32 weights.

### Step 2: Affine and symmetric quantizers

```python
def quantize_symmetric(x, bits, dim=None):
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        max_abs = x.abs().max()
    else:
        max_abs = x.abs().amax(dim=dim, keepdim=True)
    scale = max_abs.clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q, scale
```

`quantize_affine` follows the formulas above. On the middle weight matrix, both give a mean absolute error near `6.5e-4`, well below the bound `scale / 2 = 1.3e-3`. The weights are nearly centered, so the zero point is 125 out of 255 and the asymmetric version gains almost nothing.

### Step 3: Per-channel scales

The `dim=1` argument computes one maximum per output row. The rows' max magnitudes range from 0.06 to 0.34, so per-tensor scales waste most of the grid on the small rows. Per-channel scales halve the error at both 8 and 4 bits.

### Step 4: Bit-width sweep

`fake_quantize_weights` quantizes and immediately dequantizes every weight matrix (the network still runs in float, but with values from the low-precision grid). Accuracy is unchanged at 8 and 6 bits, drops by about half a point at 4 bits, and at 2 bits per-tensor quantization collapses to chance (10.6%) while per-channel still reaches 79%. The table also shows the weight storage: 337,920 bytes in fp32, 84,480 in int8, 42,240 in int4.

### Step 5: An int8 layer

```python
class Int8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        q, scale = quantize_symmetric(linear.weight.detach(), 8, dim=1)
        self.register_buffer("weight_q", q.to(torch.int8))
        self.register_buffer("weight_scale", scale.squeeze(1))
        self.register_buffer("bias", linear.bias.detach().clone())

    def forward(self, x):
        x_q, x_scale = quantize_symmetric(x, 8)
        accumulator = x_q.to(torch.int32) @ self.weight_q.to(torch.int32).T
        return accumulator.float() * (x_scale * self.weight_scale) + self.bias
```

Weights are stored as `torch.int8` with one fp32 scale per row; activations are quantized per batch (dynamic quantization). The matmul runs on int32 tensors. The quantized model makes the same prediction as fp32 on 100% of test images, and its saved `state_dict` shrinks from 343 KB to 92 KB.

## Use It

```python
from torch.ao.quantization import quantize_dynamic
library_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

This is the same scheme as `Int8Linear`: int8 per-channel weights and dynamically quantized activations. Accuracy and size match yours. The difference is speed: on CPU, `quantize_dynamic` calls optimized int8 kernels (FBGEMM or oneDNN) and beats fp32, while your version converts to int32 and uses a generic matmul, so it is slower than fp32. Quantization only pays off in time when the hardware and kernels support low-precision arithmetic.

Note that `torch.ao.quantization` is deprecated in recent PyTorch releases (the script silences the warnings) and its successor is the separate `torchao` library, which provides int8 and int4 weight-only quantization, float8 training and quantization-aware training. For large language models, `bitsandbytes`, GPTQ, AWQ and llama.cpp's GGUF formats implement the group-wise 4-bit schemes described above.

## Exercises

1. **Easy.** Add one artificial outlier: set `model[2].weight[0, 0] = 5.0` and rerun the per-tensor and per-channel 4-bit experiments. What happens to each, and why?
2. **Medium.** Implement group-wise quantization: reshape each weight row into groups of 32 and give each group its own scale. Compare 3-bit and 2-bit accuracy with per-channel quantization, and count the extra bytes the scales cost.
3. **Hard.** Implement quantization-aware training for 2-bit per-channel weights: in the forward pass use `w + (fake_quant(w) - w).detach()` (the straight-through estimator) and fine-tune for 10 epochs. How much of the gap to fp32 do you recover?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Quantization | "Lower precision" | Representing values as low-bit integers plus scale (and zero-point) parameters |
| Scale | "Step size" | The real-valued distance between adjacent integer levels |
| Zero point | "Offset" | The integer that represents the real value 0 in affine quantization |
| Symmetric quantization | "No offset" | Zero point fixed at 0, range set by `max\|x\|` |
| Per-channel | "One scale per row" | Separate scale for each output channel of a weight tensor |
| Dynamic quantization | "Quantize activations on the fly" | Weights quantized ahead of time, activation scales computed for each batch at runtime |
| Fake quantization | "Simulated quantization" | Quantize then immediately dequantize, so float code sees the rounding error |
| Int32 accumulation | "Wide accumulator" | Summing int8 products in 32-bit integers to avoid overflow before rescaling |

## Further Reading

- [Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (2017)](https://arxiv.org/abs/1712.05877) - the affine scheme and integer-only inference used across the industry
- [Nagel et al., A White Paper on Neural Network Quantization (2021)](https://arxiv.org/abs/2106.08295) - a practical, thorough guide to post-training quantization and QAT
- [Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (2022)](https://arxiv.org/abs/2210.17323) - 3-4 bit group-wise quantization of large language models
- [torchao](https://github.com/pytorch/ao) - PyTorch's current library for quantization and low-precision training
