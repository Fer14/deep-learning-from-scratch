import io
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def load_data():
    digits = load_digits()
    x = torch.tensor(digits.data, dtype=torch.float32) / 16.0
    y = torch.tensor(digits.target)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    x, y = x[order], y[order]
    return (x[:1400], y[:1400]), (x[1400:], y[1400:])


def make_mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def train(model, data, epochs=40):
    x, y = data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for i in range(0, len(x), 64):
            loss = F.cross_entropy(model(x[i:i + 64]), y[i:i + 64])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def accuracy(model, data):
    x, y = data
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def quantize_affine(x, bits):
    qmin, qmax = 0, 2**bits - 1
    scale = (x.max() - x.min()).clamp(min=1e-8) / (qmax - qmin)
    zero_point = torch.round(qmin - x.min() / scale).clamp(qmin, qmax)
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return q, scale, zero_point


def dequantize_affine(q, scale, zero_point):
    return (q - zero_point) * scale


def quantize_symmetric(x, bits, dim=None):
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        max_abs = x.abs().max()
    else:
        max_abs = x.abs().amax(dim=dim, keepdim=True)
    scale = max_abs.clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q, scale


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


def replace_linears(model, make_layer):
    quantized = make_mlp()
    quantized.load_state_dict(model.state_dict())
    for i, layer in enumerate(quantized):
        if isinstance(layer, nn.Linear):
            quantized[i] = make_layer(layer)
    return quantized.eval()


def fake_quantize_weights(model, bits, per_channel):
    def make_layer(layer):
        q, scale = quantize_symmetric(layer.weight.detach(), bits, dim=1 if per_channel else None)
        layer.weight.data = q * scale
        return layer
    return replace_linears(model, make_layer)


def size_in_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def time_inference(model, x, repeats=200):
    with torch.no_grad():
        for _ in range(10):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
    return (time.perf_counter() - start) / repeats


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    train_data, test_data = load_data()
    model = train(make_mlp(), train_data)
    base_accuracy = accuracy(model, test_data)
    print(f"fp32 MLP 64-256-256-10: test accuracy {base_accuracy:.4f}, {sum(p.numel() for p in model.parameters())} parameters\n")

    print("1) Affine (asymmetric) vs symmetric 8-bit quantization of one weight matrix")
    weight = model[2].weight.detach()
    q, scale, zero_point = quantize_affine(weight, 8)
    affine_error = (dequantize_affine(q, scale, zero_point) - weight).abs().mean()
    q_sym, scale_sym = quantize_symmetric(weight, 8)
    symmetric_error = (q_sym * scale_sym - weight).abs().mean()
    print(f"  weight range [{weight.min():.3f}, {weight.max():.3f}]")
    print(f"  affine uint8:   scale {scale:.5f}, zero point {int(zero_point):3d}, mean |error| {affine_error:.6f}")
    print(f"  symmetric int8: scale {scale_sym:.5f}, zero point   0, mean |error| {symmetric_error:.6f}")
    print(f"  rounding error is at most scale/2 = {scale_sym / 2:.5f}")

    print("\n2) Per-tensor vs per-channel scales (one scale for the matrix vs one per output row)")
    row_ranges = weight.abs().amax(1)
    print(f"  row max |w| ranges from {row_ranges.min():.3f} to {row_ranges.max():.3f}")
    for bits in [8, 4]:
        q_t, s_t = quantize_symmetric(weight, bits)
        q_c, s_c = quantize_symmetric(weight, bits, dim=1)
        print(f"  {bits}-bit mean |error|: per-tensor {(q_t * s_t - weight).abs().mean():.6f} | "
              f"per-channel {(q_c * s_c - weight).abs().mean():.6f}")

    print("\n3) Weight-only quantization of the whole model, sweeping the bit width")
    print(f"  {'bits':>4s} {'per-tensor acc':>15s} {'per-channel acc':>16s} {'weight bytes':>13s}")
    n_weights = sum(layer.weight.numel() for layer in model if isinstance(layer, nn.Linear))
    for bits in [8, 6, 4, 3, 2]:
        tensor_acc = accuracy(fake_quantize_weights(model, bits, per_channel=False), test_data)
        channel_acc = accuracy(fake_quantize_weights(model, bits, per_channel=True), test_data)
        print(f"  {bits:4d} {tensor_acc:15.4f} {channel_acc:16.4f} {n_weights * bits // 8:13d}")
    print(f"  fp32 weights take {n_weights * 4} bytes")

    print("\n4) Real int8 inference: int8 weights, int8 activations, int32 accumulation")
    int8_model = replace_linears(model, Int8Linear)
    x_test, _ = test_data
    with torch.no_grad():
        logits_fp32, logits_int8 = model(x_test), int8_model(x_test)
    agreement = (logits_fp32.argmax(1) == logits_int8.argmax(1)).float().mean()
    print(f"  accuracy {accuracy(int8_model, test_data):.4f} (fp32 {base_accuracy:.4f}), "
          f"same prediction as fp32 on {agreement:.1%} of test images")
    print(f"  max |logit difference| {(logits_fp32 - logits_int8).abs().max():.4f}")
    print(f"  state_dict size: fp32 {size_in_bytes(model)} bytes | int8 {size_in_bytes(int8_model)} bytes")

    print("\n5) Use It: PyTorch dynamic quantization (int8 Linear kernels on CPU)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.ao.quantization import quantize_dynamic
        library_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        library_accuracy = accuracy(library_model, test_data)
        library_size = size_in_bytes(library_model)
    big_batch = x_test.repeat(4, 1)
    print(f"  accuracy {library_accuracy:.4f}, state_dict size {library_size} bytes")
    print(f"  CPU time per batch of {len(big_batch)}: fp32 {time_inference(model, big_batch) * 1e3:.3f} ms | "
          f"quantize_dynamic {time_inference(library_model, big_batch) * 1e3:.3f} ms | "
          f"our int8 (emulated) {time_inference(int8_model, big_batch) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
