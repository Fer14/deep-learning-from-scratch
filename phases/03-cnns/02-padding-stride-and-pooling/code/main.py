import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits


def output_size(input_size, kernel_size, padding, stride, dilation=1):
    effective_kernel = dilation * (kernel_size - 1) + 1
    return (input_size + 2 * padding - effective_kernel) // stride + 1


def zero_pad(images, padding):
    batch, channels, height, width = images.shape
    padded = torch.zeros(batch, channels, height + 2 * padding, width + 2 * padding)
    padded[:, :, padding:padding + height, padding:padding + width] = images
    return padded


def extract_patches(images, kernel_size, stride):
    batch, channels, height, width = images.shape
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    patches = images.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    return patches.reshape(batch, channels, out_height, out_width, kernel_size * kernel_size)


def conv2d(images, kernels, bias, padding=0, stride=1):
    out_channels, in_channels, kernel_size, _ = kernels.shape
    padded = zero_pad(images, padding) if padding else images
    patches = extract_patches(padded, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4).reshape(*patches.shape[0:1], *patches.shape[2:4], -1)
    output = patches @ kernels.reshape(out_channels, -1).T + bias
    return output.permute(0, 3, 1, 2)


def max_pool2d(images, kernel_size, stride):
    return extract_patches(images, kernel_size, stride).max(dim=-1).values


def avg_pool2d(images, kernel_size, stride):
    return extract_patches(images, kernel_size, stride).mean(dim=-1)


def global_avg_pool(images):
    return images.mean(dim=(2, 3))


def main():
    torch.manual_seed(0)
    digits = load_digits()
    images = torch.tensor(digits.images[:32] / 16.0, dtype=torch.float32).unsqueeze(1)

    print("Output size formula: floor((n + 2p - k) / s) + 1")
    print(f"{'n':>3} {'k':>3} {'p':>3} {'s':>3} {'out':>4}")
    for n, k, p, s in [(8, 3, 0, 1), (8, 3, 1, 1), (8, 3, 1, 2), (8, 5, 2, 1), (32, 5, 0, 1), (32, 3, 1, 2), (224, 7, 3, 2)]:
        print(f"{n:>3} {k:>3} {p:>3} {s:>3} {output_size(n, k, p, s):>4}")

    kernels = torch.randn(6, 1, 3, 3)
    bias = torch.randn(6)
    print("\nYour conv vs F.conv2d:")
    for padding, stride in [(0, 1), (1, 1), (1, 2), (2, 3)]:
        mine = conv2d(images, kernels, bias, padding=padding, stride=stride)
        reference = F.conv2d(images, kernels, bias, padding=padding, stride=stride)
        print(f"  padding={padding} stride={stride}: shape {tuple(mine.shape)}, "
              f"max diff {(mine - reference).abs().max():.1e}")

    print(f"\nWithout padding, an 8x8 image shrinks: 8 -> {output_size(8, 3, 0, 1)} -> "
          f"{output_size(6, 3, 0, 1)} -> {output_size(4, 3, 0, 1)} after three 3x3 convs")
    print(f"With 'same' padding (p=1) it stays 8 -> {output_size(8, 3, 1, 1)} -> {output_size(8, 3, 1, 1)}")
    corner_uses = 1
    center_uses = 9
    print(f"Without padding a corner pixel is seen by {corner_uses} kernel position, a center pixel by {center_uses}")

    feature_maps = F.relu(conv2d(images, kernels, bias, padding=1))
    pooled_max = max_pool2d(feature_maps, 2, 2)
    pooled_avg = avg_pool2d(feature_maps, 2, 2)
    print(f"\nFeature maps {tuple(feature_maps.shape)} -> 2x2 max pool {tuple(pooled_max.shape)}")
    print(f"Max pool vs F.max_pool2d diff: {(pooled_max - F.max_pool2d(feature_maps, 2, 2)).abs().max():.1e}")
    print(f"Avg pool vs F.avg_pool2d diff: {(pooled_avg - F.avg_pool2d(feature_maps, 2, 2)).abs().max():.1e}")
    print(f"Global average pool: {tuple(feature_maps.shape)} -> {tuple(global_avg_pool(feature_maps).shape)}")

    example = torch.tensor([[[[1.0, 3.0, 2.0, 0.0],
                              [4.0, 2.0, 1.0, 1.0],
                              [0.0, 1.0, 5.0, 6.0],
                              [2.0, 2.0, 7.0, 8.0]]]])
    print("\n4x4 example:")
    print(example[0, 0])
    print("2x2 max pool, stride 2:")
    print(max_pool2d(example, 2, 2)[0, 0])
    print("2x2 avg pool, stride 2:")
    print(avg_pool2d(example, 2, 2)[0, 0])

    shifted = torch.roll(feature_maps, shifts=1, dims=3)
    raw_change = (shifted - feature_maps).norm() / feature_maps.norm()
    pooled_change = (max_pool2d(shifted, 2, 2) - pooled_max).norm() / pooled_max.norm()
    print(f"\nShift feature maps by 1 pixel: raw maps change by {raw_change:.0%}, "
          f"max-pooled maps change by {pooled_change:.0%}")

    pooling_input = feature_maps.clone().requires_grad_(True)
    max_pool2d(pooling_input, 2, 2).sum().backward()
    routed = (pooling_input.grad != 0).float().mean()
    print(f"Max pool backward: only {routed:.0%} of inputs receive gradient (one winner per window)")

    strided = conv2d(images, kernels, bias, padding=1, stride=2)
    conv_then_pool = max_pool2d(conv2d(images, kernels, bias, padding=1), 2, 2)
    print(f"\nDownsampling 8x8 -> 4x4 two ways: stride-2 conv {tuple(strided.shape[2:])}, "
          f"conv + 2x2 pool {tuple(conv_then_pool.shape[2:])}")
    stride1_flops = 8 * 8 * 9 * 6
    stride2_flops = 4 * 4 * 9 * 6
    print(f"Multiply-adds per image: stride 1 conv {stride1_flops}, stride 2 conv {stride2_flops}")

    dilated = F.conv2d(images, kernels, bias, padding=2, dilation=2)
    print(f"\nDilation 2: a 3x3 kernel covers {2 * (3 - 1) + 1}x{2 * (3 - 1) + 1} pixels, "
          f"output {tuple(dilated.shape[2:])} with padding 2 "
          f"(formula gives {output_size(8, 3, 2, 1, dilation=2)})")


if __name__ == "__main__":
    main()
