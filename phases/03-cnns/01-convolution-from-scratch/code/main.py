import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits


def conv2d_loops(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    out_height = image_height - kernel_height + 1
    out_width = image_width - kernel_width + 1
    output = torch.zeros(out_height, out_width)
    for row in range(out_height):
        for col in range(out_width):
            patch = image[row:row + kernel_height, col:col + kernel_width]
            output[row, col] = (patch * kernel).sum()
    return output


def conv2d_multichannel_loops(images, kernels, bias):
    batch, in_channels, height, width = images.shape
    out_channels, _, kernel_height, kernel_width = kernels.shape
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1
    output = torch.zeros(batch, out_channels, out_height, out_width)
    for n in range(batch):
        for o in range(out_channels):
            for row in range(out_height):
                for col in range(out_width):
                    patch = images[n, :, row:row + kernel_height, col:col + kernel_width]
                    output[n, o, row, col] = (patch * kernels[o]).sum() + bias[o]
    return output


def im2col(images, kernel_height, kernel_width):
    batch, channels, height, width = images.shape
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1
    patches = images.unfold(2, kernel_height, 1).unfold(3, kernel_width, 1)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.reshape(batch, out_height * out_width, channels * kernel_height * kernel_width), out_height, out_width


def conv2d_im2col(images, kernels, bias):
    out_channels, _, kernel_height, kernel_width = kernels.shape
    columns, out_height, out_width = im2col(images, kernel_height, kernel_width)
    flat_kernels = kernels.reshape(out_channels, -1)
    output = columns @ flat_kernels.T + bias
    return output.permute(0, 2, 1).reshape(images.shape[0], out_channels, out_height, out_width)


def correlation_vs_convolution(image, kernel):
    correlated = conv2d_loops(image, kernel)
    convolved = conv2d_loops(image, torch.flip(kernel, dims=(0, 1)))
    return correlated, convolved


def main():
    torch.manual_seed(0)
    digits = load_digits()
    images = torch.tensor(digits.images / 16.0, dtype=torch.float32)
    digit = images[0]

    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    sobel_y = sobel_x.T.contiguous()
    blur = torch.full((3, 3), 1.0 / 9.0)

    vertical_edges = conv2d_loops(digit, sobel_x)
    horizontal_edges = conv2d_loops(digit, sobel_y)
    blurred = conv2d_loops(digit, blur)
    print(f"Digit label: {digits.target[0]}, image shape: {tuple(digit.shape)}")
    print(f"3x3 kernel on 8x8 image -> output shape {tuple(vertical_edges.shape)}")
    print("Vertical-edge response (Sobel x), rounded:")
    print(vertical_edges.round(decimals=1))

    correlated, convolved = correlation_vs_convolution(digit, sobel_x)
    print(f"\nFlipping Sobel x negates it, so true convolution = -cross-correlation: "
          f"{torch.allclose(correlated, -convolved)}")

    batch = images[:16].unsqueeze(1)
    kernels = torch.randn(4, 1, 3, 3)
    bias = torch.randn(4)
    loop_output = conv2d_multichannel_loops(batch, kernels, bias)
    im2col_output = conv2d_im2col(batch, kernels, bias)
    torch_output = F.conv2d(batch, kernels, bias)
    print(f"\nLoop conv vs im2col conv max diff:   {(loop_output - im2col_output).abs().max():.2e}")
    print(f"im2col conv vs F.conv2d max diff:    {(im2col_output - torch_output).abs().max():.2e}")

    rgb_like = torch.randn(8, 3, 32, 32)
    rgb_kernels = torch.randn(16, 3, 5, 5)
    rgb_bias = torch.zeros(16)
    print(f"3-channel input {tuple(rgb_like.shape)} with 16 kernels of shape {tuple(rgb_kernels.shape[1:])} "
          f"-> {tuple(conv2d_im2col(rgb_like, rgb_kernels, rgb_bias).shape)}")
    print(f"im2col vs F.conv2d on 3 channels max diff: "
          f"{(conv2d_im2col(rgb_like, rgb_kernels, rgb_bias) - F.conv2d(rgb_like, rgb_kernels, rgb_bias)).abs().max():.2e}")

    timing_batch = images[:64].unsqueeze(1)
    start = time.perf_counter()
    conv2d_multichannel_loops(timing_batch, kernels, bias)
    loop_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(100):
        conv2d_im2col(timing_batch, kernels, bias)
    im2col_seconds = (time.perf_counter() - start) / 100
    start = time.perf_counter()
    for _ in range(100):
        F.conv2d(timing_batch, kernels, bias)
    torch_seconds = (time.perf_counter() - start) / 100
    print(f"\nTime for 64 images: loops {loop_seconds * 1000:.1f} ms, im2col {im2col_seconds * 1000:.3f} ms, "
          f"F.conv2d {torch_seconds * 1000:.3f} ms")

    dense_params = 64 * 36
    conv_params = 9
    print(f"\nParameters to map 8x8 -> 6x6: dense layer {dense_params}, one shared 3x3 kernel {conv_params}")

    shifted = torch.roll(digit, shifts=1, dims=1)
    shifted_response = conv2d_loops(shifted, sobel_x)
    print(f"Translation equivariance (shift input, output shifts too): "
          f"{torch.allclose(shifted_response[:, 1:], vertical_edges[:, :-1], atol=1e-6)}")

    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [(digit, "input"), (vertical_edges, "Sobel x"), (horizontal_edges, "Sobel y"), (blurred, "blur")]
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    output_path = Path(__file__).parent / "convolution_filters.png"
    figure.savefig(output_path, dpi=100)
    print(f"\nSaved {output_path.name}")


if __name__ == "__main__":
    main()
