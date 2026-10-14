# Phase 07: GANs

> Two networks, one game: a forger that learns only from a detective's mistakes.

This phase builds generative adversarial networks from the ground up. You start with a generator and a discriminator made of raw weight tensors learning a 2D ring, then break GANs on purpose: saturating gradients, mode collapse and mode hopping on a mixture of eight Gaussians, and the fixes (balanced players, packing). DCGAN brings convolutions and transposed convolutions (built from scratch) to 8x8 digits. The Wasserstein GAN replaces "real or fake" with a distance, implemented with weight clipping and with a gradient penalty. Conditional GANs let you ask for a specific digit. The phase closes with the ideas behind StyleGAN (mapping network, AdaIN, style mixing, truncation) and pix2pix-style image-to-image translation. The 2D lessons run on a CPU in under a minute; the digit lessons use the GPU when available and also run on a CPU.

**Prerequisites:** Phase 01 (Neural Networks), Phase 02 (Training Deep Nets), Phase 03 (CNNs), Phase 06 (Autoencoders and Self-Supervised)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [A GAN from Scratch](01-a-gan-from-scratch/) | A scratch generator and discriminator learning a 2D ring, scored with sliced Wasserstein distance |
| 02 | [Training Dynamics and Mode Collapse](02-training-dynamics-and-mode-collapse/) | Saturation measurements, a mode-hopping generator on 8 Gaussians, and three fixes compared |
| 03 | [DCGAN](03-dcgan/) | Transposed convolution from scratch and a DCGAN on 8x8 digits, compared with an MLP GAN |
| 04 | [Wasserstein GAN](04-wasserstein-gan/) | A critic with weight clipping and with a gradient penalty, plus spectral normalization by power iteration |
| 05 | [Conditional GANs](05-conditional-gans/) | A label-conditioned DCGAN that draws the digit you ask for, with a style-vs-content grid |
| 06 | [StyleGAN and Image-to-Image Intuition](06-stylegan-and-image-to-image-intuition/) | AdaIN, a tiny style-based generator with mixing and truncation, and pix2pix-style super-resolution |

Run the first lesson:

```bash
uv run python phases/07-gans/01-a-gan-from-scratch/code/main.py
```
