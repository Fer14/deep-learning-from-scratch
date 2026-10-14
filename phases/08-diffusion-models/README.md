# Phase 08: Diffusion Models

> Learn to remove a little noise, then do it a thousand times.

This phase builds diffusion models from the forward noising process up. You start by destroying data with a Gaussian schedule and measuring how fast information disappears, then write a complete DDPM (training loss, timestep embedding, ancestral sampler) by hand. The score-matching lesson explains why it works: a noise predictor is a scaled score, and annealed Langevin dynamics turns a score into samples. DDIM reuses the same trained model with a deterministic sampler that needs 50 steps instead of 1000 and can invert data back to noise. Classifier-free guidance trades diversity for adherence with one extra forward pass. Latent diffusion moves the whole process into an autoencoder's latent space, the recipe behind Stable Diffusion. The phase ends with flow matching: regress a velocity field onto straight lines, sample with any ODE solver, and straighten the paths with reflow until one step is enough. The 2D lessons run on one CPU thread in under a minute; latent diffusion uses the GPU when available.

**Prerequisites:** Phase 02 (Training Deep Nets), Phase 05 (Transformers: Positional Encodings), Phase 06 (Autoencoders and Self-Supervised), Phase 07 (GANs)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Forward Noising](01-forward-noising/) | Linear and cosine schedules, the closed-form `q(x_t | x_0)`, SNR curves and a noised-digit classifier test |
| 02 | [DDPM from Scratch](02-ddpm-from-scratch/) | A time-conditioned noise predictor and the 1000-step ancestral sampler on a 2D swiss roll |
| 03 | [Score Matching Intuition](03-score-matching-intuition/) | Exact and learned scores, plain vs annealed Langevin, and denoising score matching |
| 04 | [Faster Sampling with DDIM](04-faster-sampling-ddim/) | The DDIM sampler with `eta`, a steps-vs-quality sweep and deterministic inversion |
| 05 | [Classifier-Free Guidance](05-classifier-free-guidance/) | Label dropout, guided DDIM sampling and the guidance-scale trade-off on overlapping moons |
| 06 | [Latent Diffusion](06-latent-diffusion/) | An autoencoder plus a latent DDPM on digits, compared with pixel-space diffusion |
| 07 | [Flow Matching](07-flow-matching/) | A velocity field on straight paths, Euler and RK45 sampling, and one round of reflow |

Run the first lesson:

```bash
uv run python phases/08-diffusion-models/01-forward-noising/code/main.py
```
