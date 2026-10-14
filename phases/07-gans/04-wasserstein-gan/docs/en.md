# Wasserstein GAN

> Stop asking "real or fake?" and start asking "how far?". A distance that shrinks smoothly gives the generator a gradient even when it is hopelessly wrong.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 07 Lessons 01-02 (A GAN from Scratch, Training Dynamics and Mode Collapse), Phase 00 (Tensors and Autodiff: PyTorch Tensors and Autograd)
**Time:** ~75 minutes

## Learning Objectives

- Explain why the Jensen-Shannon divergence gives no gradient between non-overlapping distributions, while the Wasserstein-1 distance does
- Implement a WGAN critic trained on the Kantorovich-Rubinstein dual, with weight clipping and with a gradient penalty computed through `torch.autograd.grad(create_graph=True)`
- Show that the critic's estimate tracks an independent sample-quality metric
- Relate the Lipschitz constraint to spectral normalization and compute a largest singular value by power iteration

## The Problem

The original GAN discriminator ends up measuring the Jensen-Shannon divergence between data and generator. Real images live on a thin manifold in pixel space, and so do a generator's outputs. Early in training the two manifolds typically do not overlap at all. In that situation the JS divergence is a constant, `log 2`, no matter how close or far the generator is. A perfect discriminator separates them completely, saturates, and passes back no useful gradient. This is the deeper reason behind the vanishing gradients and instability of lessons 01 and 02.

Arjovsky, Chintala and Bottou (2017) proposed replacing the divergence with the Wasserstein-1 distance, the "earth mover's distance": the minimum average distance you have to move probability mass to turn one distribution into the other. It keeps decreasing smoothly as the generator approaches the data, even when they do not overlap, so the generator always knows which way to go.

## The Concept

### Two point masses

Let `P` be all mass at 0 and `Q` all mass at `theta`.

```
theta   JS(P, Q)    W1(P, Q)
0       0           0
0.5     log 2       0.5
1.0     log 2       1.0
4.0     log 2       4.0
```

JS jumps from `log 2` to 0 only at the exact match, and is flat everywhere else: its gradient with respect to `theta` is zero. W1 is `|theta|`, with gradient pointing straight at the target. This is a one-dimensional caricature of two disjoint image manifolds.

### The dual form

Computing W1 directly needs an optimal transport plan. The Kantorovich-Rubinstein duality gives a form you can optimize with a neural network:

```
W1(P_data, P_G) = max over 1-Lipschitz f of   E_{x ~ data}[f(x)] - E_{z}[f(G(z))]
```

The network `f` is called a critic, not a discriminator: it outputs an unbounded score, no sigmoid, no probability. Training:

```
critic:     maximize  mean f(real) - mean f(fake)          (subject to f being 1-Lipschitz)
generator:  minimize  -mean f(G(z))
```

### Enforcing the Lipschitz constraint

Without a constraint the critic would push scores to infinity. 1-Lipschitz means `|f(a) - f(b)| <= ||a - b||`, or equivalently `||grad f|| <= 1` everywhere.

| Method | How | Problem |
|---|---|---|
| Weight clipping (WGAN) | clamp every weight to `[-c, c]` after each step | crude: weights pile up at the bounds, the critic becomes too simple, its scale is tiny |
| Gradient penalty (WGAN-GP) | add `lambda * (||grad_x f(x_hat)|| - 1)^2` at random points between real and fake | needs a second backward pass |
| Spectral normalization | divide each weight matrix by its largest singular value | bounds the Lipschitz constant per layer; cheap |

The gradient penalty is evaluated at `x_hat = alpha * real + (1 - alpha) * fake`, `alpha ~ U(0, 1)`, because the optimal critic has gradient norm exactly 1 on the straight lines connecting matched real and fake points.

### Training details that matter

- Train the critic more than the generator (`n_critic = 5`): the critic's score is only a valid distance estimate when it is near optimal. Unlike a GAN discriminator, a strong critic does not kill the generator's gradients.
- WGAN-GP uses Adam with `beta1 = 0` (no first-moment momentum) and `beta2 = 0.9`. The script shows the difference this makes in practice.
- No batch norm in the critic with a gradient penalty: the penalty is per sample, and batch norm mixes samples.

## Build It

The data is the 8-Gaussian mixture from lesson 02. Both networks are 2-hidden-layer MLPs with 128 ReLU units and 16-dimensional noise.

### Step 1: The gradient penalty

```python
def gradient_penalty(critic, real, fake, generator):
    alpha = torch.rand(len(real), 1, generator=generator)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(mixed)
    grads, = torch.autograd.grad(scores.sum(), mixed, create_graph=True)
    return ((grads.norm(dim=1) - 1) ** 2).mean()
```

`create_graph=True` keeps the graph of the gradient itself, so the penalty can be differentiated with respect to the critic's weights (a gradient of a gradient). Summing the scores before differentiating gives each sample's own input gradient, because samples do not interact.

### Step 2: The training loop

```python
for _ in range(n_critic):
    real = sample_mixture(batch_size, rng)
    with torch.no_grad():
        fake = generator_net(torch.randn(batch_size, NOISE_DIM, generator=rng))
    wasserstein_estimate = critic(real).mean() - critic(fake).mean()
    loss_c = -wasserstein_estimate + 1.0 * gradient_penalty(critic, real, fake, rng)
    ...
loss_g = -critic(generator_net(torch.randn(batch_size, NOISE_DIM, generator=rng))).mean()
```

The GP paper used `lambda = 10` for images and `0.1` for its 2D toy problems; this script uses 1, with Adam at learning rate `1e-3` and `betas = (0, 0.9)`. While preparing this lesson, a run with the usual GAN settings (`beta1 = 0.5`, `lambda = 10`, learning rate `1e-4`) oscillated and never covered a mode within 1500 steps; switching to `beta1 = 0` was the change that mattered most.

### Step 3: Watch the critic

```
G step    0 | critic W estimate  0.4811 | sliced W 1.212 | modes 0/8 | on-mode 0.00
G step  500 | critic W estimate  0.3945 | sliced W 0.194 | modes 4/8 | on-mode 0.19
G step  750 | critic W estimate  0.2879 | sliced W 0.121 | modes 8/8 | on-mode 0.46
G step 1000 | critic W estimate  0.1981 | sliced W 0.119 | modes 8/8 | on-mode 0.49
G step 1500 | critic W estimate  0.1013 | sliced W 0.091 | modes 8/8 | on-mode 0.70
correlation between the critic's W estimate and the sliced W metric: 0.97
critic gradient norm on probe points: mean 1.084, max 1.413 (target 1)
```

This is the practical win of WGANs. In lesson 02 the losses told you nothing; here the critic's estimate falls steadily and correlates at 0.97 with an independent metric (the sliced Wasserstein distance, computed by sorting projections). You can read training progress off the loss curve. All eight modes are found, with no collapse. The penalty keeps gradient norms near 1, not exactly 1: it is a soft constraint.

### Step 4: Weight clipping

The original WGAN: RMSprop, every critic weight clamped to `[-0.01, 0.01]` after each step.

```
G step  500 | critic W estimate  0.0001 | sliced W 0.296 | modes 0/8 | on-mode 0.04
G step 1000 | critic W estimate  0.0000 | sliced W 0.265 | modes 0/8 | on-mode 0.07
fraction of critic weights sitting at +-0.01: 0.55
```

It does not diverge, and it does spread mass over the right region (sliced W 0.265), but it is slow and blurry. More than half of the weights sit exactly at the clipping bound: the constraint, not the data, decides what the critic can represent. The estimate is also scaled down to about `1e-4`, because clipping makes the critic `K`-Lipschitz for some tiny `K`, so the number only means something up to that unknown factor. `wgan_samples.png` shows both generators.

## Use It

Spectral normalization (Miyato et al., 2018) is the cheap, library-provided way to bound a critic's Lipschitz constant. A linear layer `x -> W x` is exactly `sigma_max(W)`-Lipschitz, where `sigma_max` is the largest singular value. You can find it with power iteration:

```python
def power_iteration(weight, steps, generator):
    u = torch.randn(weight.shape[0], generator=generator)
    for _ in range(steps):
        v = weight.T @ u
        v = v / v.norm()
        u = weight @ v
        u = u / u.norm()
    return float(u @ weight @ v)
```

```
largest singular value: power iteration 0.9514 | torch.linalg.matrix_norm(ord=2) 0.9515
after spectral_norm: largest singular value 1.0164
```

`torch.nn.utils.parametrizations.spectral_norm` stores a `u` vector and runs one power iteration step per forward pass in training mode, then divides the weight by the estimate. After a few calls the effective weight has a largest singular value of about 1. Since ReLU is 1-Lipschitz, a stack of spectrally normalized layers is at most 1-Lipschitz overall. Spectral norm is often combined with the hinge loss instead of the Wasserstein loss (SNGAN, BigGAN).

## Exercises

1. **Easy.** Rerun WGAN-GP with `n_critic = 1`. Does the correlation between the critic estimate and the sliced Wasserstein distance survive?
2. **Medium.** Implement the one-sided penalty `max(0, ||grad|| - 1)^2` and compare it with the two-sided one.
3. **Hard.** Train a critic only (fixed point-mass generator at `theta`, data at 0, both in 1D) for several values of `theta`, and plot the final critic gap against `theta`. Do the same with a standard GAN discriminator and plot its loss. Which one recovers the W1 column of the table in the Concept section?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Wasserstein-1 distance | "Earth mover's distance" | The minimum expected distance needed to transport one distribution's mass onto another |
| Critic | "The WGAN discriminator" | A network with unbounded output trained to maximize `E f(real) - E f(fake)` under a Lipschitz constraint |
| Kantorovich-Rubinstein duality | "The dual form" | `W1 = sup over 1-Lipschitz f of E_P f - E_Q f` |
| Lipschitz constant | "How steep a function can be" | The smallest `K` with `|f(a) - f(b)| <= K ||a - b||` for all `a, b` |
| Weight clipping | "The original WGAN trick" | Clamping all critic weights to `[-c, c]`, a crude way to bound the Lipschitz constant |
| Gradient penalty | "WGAN-GP" | A loss term `lambda (||grad f(x_hat)|| - 1)^2` at interpolates between real and fake |
| Spectral normalization | "SN" | Dividing each weight by its largest singular value, estimated by power iteration |
| `create_graph=True` | "Double backprop" | Building a graph for the gradient so a loss on the gradient can itself be differentiated |

## Further Reading

- [Arjovsky, Chintala and Bottou, Wasserstein GAN (arXiv:1701.07875)](https://arxiv.org/abs/1701.07875) - the point-mass example, the dual form and weight clipping
- [Gulrajani et al., Improved Training of Wasserstein GANs (arXiv:1704.00028)](https://arxiv.org/abs/1704.00028) - the gradient penalty and the Adam settings used here
- [Arjovsky and Bottou, Towards Principled Methods for Training GANs (arXiv:1701.04862)](https://arxiv.org/abs/1701.04862) - why disjoint supports break the original GAN
- [Miyato et al., Spectral Normalization for GANs (arXiv:1802.05957)](https://arxiv.org/abs/1802.05957) - power iteration and per-layer Lipschitz control
