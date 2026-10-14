# Weight Initialization

> Before the first gradient step, the random weights already decide whether signals survive the trip through the network.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Neural Networks: Activation Functions, Backpropagation by Hand), Phase 02 lessons SGD, Momentum and Adam and Learning Rate Schedules
**Time:** ~60 minutes

## Learning Objectives

- Measure how activation and gradient magnitudes grow or shrink layer by layer for different initial weight scales
- Derive the Xavier (Glorot) and He (Kaiming) variances from the requirement that the variance stays constant across layers
- Show that a constant initialization never breaks symmetry between hidden units
- Use `torch.nn.init` and know what `nn.Linear` does by default

## The Problem

Take a 10-layer MLP, fill every weight matrix with `np.random.randn`, and push a batch through it. With ReLU, the output is around `1e10`. Fill it with `0.01 * np.random.randn` instead and the output is around `1e-10`. Neither network can learn: in the first case the loss is infinite and gradients overflow; in the second every layer passes on almost nothing, so the gradient that reaches the first layers is zero for all practical purposes.

For a long time this was a big reason deep networks were "impossible to train" and people resorted to layer-by-layer pretraining. The fix turned out to be a one-line change: scale the random weights by the right function of the layer width. Every framework does this for you now, but you need to understand it, because it is the first thing to break when you write a custom layer, change an activation function, or stack 100 layers.

## The Concept

### Variance through one layer

A layer computes `z_j = sum_i W_ij x_i` over `n_in` inputs. If the weights and inputs are independent with zero mean,

```
Var(z) = n_in * Var(W) * Var(x)
```

So each layer multiplies the variance by `n_in * Var(W)`. After `L` layers you get that factor to the power `L`:

```
N(0, 0.01), width 256:  n_in * Var(W) = 256 * 1e-4 = 0.0256   ->  0.0256^10 ~ 1e-16
N(0, 1),    width 256:  n_in * Var(W) = 256 * 1    = 256      ->  256^10    ~ 1e24
```

Unless the factor is close to 1, the signal vanishes or explodes exponentially with depth.

```
   factor < 1                factor = 1               factor > 1
 x -> . -> . -> . -> 0     x -> x -> x -> x        x -> X -> XX -> inf
```

### Xavier / Glorot

For activations that behave like the identity near zero (tanh, sigmoid around its center), keeping the forward variance constant means `Var(W) = 1 / n_in`. Keeping the backward (gradient) variance constant means `Var(W) = 1 / n_out`. Glorot and Bengio compromised:

```
Var(W) = 2 / (n_in + n_out)
```

### He / Kaiming

ReLU zeroes out half of its inputs, so it halves the second moment of the signal. To compensate you double the variance:

```
Var(W) = 2 / n_in
```

This is the init that made very deep ReLU networks (and ResNets) trainable from scratch.

| Init | Std | Use with |
|------|-----|----------|
| Xavier / Glorot | `sqrt(2 / (n_in + n_out))` | tanh, sigmoid, linear layers |
| He / Kaiming | `sqrt(2 / n_in)` | ReLU, Leaky ReLU, GELU (close enough) |
| LeCun | `sqrt(1 / n_in)` | SELU, and the default many libraries approximate |

### Backward pass, same story

Backprop multiplies the gradient by `W^T` and by the activation derivative at every layer. The same variance factor applies in reverse, so a bad init gives vanishing or exploding gradients too. A good init keeps both the forward signal and the backward gradient at roughly constant scale.

### Symmetry breaking

Why random at all? If every weight starts at the same value, every hidden unit in a layer computes the same output and receives the same gradient. They stay identical forever: a layer of 32 units is really one unit copied 32 times. Randomness is what lets units specialize. Biases, on the other hand, can safely start at zero.

## Build It

### Step 1: Four initializers

```python
def small_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, 0.01, size=(fan_in, fan_out))


def unit_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, 1.0, size=(fan_in, fan_out))


def xavier_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))


def he_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
```

### Step 2: Watch the signal

Push unit-variance inputs through 10 layers of width 256, record the std after each activation, then push a random gradient back through the same layers:

```python
for W in weights:
    z = hidden @ W
    hidden = activation(z)
    forward_stds.append(hidden.std())
grad = rng.normal(size=hidden.shape)
for W, z, a in zip(reversed(weights), reversed(pre_activations), reversed(outputs)):
    grad = (grad * derivative(z, a)) @ W.T
    backward_stds.append(grad.std())
```

The output at layers 1, 4, 7 and 10:

```
  relu N(0, 0.01) | forward   9.33e-02  1.33e-04  2.07e-07  3.84e-10
  relu N(0, 1)    | forward   9.33e+00  1.33e+04  2.07e+07  3.84e+10
  relu Xavier     | forward   5.83e-01  2.02e-01  7.72e-02  3.49e-02
  relu He         | forward   8.25e-01  8.10e-01  8.74e-01  1.12e+00
```

The pattern is exactly the geometric series from the concept section. With ReLU, Xavier is too small by a factor of about `sqrt(2)` per layer, so it shrinks slowly; He stays flat. With tanh, `N(0, 1)` does not explode (tanh is bounded) but every unit is saturated near +-1, and the backward pass shows the gradient exploding toward the input layers instead (4e4 at layer 1). Xavier with tanh shrinks gently, which is the known weakness of tanh in deep stacks.

### Step 3: Train with each

A 9-layer ReLU MLP (64 -> 8 x 128 -> 10) on standardized 8x8 digits, SGD with momentum, 5 epochs:

```
N(0, 0.01) | initial loss      2.303 | test loss after 5 epochs    2.311 | test acc 0.083
N(0, 1)    | initial loss     23.842 | test loss after 5 epochs      nan | test acc 0.108
Xavier     | initial loss      2.305 | test loss after 5 epochs    0.111 | test acc 0.972
He         | initial loss      2.812 | test loss after 5 epochs    0.110 | test acc 0.965
```

Small init never leaves chance level: the gradient reaching the early layers is about `1e-10`. Unit init starts with absurdly confident wrong predictions (the 23.8 is capped by the `1e-12` inside the log) and goes to NaN. Xavier and He both train at this depth. Make the network 30 layers deep and Xavier starts failing too.

### Step 4: Constant init

```python
constant_init = lambda fan_in, fan_out, rng: np.full((fan_in, fan_out), 0.05)
```

After three epochs every column of the first weight matrix is still identical (max difference `0.00e+00`) and the 32-unit network scores 14% on the test set. Every unit received the same gradient at every step.

## Use It

PyTorch puts the formulas in `torch.nn.init`. Note that PyTorch stores `Linear` weights as `(out_features, in_features)`, and the functions compute fans from that shape:

```python
weight = torch.empty(256, 512)
torch.nn.init.xavier_normal_(weight)
torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")
```

```
xavier_normal_   std 0.0511 vs formula sqrt(2/(in+out)) = 0.0510
kaiming_normal_  std 0.0624 vs formula sqrt(2/in)        = 0.0625
nn.Linear default std 0.0255 vs uniform(+-1/sqrt(in)) std = 0.0255
```

The default for `nn.Linear` and `nn.Conv2d` is uniform on `[-1/sqrt(n_in), 1/sqrt(n_in)]`, a variance of `1 / (3 n_in)`. That is smaller than He by a factor of 6. Stack 10 of those with ReLU and nothing in between:

```
nn.Linear default init : 8.54e-05
kaiming_normal_ (He)   : 7.95e-01
xavier_normal_         : 2.48e-02
```

In practice the default rarely hurts because modern networks have normalization layers and residual connections (next lessons and Phase 03) that rescale the signal anyway. When you build a plain deep MLP or a custom layer, initialize explicitly:

```python
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        torch.nn.init.zeros_(module.bias)
```

## Exercises

1. **Easy.** Rerun Step 2 with depth 50. At which layer does Xavier with ReLU fall below `1e-3`? Does He stay near 1?
2. **Medium.** Derive the gain for Leaky ReLU with negative slope `a` (the fraction of the second moment it keeps is `(1 + a^2) / 2`) and verify it against `torch.nn.init.calculate_gain("leaky_relu", a)`.
3. **Hard.** Implement LSUV (layer-sequential unit-variance) init: start from an orthogonal init, then for each layer rescale `W` until the output std on a batch is 1. Compare it with He on a 30-layer tanh network.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Fan-in / fan-out | "Layer width" | Number of inputs to / outputs from each unit of a layer |
| Vanishing activations | "Signal dies" | Per-layer variance factor below 1, so activations shrink exponentially with depth |
| Exploding activations | "NaNs at step 0" | Per-layer variance factor above 1, so activations grow exponentially |
| Xavier / Glorot init | "Xavier" | `Var(W) = 2/(n_in + n_out)`, balancing forward and backward variance for tanh-like units |
| He / Kaiming init | "Kaiming" | `Var(W) = 2/n_in`, compensating for ReLU zeroing half the inputs |
| Symmetry breaking | "Why init randomly" | Different initial weights so units in a layer get different gradients and learn different features |

## Further Reading

- [Glorot and Bengio, Understanding the difficulty of training deep feedforward neural networks (2010)](https://proceedings.mlr.press/v9/glorot10a.html) - the Xavier init paper, with activation histograms like Step 2
- [He et al., Delving Deep into Rectifiers (2015)](https://arxiv.org/abs/1502.01852) - derives the ReLU init and PReLU
- [Karpathy, Building makemore Part 3: Activations, Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) - live diagnosis of init problems with histograms
- [PyTorch docs: torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) - every initializer and the gain table
