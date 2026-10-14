# Activation Functions

> The bend between layers decides what a network can represent and whether its gradients survive the trip back.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 00 (Tensors and Autodiff), Multilayer Perceptrons
**Time:** ~45 minutes

## Learning Objectives

- Implement sigmoid, tanh, ReLU, leaky ReLU, GELU and softmax together with their derivatives
- Explain saturation and why the sigmoid's maximum derivative of 0.25 causes vanishing gradients in deep stacks
- Measure gradient magnitude after backpropagating through 20 layers for each activation
- Write a numerically stable softmax and match every function against `torch.nn.functional`

## The Problem

You saw that without a nonlinearity, depth is useless. But which nonlinearity? For twenty years the answer was the sigmoid, because it looked like a smooth version of a firing neuron. Networks deeper than a few layers would barely train, and nobody was quite sure why.

Part of the answer turned out to be the activation's derivative. During backpropagation the gradient is multiplied by `f'(z)` at every layer. If `f'` is small, the product of twenty of them is tiny and the first layers stop learning. Switching to ReLU (Nair and Hinton 2010, Glorot et al. 2011) was one of the simple changes that made deep networks practical. Today transformers mostly use GELU or its relatives. Knowing what each one does to the forward signal and the backward gradient lets you read architecture choices instead of copying them.

## The Concept

### The common activations

| Name | `f(z)` | `f'(z)` | Range |
|------|--------|---------|-------|
| Sigmoid | `1 / (1 + e^-z)` | `f(z)(1 - f(z))` | (0, 1) |
| Tanh | `(e^z - e^-z) / (e^z + e^-z)` | `1 - f(z)^2` | (-1, 1) |
| ReLU | `max(0, z)` | `1 if z > 0 else 0` | [0, inf) |
| Leaky ReLU | `z if z > 0 else a z` | `1 if z > 0 else a` | (-inf, inf) |
| GELU | `z * Phi(z)` | smooth, slightly above 1 near z = 1 | about (-0.17, inf) |

`Phi` is the standard normal CDF. The common tanh approximation is

```
GELU(z) ~ 0.5 z (1 + tanh( sqrt(2/pi) (z + 0.044715 z^3) ))
```

```
   sigmoid               tanh                  relu
 1 |       ______      1 |       ______          |        /
   |     /               |     /                 |      /
.5 |    /              0 |    /                  |    /
   |   /                 |   /                   |  /
 0 |__/              -1  |__/                  0 |/________
   +--------- z          +--------- z            +--------- z
   flat ends: f' ~ 0     flat ends: f' ~ 0       f' = 0 left, 1 right
```

### Saturation

Where the curve is flat, the derivative is near zero. The sigmoid and tanh saturate at both ends: for `|z| > 5` their derivatives are below 0.01. Even at its best (`z = 0`) the sigmoid's derivative is only 0.25. Backpropagating through `L` sigmoid layers, the activation derivatives alone contribute a factor of at most `0.25^L`, about `1e-12` for `L = 20`, unless the weights are large enough to compensate.

ReLU has derivative exactly 1 for positive inputs, so it passes gradients through unchanged on the active path. Its weakness is the other side: a unit whose input is always negative gets zero gradient forever, a "dead ReLU". Leaky ReLU keeps a small slope `a` (usually 0.01) to avoid that. GELU is a smooth ReLU that weights inputs by how likely they are to be positive under a Gaussian; it is the default in BERT and GPT models.

### Zero-centered outputs

Sigmoid outputs are always positive, so all gradients flowing into the next layer's weights for one sample have the same sign, which makes updates zig-zag. Tanh is zero-centered, which is why it beat the sigmoid for hidden layers even before ReLU.

### Output activations are a separate choice

Hidden activations shape the representation. The output activation shapes the prediction:

| Task | Output activation | Pairs with |
|------|-------------------|-----------|
| Regression | identity | MSE |
| Binary classification | sigmoid | binary cross-entropy |
| Multiclass classification | softmax | cross-entropy |

Softmax turns a vector of scores (logits) into a probability distribution:

```
softmax(z)_i = e^(z_i) / sum_j e^(z_j)
```

It is invariant to adding a constant to every logit, which gives the stable implementation: subtract `max(z)` before exponentiating so the largest exponent is `e^0 = 1` and nothing overflows.

## Build It

### Step 1: Functions and derivatives

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
```

Tanh and leaky ReLU follow the same pattern. For GELU the script uses a central finite difference for the derivative, which is accurate enough for plotting and for the depth experiment.

### Step 2: Maximum derivatives

```
  max sigmoid   ' = 0.250
  max tanh      ' = 1.000
  max relu      ' = 1.000
  max leaky_relu' = 1.000
  max gelu      ' = 1.129
```

### Step 3: Gradients through 20 layers

```python
def gradient_through_depth(name, depth, width, seed, gain=1.0):
    rng = np.random.default_rng(seed)
    activation, derivative = ACTIVATIONS[name]
    h = rng.normal(size=(64, width))
    weights, pre_activations = [], []
    for _ in range(depth):
        W = rng.normal(0.0, np.sqrt(gain / width), size=(width, width))
        z = h @ W
        weights.append(W)
        pre_activations.append(z)
        h = activation(z)
    grad = np.ones_like(h)
    norms = []
    for W, z in zip(reversed(weights), reversed(pre_activations)):
        grad = (grad * derivative(z)) @ W.T
        norms.append(np.linalg.norm(grad) / np.sqrt(grad.size))
    return norms[::-1]
```

The backward loop is the chain rule applied layer by layer: multiply by `f'(z)` elementwise, then by `W^T`. You will derive this formally in the backpropagation lesson.

```
  sigmoid    layer 20 -> 2.26e-01   layer 1 -> 2.44e-13
  tanh       layer 20 -> 9.34e-01   layer 1 -> 2.09e-01
  relu       layer 20 -> 7.47e-01   layer 1 -> 2.11e-03
  leaky_relu layer 20 -> 7.46e-01   layer 1 -> 2.11e-03
  gelu       layer 20 -> 4.77e-01   layer 1 -> 2.21e-06
  relu + He  layer 20 -> 1.06e+00   layer 1 -> 2.16e+00
```

The sigmoid gradient is dead by the first layer. ReLU with weight variance `1/width` also shrinks, because ReLU zeroes half of its inputs and halves the signal variance at every layer. Doubling the variance to `2/width` (He initialization) compensates and the gradient stays order 1. Activation and initialization must be chosen together, a topic Phase 02 (Training Deep Nets) treats in depth.

### Step 4: Stable softmax

```python
def softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
```

`softmax([2, 1, 0.1]) = [0.659, 0.242, 0.099]`, and adding 1000 to every logit gives the same answer instead of `nan`.

The script also saves `activations.png` with every function and its derivative side by side.

## Use It

```python
import torch.nn.functional as F

torch.sigmoid(zt)
torch.tanh(zt)
F.relu(zt)
F.leaky_relu(zt, 0.01)
F.gelu(zt, approximate="tanh")
torch.softmax(logits, dim=0)
```

Every one matches your NumPy version to machine precision. The default `F.gelu` uses the exact `erf` formula; it differs from the tanh approximation by at most about `1.5e-4`. In model code you usually see the module versions (`nn.ReLU()`, `nn.GELU()`) inside `nn.Sequential`. You almost never apply softmax yourself before the loss: `F.cross_entropy` takes raw logits and does a fused, stable log-softmax inside, as you will see in the next lesson.

## Exercises

1. **Easy.** Show numerically that `tanh(z) = 2 sigmoid(2z) - 1` on a grid of points. What does this say about the expressive power of the two?
2. **Medium.** Implement SiLU (`z * sigmoid(z)`, also called Swish) and its analytic derivative, check it against a finite difference and `F.silu`, and add it to the depth experiment.
3. **Hard.** Count dead ReLUs: build a 10-layer ReLU MLP of width 64, initialize the biases to `-1.0`, push 1000 random inputs through, and report the fraction of units that are zero for every input in each layer. Repeat with leaky ReLU and explain the difference for training.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Activation function | "The nonlinearity" | An elementwise function applied after each linear layer so depth adds expressive power |
| Saturation | "The neuron is maxed out" | The region where `f'(z)` is near zero, so almost no gradient flows back |
| Vanishing gradient | "Early layers don't learn" | The gradient shrinks geometrically as it is multiplied by small derivatives and weights across many layers |
| ReLU | "max(0, x)" | A piecewise-linear activation with derivative 1 on the positive side and 0 on the negative side |
| Dead ReLU | "A dead neuron" | A ReLU unit whose input is negative for every sample, so it outputs 0 and never gets gradient |
| GELU | "The transformer activation" | `z * Phi(z)`, a smooth gate that weights the input by the probability a standard normal is below it |
| Softmax | "Turns scores into probabilities" | `exp(z_i) / sum exp(z_j)`, a normalized exponential; invariant to adding a constant to all logits |
| Logits | "Raw outputs" | The unnormalized scores fed to softmax or sigmoid |

## Further Reading

- [Glorot, Bordes and Bengio (2011), Deep Sparse Rectifier Neural Networks](https://proceedings.mlr.press/v15/glorot11a.html) - why ReLU helps deep networks train
- [Hendrycks and Gimpel (2016), Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) - the activation used in BERT and GPT
- [Dive into Deep Learning, Section 5.1.2: Activation Functions](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#activation-functions) - plots and derivatives of the classic choices
- [PyTorch docs: Non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) - the full list of built-in activations
