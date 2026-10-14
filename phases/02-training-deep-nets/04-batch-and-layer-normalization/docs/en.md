# Batch and Layer Normalization

> Instead of hoping the init keeps every layer's activations well scaled, force them to be: subtract the mean, divide by the std, then let the network learn the scale it wants.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Neural Networks: Backpropagation by Hand), Phase 02 lessons SGD, Momentum and Adam and Weight Initialization
**Time:** ~90 minutes

## Learning Objectives

- Implement batch normalization and layer normalization forward and backward in NumPy and verify the gradients numerically
- Explain the learnable scale and shift (`gamma`, `beta`) and why they do not undo the normalization
- Explain the difference between train mode and eval mode for batch norm and why running statistics exist
- Choose between batch norm and layer norm and match both against `nn.BatchNorm1d` and `nn.LayerNorm`

## The Problem

In the last lesson you kept activations well scaled with a careful init. That only holds at step 0. As soon as the weights start moving, each layer's input distribution shifts, the layer after it has to keep adapting, and a deep network can drift back into vanishing or exploding activations. You also saw how sensitive things are: an init that is off by a factor of 2 per layer kills a 10-layer network.

Normalization layers fix the scale explicitly at every step. Insert one after a linear layer and its output has zero mean and unit variance no matter what the weights do. In practice this lets you train much deeper networks, use larger learning rates, and care far less about the init. Batch norm made very deep CNNs practical in 2015; layer norm is inside every transformer. You will see both in nearly every architecture in the rest of this course.

## The Concept

### The normalization step

Given a matrix of activations `x` with shape `(batch, features)`:

```
mean = average of x over some axis
var  = variance of x over the same axis
x_hat = (x - mean) / sqrt(var + eps)
y = gamma * x_hat + beta
```

`eps` (1e-5) avoids dividing by zero. `gamma` and `beta` are learned per feature, initialized to 1 and 0. If the network wants the original scale back it can learn `gamma = sqrt(var)` and `beta = mean`. The point is that the scale is now a direct parameter instead of an emergent product of many weight matrices.

### Which axis

```
                features ->
            +---+---+---+---+
 batch   x1 | . | . | . | . |   LayerNorm: statistics along this row,
   |     x2 | . | . | . | . |   one mean/var per example
   v     x3 | . | . | . | . |
            +---+---+---+---+
              ^
              BatchNorm: statistics down this column,
              one mean/var per feature
```

| | BatchNorm | LayerNorm |
|--|-----------|-----------|
| Statistics over | the batch, per feature | the features, per example |
| Depends on other examples | yes | no |
| Train vs eval behavior | different (running stats) | identical |
| Works with batch size 1 | no | yes |
| Typical home | CNNs | transformers, RNNs |

For images, BatchNorm2d computes statistics over batch, height and width for each channel.

### Train mode and eval mode

At training time batch norm uses the current batch's mean and variance. At test time you may get one example at a time, and you want the prediction for an image not to depend on which other images happen to be in the batch. So during training batch norm keeps exponential moving averages:

```
running_mean = (1 - m) * running_mean + m * batch_mean
running_var  = (1 - m) * running_var  + m * batch_var * n / (n - 1)
```

with momentum `m = 0.1` (PyTorch's convention; Keras calls `1 - m` the momentum). In eval mode those running values replace the batch statistics. Forgetting to switch modes is one of the most common bugs in deep learning code.

### The backward pass

Every output depends on every input in the same group through the mean and variance, so the gradient has three terms. Writing `g = dL/dx_hat = dL/dy * gamma` and `N` for the group size:

```
dL/dx = (1 / (N * sqrt(var + eps))) * ( N * g - sum(g) - x_hat * sum(g * x_hat) )
dL/dgamma = sum over batch of dL/dy * x_hat
dL/dbeta  = sum over batch of dL/dy
```

The two subtracted terms project out the components of the gradient that would change the group's mean or scale, which the normalization removes anyway. The same formula works for both layers; only the axis changes.

### Why it helps

The original paper argued that batch norm reduces "internal covariate shift". Later work showed the bigger effect is that it makes the loss surface smoother, so larger learning rates are stable. Either way, the practical effects are robust:

- activations stay well scaled regardless of init and depth
- training tolerates higher learning rates
- batch statistics add a little noise, acting as a mild regularizer

## Build It

### Step 1: A shared forward and backward

```python
def normalize_forward(x, gamma, beta, axis, eps):
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    return gamma * x_hat + beta, (x_hat, inv_std, mean, var)


def normalize_backward(grad_out, gamma, cache, axis):
    x_hat, inv_std, _, _ = cache
    n = x_hat.shape[axis]
    grad_x_hat = grad_out * gamma
    return inv_std / n * (
        n * grad_x_hat
        - grad_x_hat.sum(axis=axis, keepdims=True)
        - x_hat * (grad_x_hat * x_hat).sum(axis=axis, keepdims=True)
    )
```

`BatchNorm` calls these with `axis=0` and adds running statistics; `LayerNorm` calls them with `axis=1`. On a batch whose features have means `[3.2, -3.0, 0.2]` and stds `[10.6, 0.09, 0.92]`, the output has mean 0 and std 1 in every column.

### Step 2: Gradient check

Compare the analytic backward with central differences on a tiny `(6, 4)` input:

```
BatchNorm: max |dx analytic - numeric| = 1.3e-09, max |dgamma diff| = 7.8e-10
LayerNorm: max |dx analytic - numeric| = 1.0e-09, max |dgamma diff| = 6.3e-10
```

Always do this when you write a backward pass by hand. The Debugging Training lesson turns it into a reusable tool.

### Step 3: A layer-based MLP

This lesson switches from a parameter dictionary to small layer objects, each with `forward(x, training)`, `backward(grad)`, `params` and `grads`. The training loop runs the layers forward, then backward in reverse, then applies momentum to every layer's parameters:

```python
loss, grad = cross_entropy(run_forward(layers, X[batch], True), y[batch])
for layer in reversed(layers):
    grad = layer.backward(grad)
for layer, velocity in zip(layers, velocities):
    for name in layer.params:
        velocity[name] = beta * velocity[name] + layer.grads[name]
        layer.params[name] -= lr * velocity[name]
```

Train an 8-layer MLP (`Linear -> Norm -> ReLU` blocks) on digits with a deliberately poor init (std 0.05, far below He's 0.125) for 5 epochs:

```
norm=none  lr=0.02 | first-epoch loss 2.304 | last-epoch loss 2.303 | test acc 0.083
norm=none  lr=0.2  | first-epoch loss 2.311 | last-epoch loss 2.318 | test acc 0.093
norm=batch lr=0.02 | first-epoch loss 0.906 | last-epoch loss 0.130 | test acc 0.987
norm=batch lr=0.2  | first-epoch loss 1.236 | last-epoch loss 0.326 | test acc 0.970
norm=layer lr=0.02 | first-epoch loss 1.323 | last-epoch loss 0.357 | test acc 0.914
norm=layer lr=0.2  | first-epoch loss 2.715 | last-epoch loss 2.318 | test acc 0.093
```

Without normalization the network never leaves chance. Batch norm trains well at both learning rates. Layer norm also rescues the bad init, but in this plain MLP it is less forgiving of a large learning rate: normalizing each example across its features is a weaker constraint than normalizing each feature across the batch. In transformers, where it is combined with residual connections and warmup, it is the standard choice.

### Step 4: Train mode versus eval mode

```
eval-mode test acc 0.970
eval mode on a batch of 1  -> predicted 2 (uses running mean and variance)
train mode on a batch of 1 -> predicted 1 (every feature normalizes to 0, the input is ignored)
```

With a single example in train mode, each feature's batch mean equals the value itself, so `x_hat = 0` everywhere and the output is just `beta`, whatever the input. The prediction is the same for every image. Eval mode with running statistics gives the right answer.

## Use It

```python
bn = torch.nn.BatchNorm1d(5, dtype=torch.float64)
ln = torch.nn.LayerNorm(5, dtype=torch.float64)
out = bn(x_torch)
out.backward(upstream)
```

The script feeds the same input and upstream gradient to both implementations:

```
BatchNorm1d | out diff 4.4e-16 | dx diff 2.2e-16 | dgamma diff 8.9e-16 | running_var diff 4.4e-16
LayerNorm   | out diff 6.7e-16 | dx diff 6.7e-16 | dgamma diff 1.3e-15
```

Note that PyTorch normalizes with the biased variance but stores the unbiased one (`n / (n - 1)`) in `running_var`; your implementation does the same, which is why the running stats match.

In a real model:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(64, 128, bias=False), torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)
model.train()
model.eval()
```

- `model.train()` and `model.eval()` flip every batch norm (and dropout) layer at once. Call `eval()` before validation and inference.
- The linear layer before a batch norm does not need a bias: the norm subtracts the mean anyway and `beta` takes its role.
- Very small batches (below ~8) make batch statistics noisy. Use GroupNorm or LayerNorm instead, or sync batch norm across GPUs.
- Many recent language models use RMSNorm, which is layer norm without the mean subtraction: `x / sqrt(mean(x^2) + eps) * gamma`.

## Exercises

1. **Easy.** Train the batch norm network with batch size 4 instead of 32 and compare test accuracy in eval mode. What happens to the gap between train-mode and eval-mode accuracy?
2. **Medium.** Implement RMSNorm forward and backward, gradient check it, and add it as a third option in Step 3.
3. **Hard.** Put the norm after the ReLU instead of before it, and try a pre-norm residual MLP (`x + Linear(ReLU(Norm(x)))`) with 20 blocks. Compare batch norm and layer norm in both placements.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Batch normalization | "BatchNorm" | Normalize each feature with the batch mean and variance, then scale by `gamma` and shift by `beta` |
| Layer normalization | "LayerNorm" | Normalize each example across its features; independent of the batch |
| `gamma`, `beta` | "Affine params" | Learned per-feature scale and shift applied after normalizing |
| Running statistics | "BN buffers" | Moving averages of batch mean and variance used in eval mode |
| Train / eval mode | "model.eval()" | Switch between batch statistics and running statistics (and dropout on/off) |
| Internal covariate shift | "Why BN works" | The original motivation (inputs of each layer drift during training); later work points to a smoother loss landscape instead |
| RMSNorm | "LayerNorm without the mean" | Divide by the root mean square of the features; common in modern LLMs |

## Further Reading

- [Ioffe and Szegedy, Batch Normalization (2015)](https://arxiv.org/abs/1502.03167) - the original paper and the train/eval algorithm
- [Ba, Kiros and Hinton, Layer Normalization (2016)](https://arxiv.org/abs/1607.06450) - layer norm and why it suits recurrent networks
- [Santurkar et al., How Does Batch Normalization Help Optimization? (2018)](https://arxiv.org/abs/1805.11604) - the smoother-landscape explanation
- [Karpathy, Building makemore Part 3: Activations, Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) - implementing and debugging batch norm by hand
