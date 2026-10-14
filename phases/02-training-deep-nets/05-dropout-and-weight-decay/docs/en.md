# Dropout and Weight Decay

> A big network can memorize anything. Dropout makes it unable to rely on any single unit; weight decay makes large weights cost something.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Neural Networks: An MLP from Scratch in NumPy), Phase 02 lesson SGD, Momentum and Adam; Classic ML from Scratch Phase 04 (ML Fundamentals: bias-variance, regularization) helps
**Time:** ~75 minutes

## Learning Objectives

- Implement inverted dropout in the forward and backward pass and explain why it rescales by `1 / (1 - p)`
- Explain L2 regularization as weight decay and implement decoupled weight decay (AdamW)
- Show on a small noisy dataset how both regularizers trade training accuracy for test loss
- Explain why `Adam(weight_decay=...)` and `AdamW(weight_decay=...)` are different algorithms and match both in PyTorch

## The Problem

Give a 2-256-256-2 MLP (66,000 parameters) 80 noisy points from two interleaving moons and train it long enough. It reaches 99% training accuracy by drawing a boundary that snakes around every noisy point. On fresh data from the same distribution its accuracy drops to 83%, and its test loss is 1.45: it is confidently wrong about many points, which is worse than being uncertain.

Getting more data is the best fix, but you do not always have it. Regularization changes the training objective or procedure so that the network prefers simpler functions. Classical ML adds a penalty on the weights; deep learning kept that idea under the name weight decay and added dropout, which has no clean classical equivalent. Both show up in nearly every training recipe, and both have a detail that is easy to get wrong: dropout must be off at test time, and weight decay interacts with Adam in a way that was only fixed in 2017.

## The Concept

### Dropout

During training, each hidden activation is set to zero with probability `p`, independently for every example and every step:

```
           training (p = 0.5)              eval
 x1 --- h1 ---\                 x1 --- h1 ---\
 x2 --- h2 (X) --- out          x2 --- h2 ----- out
 x3 --- h3 ---/                 x3 --- h3 ---/
        h4 (X)                         h4
```

Each step trains a different random subnetwork. A unit cannot count on any particular other unit being present, so it has to learn features that are useful on their own. That discourages co-adapted, fragile combinations of units that only fit the training set. You can also view the full network at test time as an approximate average of all those subnetworks.

### Inverted dropout

If half of the units are dropped, the next layer sees inputs that are on average half as large as at test time, when all units are present. Inverted dropout fixes the mismatch during training by scaling the survivors:

```
mask = (uniform(0, 1) >= p) / (1 - p)
h_train = h * mask
h_eval  = h
```

so `E[h_train] = h`. Eval mode is then just the identity. The backward pass multiplies the gradient by the same mask: dropped units get no gradient.

### Weight decay as L2 regularization

Add a penalty on the squared weights to the loss:

```
L_total = L_data + (lambda / 2) * ||W||^2
dL_total/dW = dL_data/dW + lambda * W
```

With plain SGD the update becomes

```
W <- W - lr * grad - lr * lambda * W = (1 - lr * lambda) * W - lr * grad
```

Every step shrinks the weights by a constant factor before applying the gradient, hence "weight decay". Smaller weights mean a smoother function: a ReLU network with small weights cannot make sharp turns. Biases and normalization parameters are usually excluded, since shrinking them does not make the function simpler.

### Why Adam needs decoupled weight decay

With Adam, adding `lambda * W` to the gradient is not the same as decay. The penalty gradient goes through Adam's normalization `m_hat / sqrt(v_hat)`:

```
Adam + L2:  g = grad + lambda * W,   W <- W - lr * m_hat(g) / sqrt(v_hat(g))
AdamW:      W <- W - lr * lambda * W - lr * m_hat(grad) / sqrt(v_hat(grad))
```

For a weight with large data gradients, `sqrt(v)` is large and the L2 term barely matters. For a weight with tiny data gradients, the L2 term dominates `g`, Adam normalizes it to a step of size about `lr`, and the weight is driven toward zero fast. So the effective regularization depends on gradient history in an uncontrolled way. Loshchilov and Hutter proposed AdamW: apply the decay directly to the weights, outside the adaptive step. Every weight then shrinks by the same factor `1 - lr * lambda` per step, as with SGD.

The consequence for you: the same number means different things. `weight_decay=0.01` in `torch.optim.Adam` is a strong coupled L2 penalty; in `torch.optim.AdamW` it is a gentle decay of `lr * 0.01` per step. AdamW values like 0.01 to 0.1 are typical for transformers; on tiny problems you may need much larger values, as below.

### Where they go

| | Dropout | Weight decay |
|--|---------|--------------|
| Acts on | activations | weights |
| Train vs eval | different | same |
| Typical value | 0.1 - 0.5 | SGD: 1e-4 - 5e-4, AdamW: 0.01 - 0.1 |
| Typical home | MLP heads, transformer blocks | almost every model |

## Build It

### Step 1: Inverted dropout

```python
def dropout_forward(x, p, training, rng):
    if not training or p == 0.0:
        return x, None
    mask = (rng.random(x.shape) >= p) / (1.0 - p)
    return x * mask, mask
```

On a matrix of ones with `p = 0.5`: about half the entries become 0, survivors become 2, and the mean stays 1.00. In eval mode the input comes back unchanged.

### Step 2: Dropout in backprop

The forward pass stores one mask per hidden layer. The backward pass applies the mask before the ReLU derivative, mirroring the forward order `z -> relu -> dropout`:

```python
for i in reversed(range(n_layers)):
    grads[f"W{i}"] = activations[i].T @ grad
    grads[f"b{i}"] = grad.sum(axis=0)
    if i > 0:
        grad = grad @ params[f"W{i}"].T
        if masks[i - 1] is not None:
            grad = grad * masks[i - 1]
        grad = grad * (activations[i] > 0)
```

### Step 3: Adam with both kinds of decay

One optimizer class handles both, applied only to weight matrices, not biases:

```python
for name in params:
    decays = name.startswith("W")
    grad = grads[name] + (self.l2 * params[name] if decays else 0.0)
    if decays and self.weight_decay:
        params[name] -= self.lr * self.weight_decay * params[name]
    ...regular Adam update with grad...
```

### Step 4: Fight overfitting

80 noisy moons for training, 2000 for testing, 400 epochs of Adam at `lr = 3e-3`:

```
baseline                  | train loss 0.034 acc 0.988 | test loss 1.449 acc 0.835 | ||W||  42.78
dropout 0.5               | train loss 0.172 acc 0.912 | test loss 0.700 acc 0.846 | ||W||  38.47
weight decay 3.0 (AdamW)  | train loss 0.231 acc 0.887 | test loss 0.350 acc 0.859 | ||W||  11.44
dropout 0.5 + decay 1.0   | train loss 0.215 acc 0.887 | test loss 0.398 acc 0.852 | ||W||  12.17
```

The baseline has memorized the training set. Both regularizers give up training accuracy and cut test loss by a factor of 2 to 4, with a small gain in test accuracy. Weight decay shrinks the weight norm by almost 4x. `boundaries.png` shows the baseline's jagged boundary next to the smooth regularized ones. (A decay of 3.0 looks huge, but with AdamW it means multiplying the weights by `1 - 0.003 * 3 = 0.991` per step.)

### Step 5: Same number, different algorithm

```
L2 penalty 1e-2 (Adam + L2)   | test loss 0.317 acc 0.863 | ||W||   4.66
decoupled decay 1e-2 (AdamW)  | test loss 1.399 acc 0.834 | ||W||  41.79
```

With `1e-2`, coupled L2 crushes the weights to a norm of 4.7 while AdamW barely touches them. Neither is wrong; they are different knobs, and hyperparameters do not transfer between them.

## Use It

```python
layer = torch.nn.Dropout(p=0.5)
layer.train()
layer.eval()
```

`nn.Dropout` is inverted dropout: in train mode about half the entries are zero and survivors are 2.0; in eval mode it is the identity. `model.eval()` switches all dropout layers off at once, just like batch norm.

For weight decay, the script runs your optimizer and PyTorch's on the same 20 random gradients:

```
Adam(weight_decay=0.1) == L2 | max diff after 20 steps 1.1e-16
AdamW(weight_decay=0.1)     | max diff after 20 steps 1.1e-16
```

So `torch.optim.Adam(..., weight_decay=w)` adds `w * param` to the gradient (coupled L2) and `torch.optim.AdamW` decays directly. To exclude biases and norm parameters, use parameter groups:

```python
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```

## Exercises

1. **Easy.** Sweep dropout `p` in `{0.1, 0.3, 0.5, 0.7}` and plot test loss against `p`. Where is the sweet spot for this dataset?
2. **Medium.** Implement Monte Carlo dropout: keep dropout on at test time, average the softmax of 30 forward passes, and use the spread across passes as an uncertainty estimate. Is it higher near the decision boundary?
3. **Hard.** Train with SGD + momentum and show that L2 in the gradient and decoupled decay are equivalent up to a rescaling of `lambda` by the learning rate. Then show they stop being equivalent once you add a cosine learning rate schedule.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Overfitting | "Memorizing the training set" | Low training loss with much higher test loss; the model fits noise |
| Dropout | "Randomly turning off neurons" | Zeroing each activation with probability `p` during training only |
| Inverted dropout | "Scaling by keep prob" | Dividing survivors by `1 - p` in training so eval mode needs no change |
| L2 regularization | "Ridge penalty" | Adding `lambda/2 ||W||^2` to the loss; its gradient is `lambda W` |
| Weight decay | "Same as L2" | Shrinking weights by `(1 - lr lambda)` each step; equal to L2 only for plain SGD |
| AdamW | "Adam with fixed weight decay" | Adam with the decay applied to the weights outside the adaptive normalization |

## Further Reading

- [Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)](https://jmlr.org/papers/v15/srivastava14a.html) - the dropout paper with the ensemble interpretation
- [Loshchilov and Hutter, Decoupled Weight Decay Regularization (2017)](https://arxiv.org/abs/1711.05101) - why L2 and weight decay differ for Adam, and AdamW
- [Dive into Deep Learning, Section 5.6: Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) - dropout from scratch and with a framework, with the weight decay section linked from the same chapter
- [PyTorch docs: torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) - the exact update rule
