# Backpropagation by Hand

> Backprop is the chain rule, applied right to left, reusing everything the forward pass already computed.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 00 (Tensors and Autodiff), Loss Functions for Networks
**Time:** ~60 minutes

## Learning Objectives

- Apply the chain rule to a single neuron and read each factor as a local derivative
- Derive the matrix-form backward pass of a two-layer network: `dW = X^T dZ`, `dX = dZ W^T`
- Verify hand-written gradients with a finite-difference gradient check and a relative error metric
- Confirm that `torch.autograd` produces exactly the gradients you derived

## The Problem

In Phase 00 you built an autograd engine that computes gradients for you. In practice you will always use one. So why derive gradients by hand?

Because when training breaks, the bug is almost always in the backward pass or in something that affects it: a shape that broadcasts the wrong way, a forgotten mean over the batch, an activation whose derivative is zero exactly where your data lives. If you cannot write the backward pass of a two-layer network on paper, you cannot reason about vanishing gradients, about why `W^T` shows up everywhere, or about what a custom `autograd.Function` needs to return. This lesson does it once, carefully, for a network small enough to check every number.

## The Concept

### The chain rule on one neuron

Take `z = w x + b`, `a = tanh(z)`, `L = (a - t)^2`. The loss depends on `w` only through `z` and `a`:

```
dL/dw = dL/da * da/dz * dz/dw
      = 2(a - t) * (1 - a^2) * x
```

```
 w ---.
       \  z = w x + b     a = tanh(z)      L = (a - t)^2
 x -----( * , + ) ------->( tanh ) ------->( loss )
       /
 b ---'
          <-- dz/dw = x   <-- 1 - a^2      <-- 2(a - t)
```

Forward goes left to right and stores `z` and `a`. Backward goes right to left, multiplying by one local derivative per node. Notice that the local derivative of `tanh` uses `a`, which you already computed in the forward pass. That is why frameworks keep activations in memory.

### The two-layer network

```
X  (n, 3)
z1 = X W1 + b1          (n, 4)
h1 = tanh(z1)           (n, 4)
z2 = h1 W2 + b2         (n, 1)
p  = sigmoid(z2)        (n, 1)
L  = BCE(p, y)          scalar
```

Apply the chain rule from the loss back to `W1`:

```
dz2 = (p - y) / n                    (n, 1)   from the BCE + sigmoid result
dW2 = h1^T dz2                       (4, 1)
db2 = sum over rows of dz2           (1,)
dh1 = dz2 W2^T                       (n, 4)
dz1 = dh1 * (1 - h1^2)               (n, 4)   elementwise: tanh'
dW1 = X^T dz1                        (3, 4)
db1 = sum over rows of dz1           (4,)
```

### The two rules that cover every linear layer

For `Z = X W + b`, given the upstream gradient `dZ`:

```
dW = X^T dZ        same shape as W
dX = dZ  W^T       same shape as X
db = sum(dZ, axis=0)
```

You do not need to memorize them. Use shapes: `dW` must have the shape of `W`, and the only product of `X` and `dZ` that gives that shape is `X^T dZ`. The bias was broadcast over the batch in the forward pass, so its gradient sums over the batch in the backward pass. Broadcasting forward means summing backward.

### Gradient checking

A finite difference approximates any partial derivative:

```
dL/dtheta_i ~ ( L(theta_i + eps) - L(theta_i - eps) ) / (2 eps)
```

It costs two forward passes per parameter, far too slow for training but perfect for testing. Compare with the relative error

```
||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)
```

In float64 with `eps = 1e-6`, a correct gradient gives around `1e-9` or smaller. Anything above `1e-5` means a bug.

## Build It

### Step 1: The scalar chain rule

```python
x, w, b, t = 2.0, -0.5, 0.3, 1.0
z = w * x + b
a = np.tanh(z)
loss = (a - t) ** 2
dloss_da = 2 * (a - t)
da_dz = 1 - a ** 2
dz_dw = x
dloss_dw = dloss_da * da_dz * dz_dw
```

```
  forward:  z = -0.7000, a = -0.6044, loss = 2.5740
  backward: dL/da = -3.2087, da/dz = 0.6347, dz/dw = 2.0000
  dL/dw = product = -4.0734
```

The gradient is negative, so increasing `w` lowers the loss: it pushes `a` toward the target 1.

### Step 2: Forward pass with a cache

```python
def forward(X, y, params):
    z1 = X @ params["W1"] + params["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ params["W2"] + params["b2"]
    p = sigmoid(z2)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    cache = {"X": X, "y": y, "z1": z1, "h1": h1, "z2": z2, "p": p}
    return loss, cache
```

The data is 8 random points in 3D with a nonlinear label `x1 + x2 * x3 > 0`.

### Step 3: Backward pass

```python
def backward(cache, params):
    X, y, h1, p = cache["X"], cache["y"], cache["h1"], cache["p"]
    n = X.shape[0]
    dz2 = (p - y) / n
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ params["W2"].T
    dz1 = dh1 * (1 - h1 ** 2)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

Each line is one row of the derivation above. Every gradient has the shape of its parameter.

### Step 4: Gradient check

```python
def numerical_gradients(X, y, params, eps=1e-6):
    grads = {}
    for name, value in params.items():
        grad = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + eps
            plus, _ = forward(X, y, params)
            value[index] = original - eps
            minus, _ = forward(X, y, params)
            value[index] = original
            grad[index] = (plus - minus) / (2 * eps)
        grads[name] = grad
    return grads
```

```
  W1 relative error = 1.74e-10
  b1 relative error = 9.66e-10
  W2 relative error = 2.14e-10
  b2 relative error = 9.17e-11
```

### Step 5: Learn

Use the gradients for plain gradient descent with learning rate 1.0. The loss goes from 0.75 to 0.002 in 300 steps and all 8 training points are classified correctly. You have trained a neural network with gradients you derived yourself.

## Use It

```python
torch_params = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
h1 = torch.tanh(Xt @ torch_params["W1"] + torch_params["b1"])
logits = h1 @ torch_params["W2"] + torch_params["b2"]
loss = F.binary_cross_entropy_with_logits(logits, yt)
loss.backward()
```

The `.grad` of every tensor matches your hand-derived gradients to about `1e-17`. Autograd records the same graph you drew and runs the same local rules backward; it just does it for any graph. When you later need a custom operation, `torch.autograd.Function` asks you to write exactly the `backward` you wrote here, and `torch.autograd.gradcheck` runs exactly the check from Step 4.

## Exercises

1. **Easy.** Replace `tanh` with ReLU in the hidden layer. Which single line of `backward` changes, and how? Rerun the gradient check.
2. **Medium.** Add a third layer (3-4-4-1). Write its backward pass by extending the pattern, and confirm with the gradient check that every relative error is below `1e-7`.
3. **Hard.** Introduce a deliberate bug: forget the `/ n` in `dz2`. Run the gradient check and look at the relative errors. Then try a subtler bug, `db1 = dz1.mean(axis=0)`, and explain why only `b1` fails. Write down what each error pattern tells you about where the bug is.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Backpropagation | "How neural nets learn" | Reverse-mode application of the chain rule that computes all parameter gradients in one backward pass |
| Chain rule | "Multiply the derivatives" | `dL/dx = dL/dy * dy/dx`, summed over every path from `x` to `L` |
| Local derivative | "The node's gradient" | The derivative of one operation's output with respect to its input, independent of the rest of the graph |
| Upstream gradient | "The incoming grad" | `dL/d(output)` of a node, passed to it from the nodes closer to the loss |
| Activation cache | "Saved tensors" | Forward-pass values kept in memory because the backward pass needs them |
| Gradient check | "Numerical check" | Comparing analytic gradients with finite differences to catch backward bugs |
| Relative error | "How close is close" | `||a - b|| / (||a|| + ||b||)`, scale-free and robust to tiny gradients |

## Further Reading

- [Rumelhart, Hinton and Williams (1986), Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) - the paper that popularized backpropagation
- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - the chain rule on a tiny network, visually
- [Karpathy: Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) - the practical failure modes that hand derivation protects you from
- [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) - gates, local gradients and the shape trick for matrix gradients
