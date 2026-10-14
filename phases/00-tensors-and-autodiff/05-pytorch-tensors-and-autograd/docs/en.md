# PyTorch Tensors and Autograd

> You already built autograd twice. PyTorch is the industrial version: same graph, same backward sweep, plus the controls you need to train real models.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Autograd from Scratch, A Vectorized Autodiff Engine
**Time:** ~60 minutes

## Learning Objectives

- Create PyTorch tensors and predict their dtype and device, and move data to and from NumPy
- Use `requires_grad`, `backward()` and `.grad`, and explain why gradients accumulate across calls
- Call `backward()` on a non-scalar output and connect it to the vector-Jacobian product
- Compute higher-order derivatives with `torch.autograd.grad(create_graph=True)`
- Control the graph with `torch.no_grad()`, `detach()` and `retain_graph`
- Write a custom `torch.autograd.Function` and verify it with `gradcheck`

## The Problem

Your NumPy engine works and matches PyTorch to sixteen decimals, but it is missing everything around the core: it only runs on the CPU, supports a dozen operations, cannot turn the graph off for evaluation, and has no way to plug in a hand-written backward for an operation you care about. Every later phase of this course uses PyTorch for that reason.

The risk of switching is treating PyTorch as magic. Most confusing PyTorch bugs are autograd bugs in disguise: a loss that grows because gradients were never zeroed, memory that explodes because evaluation code was building a graph, a model that does not learn because a tensor was detached in the middle. Each one is obvious once you map PyTorch's API onto the engine you already wrote.

## The Concept

### The mapping

| Your engine | PyTorch |
|-------------|---------|
| `Tensor(data)` | `torch.tensor(data, requires_grad=True)` |
| `node.parents`, `node.backward_step` | `t.grad_fn`, `t.grad_fn.next_functions` |
| `loss.backward()` | `loss.backward()` |
| `p.grad` | `p.grad` |
| zeroing grads inside `backward()` | **not done**: you call `optimizer.zero_grad()` or set `p.grad = None` |
| creating a Tensor with no parents | `t.detach()` |
| no equivalent | `torch.no_grad()`, `retain_graph`, `create_graph`, custom `Function` |

### Leaves and non-leaves

Tensors you create directly are **leaves**. Tensors produced by operations have a `grad_fn` and are not leaves. By default PyTorch only stores `.grad` on leaves that have `requires_grad=True`: those are your parameters.

### Why gradients accumulate

PyTorch adds into `.grad` instead of overwriting it. This is a feature: you can call `backward()` on several small batches and get the gradient of their sum, which is how **gradient accumulation** simulates a large batch on limited memory. The cost is that you must reset `.grad` at every training step.

### Backward on a non-scalar: the vector-Jacobian product

For a scalar loss, `backward()` computes the gradient. For a vector output `y = f(x)` the full derivative is a Jacobian matrix `J` with `J[i, j] = dy_i / dx_j`. Reverse mode never builds `J`. It computes

```
v^T J        for a vector v you supply
```

which is exactly what the backward sweep propagates: `v` plays the role of `out.grad`. That is why `y.backward()` on a vector requires a `gradient=` argument, and why `loss.backward()` on a scalar is just the case `v = 1`.

### Turning the graph off

Building the graph costs memory, since every saved intermediate value is kept until `backward()`. When you only need values, such as evaluating a model or updating weights in place, wrap the code in `torch.no_grad()`. `detach()` does the same for a single tensor: it returns a view of the same data with no history, so gradients stop flowing there.

After `backward()`, PyTorch frees the saved intermediate values. A second `backward()` on the same graph fails unless you passed `retain_graph=True`.

### Higher-order derivatives

If you pass `create_graph=True`, the backward pass itself is recorded as a graph, so you can differentiate the gradient again. This powers Hessian-vector products, some meta-learning methods and gradient penalties (you will meet one in Phase 07, GANs).

## Build It

### Step 1: Tensors

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ints = torch.arange(6)
(ints + 0.5).dtype
torch.from_numpy(np.ones(2)).dtype
```

Python floats become `float32`, integers become `int64`, mixing them promotes to `float32`, and NumPy arrays keep their `float64`. Mixing `float32` model weights with `float64` data is a common source of errors, so convert data explicitly with `.float()`.

### Step 2: The first gradient

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
x.grad
```

`y.grad_fn` is `AddBackward0`, `x.is_leaf` is `True`, and `x.grad` is 8.

### Step 3: Accumulation

```python
x.grad = None
for call in range(3):
    y = x ** 2 + 2 * x
    y.backward()
```

`x.grad` goes 8, 16, 24. Setting it to `None` (what `optimizer.zero_grad()` does by default) resets it.

### Step 4: Vector-Jacobian products

```python
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v ** 2
weights = torch.tensor([1.0, 0.1, 0.01])
out.backward(gradient=weights)
```

`v.grad` is `2v * weights = [2, 0.4, 0.06]`. The script also builds the full Jacobian with `torch.autograd.functional.jacobian` and shows that `weights @ J` gives the same vector, without reverse mode ever forming `J`.

### Step 5: Second derivatives

```python
z = torch.tensor(2.0, requires_grad=True)
f = z ** 3
(first,) = torch.autograd.grad(f, z, create_graph=True)
(second,) = torch.autograd.grad(first, z)
```

`first` is `3z^2 = 12` and `second` is `6z = 12`. `torch.autograd.grad` returns gradients instead of writing them into `.grad`.

### Step 6: no_grad, detach and retain_graph

```python
with torch.no_grad():
    prediction = w * 2
detached = w.detach()
loss = (w ** 2).sum()
loss.backward(retain_graph=True)
loss.backward()
```

`prediction` has `requires_grad=False`. `detached` shares storage with `w` (same `data_ptr()`) but has no history. Two backward calls succeed with `retain_graph=True`; without it the second call raises "Trying to backward through the graph a second time".

### Step 7: A custom autograd.Function

When you want a hand-written backward (for speed, numerical stability or an operation PyTorch lacks), subclass `torch.autograd.Function`:

```python
class ScaledSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return torch.log1p(torch.exp(beta * x)) / beta

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * torch.sigmoid(ctx.beta * x), None
```

This is your `backward_step` closure with an explicit interface: `ctx` stores what the backward needs, and `backward` returns one gradient per input of `forward` (`None` for the non-tensor `beta`). `torch.autograd.gradcheck` compares it with finite differences in float64 and returns `True`.

### Step 8: Training with raw autograd

```python
for step in range(200):
    loss = ((X @ weight + bias - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        weight -= 0.1 * weight.grad
        bias -= 0.1 * bias.grad
    weight.grad = None
    bias.grad = None
```

The update is inside `no_grad` because it must not become part of the graph. The model recovers `w = [1.994, -1.003, 0.495]` and `b = 0.300` against the true `[2, -1, 0.5]` and `0.3`.

## Use It

`torch.optim` packages the update and the reset:

```python
optimizer = torch.optim.SGD([weight, bias], lr=0.1)
for step in range(200):
    optimizer.zero_grad()
    loss = ((X @ weight + bias - y) ** 2).mean()
    loss.backward()
    optimizer.step()
```

The result is identical to the manual loop. This four-line pattern, zero grad, forward, backward, step, is the heart of every training loop in the rest of the course. Phase 02 (Training Deep Nets) opens up `optimizer.step()` and builds momentum and Adam by hand.

## Exercises

1. **Easy.** Remove `weight.grad = None` from the manual loop and run it. Describe what happens to the loss and explain it using the accumulation rule.
2. **Medium.** Use `torch.autograd.grad` with `create_graph=True` to compute a Hessian-vector product `H v` for `f(w) = sum(w^4)` without forming `H`, and check it against the analytic `12 w^2 * v`.
3. **Hard.** Write a custom `autograd.Function` for a numerically stable `log_sum_exp` along the last axis, with backward `grad_output[..., None] * softmax(x)`. Verify it with `gradcheck` and show that a naive `torch.log(torch.exp(x).sum(-1))` overflows for inputs around 1000 while yours does not.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| requires_grad | "Track this tensor" | A flag telling autograd to record operations on this tensor and compute its gradient |
| Leaf tensor | "A parameter" | A tensor created directly, not by an operation; `.grad` is populated only on leaves by default |
| grad_fn | "The backward function" | The node that knows how to send gradients from this tensor to its inputs |
| Vector-Jacobian product | "Backward with a gradient argument" | `v^T J`, what one reverse sweep computes for a vector output |
| no_grad | "Inference mode" | A context in which operations are not recorded, saving memory and time |
| detach | "Stop gradient" | Returns a tensor sharing the same data but with no history |
| retain_graph | "Keep the graph" | Prevents PyTorch from freeing saved values after `backward()`, so you can call it again |
| autograd.Function | "A custom op" | A user-defined forward and backward pair that plugs into autograd |

## Further Reading

- [PyTorch: A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) - official walkthrough of the graph, vector-Jacobian products and freezing parameters
- [PyTorch: Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) - saved tensors, in-place operations, no_grad versus inference_mode
- [PyTorch: Extending torch.autograd](https://pytorch.org/docs/stable/notes/extending.html) - the full rules for writing custom `Function`s
- [Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library (2019)](https://arxiv.org/abs/1912.01703) - the design paper, including define-by-run autograd
