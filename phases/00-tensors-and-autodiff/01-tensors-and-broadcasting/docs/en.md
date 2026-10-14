# Tensors and Broadcasting

> A tensor is a block of numbers plus a shape. Broadcasting is the rule that lets blocks of different shapes talk to each other without copying.

**Type:** Learn
**Languages:** Python
**Prerequisites:** [Classic ML from Scratch](https://github.com/Fer14/classic-ml-from-scratch) Phase 01 (Linear Algebra), basic NumPy
**Time:** ~60 minutes

## Learning Objectives

- Describe a tensor by its shape, number of dimensions, dtype and strides, and explain why reshape and transpose are usually free
- Reduce a tensor along chosen axes and keep track of the resulting shape
- Apply the three broadcasting rules by hand and implement them as a function
- Replace Python loops with broadcast expressions (standardization, bias addition, pairwise distances)
- Explain why the gradient of a broadcast operation must be summed back to the original shape

## The Problem

Every deep learning model is a long chain of operations on tensors: a batch of images is a `(batch, channels, height, width)` block, a batch of sentences is `(batch, tokens, features)`, the weights of a layer are a matrix and its bias is a vector. If you write those operations as Python loops, a single forward pass over a small batch takes seconds instead of microseconds.

The other failure is silent. You add a bias of shape `(5,)` to activations of shape `(5, 1)` and get a `(5, 5)` matrix instead of an error. The loss still goes down a little, your model is wrong, and nothing crashes. Most bugs in deep learning code are shape bugs, and almost all of them come from not knowing exactly what broadcasting does.

This lesson gives you the mental model you will use in every later lesson: what a tensor is in memory, how axes work, and the precise rules that decide which shapes can be combined.

## The Concept

### A tensor is data plus a shape

| Rank | Example | Typical shape |
|------|---------|---------------|
| 0 | a loss value | `()` |
| 1 | a bias vector | `(out_features,)` |
| 2 | a weight matrix, a batch of feature vectors | `(batch, features)` |
| 3 | a batch of token sequences | `(batch, tokens, features)` |
| 4 | a batch of images | `(batch, channels, height, width)` |

In memory every tensor is one flat buffer. The shape and the **strides** say how to walk it: the stride of an axis is how many bytes you jump to move one step along that axis.

```
flat buffer:  [ 0  1  2  3  4  5  6  7  8  9 10 11 ]   float64, 8 bytes each

reshape(3, 4)          shape (3, 4)  strides (32, 8)
  0  1  2  3           one row down  = jump 4 numbers = 32 bytes
  4  5  6  7           one column    = jump 1 number  =  8 bytes
  8  9 10 11

.T                     shape (4, 3)  strides (8, 32)
                       same buffer, strides swapped, no copy
```

That is why `reshape` and `transpose` are almost free: they create a new **view** with different metadata over the same memory. Writing through a view changes the original. A transposed tensor is no longer **contiguous** (rows are not adjacent in memory), and some operations will make a copy for you.

### Axes and reductions

A reduction (`sum`, `mean`, `max`) along an axis removes that axis. With `keepdims=True` the axis stays with size 1, which is exactly what you need to broadcast the result back.

```
x.shape                    (2, 3, 4)
x.sum(axis=0)              (3, 4)
x.sum(axis=-1)             (2, 3)
x.mean(axis=(1, 2), keepdims=True)   (2, 1, 1)
```

### The broadcasting rules

When two tensors meet in an elementwise operation:

1. Align the shapes from the **right**. Pad the shorter shape with 1s on the left.
2. For each axis, the sizes are compatible if they are **equal** or one of them is **1**.
3. The result takes the larger size; a size-1 axis is virtually repeated, without copying.

```
A:      8 x 1 x 5
B:          7 x 1         ->  padded to 1 x 7 x 1
result: 8 x 7 x 5

A:      5 x 4
B:          5             ->  4 vs 5, neither is 1: error
```

The stride trick makes this free: a broadcast axis gets stride 0, so the same numbers are read again and again.

### Broadcasting backwards

If `y = x + b` with `x` of shape `(5, 3, 4)` and `b` of shape `(1, 4)`, each entry of `b` was used `5 * 3 = 15` times. In the backward pass its gradient is the **sum** of the 15 upstream gradients that touched it. In general:

```
grad_b = upstream gradient summed over every axis that was broadcast
```

You will need this exact function (`unbroadcast`) when you build the vectorized autodiff engine later in this phase.

## Build It

### Step 1: Inspect tensors

```python
def describe(name, array):
    print(f"  {name:<10} shape={str(array.shape):<15} ndim={array.ndim} strides={array.strides} dtype={array.dtype}")

images = rng.normal(size=(8, 3, 32, 32)).astype(np.float32)
describe("images", images)
```

The image batch has strides `(12288, 4096, 128, 4)`: one float32 is 4 bytes, one row of 32 pixels is 128 bytes, one 32x32 channel is 4096 bytes.

### Step 2: Views share memory

```python
flat = np.arange(12.0)
grid = flat.reshape(3, 4)
grid_t = grid.T
grid_t[0, 0] = 100.0
print(flat[0])
print(grid_t.flags["C_CONTIGUOUS"], np.shares_memory(flat, grid_t))
```

Writing to `grid_t` changes `flat`, because all three names point at the same buffer. `grid.T` reports `C_CONTIGUOUS = False`.

### Step 3: Reductions

```python
batch = np.arange(24.0).reshape(2, 3, 4)
batch.sum(axis=0).shape
batch.sum(axis=-1).shape
batch.mean(axis=(1, 2), keepdims=True).shape
```

You get `(3, 4)`, `(2, 3)` and `(2, 1, 1)`.

### Step 4: Broadcasting by hand

```python
def broadcast_shape(shape_a, shape_b):
    ndim = max(len(shape_a), len(shape_b))
    padded_a = (1,) * (ndim - len(shape_a)) + tuple(shape_a)
    padded_b = (1,) * (ndim - len(shape_b)) + tuple(shape_b)
    result = []
    for size_a, size_b in zip(padded_a, padded_b):
        if size_a == size_b or size_b == 1:
            result.append(size_a)
        elif size_a == 1:
            result.append(size_b)
        else:
            raise ValueError(f"cannot broadcast {shape_a} with {shape_b}")
    return tuple(result)
```

The script checks it against `np.broadcast_shapes` on four cases, including the `(5, 4)` with `(5,)` failure.

### Step 5: Broadcasting in practice

Standardizing columns is one line, because `X.mean(axis=0)` has shape `(3,)` and aligns with the last axis of `X` `(100, 3)`:

```python
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
```

A dense layer is `X @ W + b`: the bias `(2,)` is broadcast across all 100 rows.

Pairwise distances between two point sets are the classic example. Insert new axes so the two sets spread along different dimensions:

```python
def pairwise_distances_broadcast(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))
```

`(300, 1, 16) - (1, 200, 16)` broadcasts to `(300, 200, 16)`. On this machine the double loop takes about 200 ms and the broadcast version about 10 ms, with identical results. The price is memory: the intermediate tensor holds `300 * 200 * 16` numbers.

### Step 6: Unbroadcast

```python
def unbroadcast(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(original_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
```

First sum away the leading axes that were added by padding, then sum (keeping the axis) wherever the original size was 1. An upstream gradient of ones with shape `(5, 3, 4)` becomes a bias gradient of shape `(1, 4)` filled with 15.

### Step 7: Batched matmul

`@` treats all but the last two axes as batch axes and broadcasts them:

```python
batch_a = rng.normal(size=(10, 4, 5))
batch_b = rng.normal(size=(5, 6))
batched = batch_a @ batch_b
einsum = np.einsum("bij,jk->bik", batch_a, batch_b)
```

`einsum` spells out the same contraction with index letters. You will see it again in attention.

## Use It

PyTorch tensors follow the same rules, with the same vocabulary:

```python
t = torch.arange(12.0).reshape(3, 4)
t.stride(), t.T.stride()
torch.broadcast_shapes((8, 1, 5), (7, 1))
torch.cdist(torch.from_numpy(A), torch.from_numpy(B))
```

PyTorch strides are counted in elements, not bytes, so you see `(4, 1)` and `(1, 4)`. `torch.cdist` computes the same pairwise distances as your broadcast function. `torch.from_numpy` wraps the NumPy buffer without copying, so changes in one are visible in the other. The next lessons add the one thing NumPy does not have: tensors that remember how they were computed.

## Exercises

1. **Easy.** Predict the result shape (or the error) for `(4, 1, 3) + (2, 1)`, `(6,) * (6, 1)` and `(3, 4) @ (4,)`, then check each one with NumPy.
2. **Medium.** Write `softmax(x, axis)` for a tensor of any rank using only reductions with `keepdims=True` and broadcasting. Subtract the max along the axis first and verify that each slice sums to 1.
3. **Hard.** Implement `broadcast_to(array, shape)` with `np.lib.stride_tricks.as_strided` by setting the stride of every broadcast axis to 0. Check that the result has the right values and that `np.shares_memory` with the input is `True`.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Tensor | "A fancy matrix" | An n-dimensional array: a flat buffer plus a shape, strides and a dtype |
| Shape | "The size" | The tuple of lengths along each axis, for example `(batch, channels, height, width)` |
| Stride | "Low-level detail" | How far to jump in memory to move one step along an axis |
| View | "A copy" | A new tensor object that shares the original buffer with different shape or strides |
| Contiguous | "Normal memory" | Elements laid out in row-major order with no gaps, so the last axis has the smallest stride |
| Broadcasting | "NumPy magic" | Right-aligned shape matching where size-1 axes are virtually repeated with stride 0 |
| Unbroadcast | "Summing the gradient" | Reducing a gradient over broadcast axes so it matches the shape of the input that was broadcast |
| einsum | "Index notation" | A compact way to write products and sums over named axes |

## Further Reading

- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - the official statement of the rules with diagrams
- [Dive into Deep Learning: Data Manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html) - tensors, broadcasting and memory in PyTorch
- [PyTorch: Tensor Views](https://pytorch.org/docs/stable/tensor_view.html) - which PyTorch operations return views and which copy
- [Andrej Karpathy: Building makemore Part 4, Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) - shows broadcasting gradients summed by hand on a real network
