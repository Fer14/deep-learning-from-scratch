import time

import numpy as np
import torch


def describe(name, array):
    print(f"  {name:<10} shape={str(array.shape):<15} ndim={array.ndim} strides={array.strides} dtype={array.dtype}")


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


def unbroadcast(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(original_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def pairwise_distances_loop(A, B):
    out = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            out[i, j] = np.sqrt(np.sum((A[i] - B[j]) ** 2))
    return out


def pairwise_distances_broadcast(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def main():
    rng = np.random.default_rng(0)

    print("=== Step 1: tensors are n-dimensional arrays ===")
    scalar = np.array(3.0)
    vector = np.arange(4.0)
    matrix = np.arange(12.0).reshape(3, 4)
    images = rng.normal(size=(8, 3, 32, 32)).astype(np.float32)
    for name, array in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("images", images)]:
        describe(name, array)

    print("\n=== Step 2: reshape and transpose are views over the same memory ===")
    flat = np.arange(12.0)
    grid = flat.reshape(3, 4)
    grid_t = grid.T
    grid_t[0, 0] = 100.0
    describe("flat", flat)
    describe("grid", grid)
    describe("grid.T", grid_t)
    print(f"  flat[0] after writing through grid.T: {flat[0]}")
    print(f"  grid.T contiguous? {grid_t.flags['C_CONTIGUOUS']}  shares memory with flat? {np.shares_memory(flat, grid_t)}")

    print("\n=== Step 3: reductions collapse an axis ===")
    batch = np.arange(24.0).reshape(2, 3, 4)
    print(f"  batch.sum(axis=0).shape = {batch.sum(axis=0).shape}")
    print(f"  batch.sum(axis=-1).shape = {batch.sum(axis=-1).shape}")
    print(f"  batch.mean(axis=(1, 2), keepdims=True).shape = {batch.mean(axis=(1, 2), keepdims=True).shape}")

    print("\n=== Step 4: broadcasting rules, implemented by hand ===")
    cases = [((3, 4), (4,)), ((8, 1, 5), (7, 1)), ((2, 3, 4), (3, 1)), ((5, 4), (5,))]
    for shape_a, shape_b in cases:
        try:
            mine = broadcast_shape(shape_a, shape_b)
        except ValueError:
            mine = "error"
        try:
            reference = np.broadcast_shapes(shape_a, shape_b)
        except ValueError:
            reference = "error"
        print(f"  {str(shape_a):<10} + {str(shape_b):<7} -> mine {str(mine):<10} numpy {reference}")

    print("\n=== Step 5: broadcasting in practice ===")
    X = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    print(f"  standardized column means = {np.round(X_standardized.mean(axis=0), 6) + 0.0}")
    print(f"  standardized column stds  = {np.round(X_standardized.std(axis=0), 6)}")
    W = rng.normal(size=(3, 2))
    b = np.array([0.5, -0.5])
    print(f"  X @ W + b shape = {(X @ W + b).shape}  (b of shape {b.shape} is broadcast over 100 rows)")

    A = rng.normal(size=(300, 16))
    B = rng.normal(size=(200, 16))
    start = time.perf_counter()
    slow = pairwise_distances_loop(A, B)
    loop_time = time.perf_counter() - start
    start = time.perf_counter()
    fast = pairwise_distances_broadcast(A, B)
    broadcast_time = time.perf_counter() - start
    print(f"  pairwise distances 300x200: loop {loop_time * 1000:.1f} ms, broadcast {broadcast_time * 1000:.2f} ms, max diff {np.abs(slow - fast).max():.2e}")

    print("\n=== Step 6: broadcasting backwards (unbroadcast) ===")
    bias = np.zeros((1, 4))
    upstream = np.ones((5, 3, 4))
    grad_bias = unbroadcast(upstream, bias.shape)
    print(f"  upstream grad {upstream.shape} -> bias grad {grad_bias.shape}, values {grad_bias.ravel()}")

    print("\n=== Step 7: batched matrix multiplication ===")
    batch_a = rng.normal(size=(10, 4, 5))
    batch_b = rng.normal(size=(5, 6))
    batched = batch_a @ batch_b
    print(f"  (10, 4, 5) @ (5, 6) -> {batched.shape}")
    einsum = np.einsum("bij,jk->bik", batch_a, batch_b)
    print(f"  einsum matches matmul: {np.allclose(batched, einsum)}")

    print("\n=== Use It: the same operations in PyTorch ===")
    t = torch.arange(12.0).reshape(3, 4)
    print(f"  torch shape {tuple(t.shape)}, stride {t.stride()}, transposed stride {t.T.stride()}")
    print(f"  torch.broadcast_shapes((8, 1, 5), (7, 1)) = {tuple(torch.broadcast_shapes((8, 1, 5), (7, 1)))}")
    A_t, B_t = torch.from_numpy(A), torch.from_numpy(B)
    torch_dist = torch.cdist(A_t, B_t).numpy()
    print(f"  torch.cdist matches broadcast version: {np.allclose(torch_dist, fast)}")
    print(f"  torch.from_numpy shares memory: {np.shares_memory(torch.from_numpy(A).numpy(), A)}")


if __name__ == "__main__":
    main()
