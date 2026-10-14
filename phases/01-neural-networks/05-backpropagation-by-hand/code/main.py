import numpy as np
import torch


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params(seed):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.5, size=(3, 4)),
        "b1": np.zeros(4),
        "W2": rng.normal(0.0, 0.5, size=(4, 1)),
        "b2": np.zeros(1),
    }


def forward(X, y, params):
    z1 = X @ params["W1"] + params["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ params["W2"] + params["b2"]
    p = sigmoid(z2)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    cache = {"X": X, "y": y, "z1": z1, "h1": h1, "z2": z2, "p": p}
    return loss, cache


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


def relative_error(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def scalar_chain_rule_demo():
    x, w, b, t = 2.0, -0.5, 0.3, 1.0
    z = w * x + b
    a = np.tanh(z)
    loss = (a - t) ** 2
    dloss_da = 2 * (a - t)
    da_dz = 1 - a ** 2
    dz_dw = x
    dloss_dw = dloss_da * da_dz * dz_dw
    return z, a, loss, dloss_da, da_dz, dz_dw, dloss_dw


def main():
    print("=== Step 1: the chain rule on one neuron ===")
    z, a, loss, dloss_da, da_dz, dz_dw, dloss_dw = scalar_chain_rule_demo()
    print(f"  forward:  z = w*x + b = {z:.4f}, a = tanh(z) = {a:.4f}, loss = (a - 1)^2 = {loss:.4f}")
    print(f"  backward: dL/da = {dloss_da:.4f}, da/dz = {da_dz:.4f}, dz/dw = {dz_dw:.4f}")
    print(f"  dL/dw = product = {dloss_dw:.4f}")

    print("\n=== Step 2: forward pass of a 3-4-1 network, caching intermediates ===")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    y = (X[:, :1] + X[:, 1:2] * X[:, 2:3] > 0).astype(float)
    params = init_params(seed=0)
    loss, cache = forward(X, y, params)
    for key in ["z1", "h1", "z2", "p"]:
        print(f"  {key:2s} shape {cache[key].shape}")
    print(f"  loss = {loss:.6f}")

    print("\n=== Step 3: backward pass by hand ===")
    grads = backward(cache, params)
    for name, grad in grads.items():
        print(f"  d{name:2s} shape {grad.shape} norm {np.linalg.norm(grad):.6f}")

    print("\n=== Step 4: gradient check against finite differences ===")
    numeric = numerical_gradients(X, y, params)
    for name in params:
        print(f"  {name:2s} relative error = {relative_error(grads[name], numeric[name]):.2e}")

    print("\n=== Step 5: use the gradients to learn ===")
    for step_index in range(301):
        loss, cache = forward(X, y, params)
        grads = backward(cache, params)
        for name in params:
            params[name] -= 1.0 * grads[name]
        if step_index % 100 == 0:
            accuracy = np.mean((cache["p"] > 0.5) == y)
            print(f"  step {step_index:3d} loss {loss:.4f} accuracy {accuracy:.2f}")

    print("\n=== Use It: torch.autograd computes the same gradients ===")
    params = init_params(seed=0)
    _, cache = forward(X, y, params)
    grads = backward(cache, params)
    torch_params = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    Xt, yt = torch.tensor(X), torch.tensor(y)
    h1 = torch.tanh(Xt @ torch_params["W1"] + torch_params["b1"])
    logits = h1 @ torch_params["W2"] + torch_params["b2"]
    torch_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
    torch_loss.backward()
    for name in params:
        diff = np.max(np.abs(torch_params[name].grad.numpy() - grads[name]))
        print(f"  {name:2s} max |autograd - by hand| = {diff:.2e}")


if __name__ == "__main__":
    main()
