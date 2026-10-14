import numpy as np
import torch


def step(z):
    return (z >= 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def xor_by_hand(X):
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([-0.5, -1.5])
    W2 = np.array([1.0, -1.0])
    b2 = -0.5
    hidden = step(X @ W1 + b1)
    output = step(hidden @ W2 + b2)
    return hidden, output


def init_mlp(layer_sizes, seed):
    rng = np.random.default_rng(seed)
    params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.normal(0.0, 1.0, size=(fan_in, fan_out))
        b = np.zeros(fan_out)
        params.append((W, b))
    return params


def mlp_forward(X, params):
    activation = X
    shapes = [activation.shape]
    for index, (W, b) in enumerate(params):
        z = activation @ W + b
        activation = sigmoid(z) if index == len(params) - 1 else np.tanh(z)
        shapes.append(activation.shape)
    return activation, shapes


def count_parameters(layer_sizes):
    return sum(fan_in * fan_out + fan_out for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]))


def collapse_linear_layers(seed):
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(size=(3, 5)), rng.normal(size=5)
    W2, b2 = rng.normal(size=(5, 2)), rng.normal(size=2)
    X = rng.normal(size=(4, 3))
    two_layers = (X @ W1 + b1) @ W2 + b2
    W_combined = W1 @ W2
    b_combined = b1 @ W2 + b2
    one_layer = X @ W_combined + b_combined
    return np.max(np.abs(two_layers - one_layer))


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

    print("=== Step 1: XOR with two hand-wired layers ===")
    hidden, output = xor_by_hand(X)
    for x, h, o in zip(X.astype(int).tolist(), hidden.astype(int).tolist(), output.astype(int).tolist()):
        print(f"  x={x} hidden(OR, AND)={h} output={o}")

    print("\n=== Step 2: stacking linear layers without a nonlinearity collapses ===")
    print(f"  max difference between 2 linear layers and 1 merged layer: {collapse_linear_layers(0):.2e}")

    print("\n=== Step 3: a forward pass through a 2-8-8-1 MLP ===")
    layer_sizes = [2, 8, 8, 1]
    params = init_mlp(layer_sizes, seed=0)
    probabilities, shapes = mlp_forward(X, params)
    print(f"  activation shapes per layer: {shapes}")
    print(f"  untrained outputs: {np.round(probabilities.ravel(), 3)}")

    print("\n=== Step 4: counting parameters ===")
    for sizes in [[2, 8, 8, 1], [784, 128, 10], [784, 512, 512, 10]]:
        print(f"  {sizes}: {count_parameters(sizes):,} parameters")

    print("\n=== Use It: the same architecture in torch ===")
    torch.manual_seed(0)
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(),
        torch.nn.Linear(8, 8), torch.nn.Tanh(),
        torch.nn.Linear(8, 1), torch.nn.Sigmoid(),
    )
    with torch.no_grad():
        for layer, (W, b) in zip([m for m in torch_mlp if isinstance(m, torch.nn.Linear)], params):
            layer.weight.copy_(torch.tensor(W.T, dtype=torch.float64))
            layer.bias.copy_(torch.tensor(b, dtype=torch.float64))
        torch_out = torch_mlp(torch.tensor(X, dtype=torch.float32)).numpy().ravel()
    torch_params = sum(p.numel() for p in torch_mlp.parameters())
    print(f"  torch parameter count: {torch_params}")
    print(f"  max |numpy - torch| output difference: {np.max(np.abs(torch_out - probabilities.ravel())):.2e}")


if __name__ == "__main__":
    main()
