from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def target_function(x):
    return np.sin(3 * x) + 0.3 * x ** 2


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def bump(x, left, right, height, steepness=50.0):
    return height * (sigmoid(steepness * (x - left)) - sigmoid(steepness * (x - right)))


def bump_approximation(x, n_bumps):
    edges = np.linspace(-2.1, 2.1, n_bumps + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    heights = target_function(centers)
    steepness = 10.0 * n_bumps
    return sum(bump(x, left, right, h, steepness) for left, right, h in zip(edges[:-1], edges[1:], heights))


def relu_interpolation(x, knots):
    values = target_function(knots)
    slopes = np.diff(values) / np.diff(knots)
    output = values[0] + slopes[0] * (x - knots[0])
    for index in range(1, len(slopes)):
        output = output + (slopes[index] - slopes[index - 1]) * np.maximum(0.0, x - knots[index])
    return output


def train_one_hidden_layer(x, y, hidden_units, steps, learning_rate, seed):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 2.0, size=(1, hidden_units))
    b1 = rng.uniform(-4.0, 4.0, size=hidden_units)
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_units), size=(hidden_units, 1))
    b2 = np.zeros(1)
    m = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    v = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    for t in range(1, steps + 1):
        z = x @ W1 + b1
        h = np.tanh(z)
        prediction = h @ W2 + b2
        dpred = 2 * (prediction - y) / len(y)
        dW2 = h.T @ dpred
        db2 = dpred.sum(axis=0)
        dz = (dpred @ W2.T) * (1 - h ** 2)
        dW1 = x.T @ dz
        db1 = dz.sum(axis=0)
        for index, (param, grad) in enumerate(zip((W1, b1, W2, b2), (dW1, db1, dW2, db2))):
            m[index] = 0.9 * m[index] + 0.1 * grad
            v[index] = 0.999 * v[index] + 0.001 * grad ** 2
            m_hat = m[index] / (1 - 0.9 ** t)
            v_hat = v[index] / (1 - 0.999 ** t)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    return lambda inputs: np.tanh(inputs @ W1 + b1) @ W2 + b2


def main():
    x = np.linspace(-2, 2, 400)
    y = target_function(x)

    print("=== Step 1: two sigmoids make a bump ===")
    sample = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    print(f"  x = {sample}")
    print(f"  bump(-0.5, 0.5, height 1) = {np.round(bump(sample, -0.5, 0.5, 1.0), 3)}")

    print("\n=== Step 2: many bumps approximate any curve ===")
    bump_results = {}
    for n in [4, 16, 64]:
        approx = bump_approximation(x, n)
        bump_results[n] = approx
        print(f"  {n:2d} bumps ({2 * n:3d} sigmoid units) max error = {np.max(np.abs(approx - y)):.4f}")

    print("\n=== Step 3: ReLUs build exact piecewise-linear interpolants ===")
    for n_knots in [5, 9, 33]:
        knots = np.linspace(-2, 2, n_knots)
        approx = relu_interpolation(x, knots)
        print(f"  {n_knots - 2:2d} ReLU units: max error = {np.max(np.abs(approx - y)):.4f}")

    print("\n=== Step 4: gradient descent finds its own approximation ===")
    x_col, y_col = x[:, None], y[:, None]
    learned = {}
    for hidden in [2, 8, 32]:
        model = train_one_hidden_layer(x_col, y_col, hidden, steps=3000, learning_rate=0.01, seed=0)
        prediction = model(x_col)
        learned[hidden] = prediction.ravel()
        print(f"  {hidden:2d} tanh units: MSE = {np.mean((prediction - y_col) ** 2):.5f}")

    print("\n=== Step 5: approximation is not extrapolation ===")
    x_far = np.array([[3.0], [4.0]])
    far_prediction = train_one_hidden_layer(x_col, y_col, 32, steps=3000, learning_rate=0.01, seed=0)(x_far).ravel()
    print(f"  x = {x_far.ravel()} true = {np.round(target_function(x_far.ravel()), 3)} predicted = {np.round(far_prediction, 3)}")

    plot_path = Path(__file__).parent / "approximations.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, y, "k", linewidth=2, label="target")
    for n, approx in bump_results.items():
        axes[0].plot(x, approx, label=f"{n} bumps")
    axes[0].set_title("hand-built sigmoid bumps")
    axes[1].plot(x, y, "k", linewidth=2, label="target")
    for hidden, prediction in learned.items():
        axes[1].plot(x, prediction, label=f"{hidden} units")
    axes[1].set_title("trained one-hidden-layer nets")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"\n  saved {plot_path.name}")

    print("\n=== Use It: the same experiment in torch ===")
    torch.manual_seed(0)
    xt = torch.tensor(x_col, dtype=torch.float32)
    yt = torch.tensor(y_col, dtype=torch.float32)
    for hidden in [2, 8, 32]:
        net = torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        for _ in range(3000):
            loss = torch.nn.functional.mse_loss(net(xt), yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  {hidden:2d} tanh units: MSE = {loss.item():.5f}")


if __name__ == "__main__":
    main()
