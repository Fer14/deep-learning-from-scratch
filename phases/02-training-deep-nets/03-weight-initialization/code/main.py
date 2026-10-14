import numpy as np
import torch
from sklearn.datasets import load_digits

WIDTH = 256
DEPTH = 10


def small_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, 0.01, size=(fan_in, fan_out))


def unit_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, 1.0, size=(fan_in, fan_out))


def xavier_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))


def he_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


INITS = {
    "N(0, 0.01)": small_normal,
    "N(0, 1)": unit_normal,
    "Xavier": xavier_normal,
    "He": he_normal,
}

ACTIVATIONS = {
    "tanh": (np.tanh, lambda z, a: 1.0 - a**2),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z, a: (z > 0).astype(z.dtype)),
}


def activation_and_gradient_stats(init, activation_name, rng):
    activation, derivative = ACTIVATIONS[activation_name]
    weights = [init(WIDTH, WIDTH, rng) for _ in range(DEPTH)]
    hidden = rng.normal(size=(256, WIDTH))
    pre_activations, outputs, forward_stds = [], [], []
    for W in weights:
        z = hidden @ W
        hidden = activation(z)
        pre_activations.append(z)
        outputs.append(hidden)
        forward_stds.append(hidden.std())
    grad = rng.normal(size=hidden.shape)
    backward_stds = []
    for W, z, a in zip(reversed(weights), reversed(pre_activations), reversed(outputs)):
        grad = (grad * derivative(z, a)) @ W.T
        backward_stds.append(grad.std())
    return forward_stds, backward_stds[::-1]


def load_data(seed):
    digits = load_digits()
    X = (digits.data - digits.data.mean(axis=0)) / (digits.data.std(axis=0) + 1e-8)
    y = digits.target
    order = np.random.default_rng(seed).permutation(len(y))
    X, y = X[order], y[order]
    return X[:1400], y[:1400], X[1400:], y[1400:]


def init_mlp(sizes, init, rng):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"W{i}"] = init(fan_in, fan_out, rng)
        params[f"b{i}"] = np.zeros(fan_out)
    return params


def forward(params, X):
    n_layers = len(params) // 2
    activations = [X]
    hidden = X
    for i in range(n_layers):
        z = hidden @ params[f"W{i}"] + params[f"b{i}"]
        hidden = np.maximum(z, 0.0) if i < n_layers - 1 else z
        activations.append(hidden)
    return activations


def loss_and_grads(params, X, y):
    n_layers = len(params) // 2
    activations = forward(params, X)
    logits = activations[-1] - activations[-1].max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    grads = {}
    for i in reversed(range(n_layers)):
        grads[f"W{i}"] = activations[i].T @ grad
        grads[f"b{i}"] = grad.sum(axis=0)
        if i > 0:
            grad = (grad @ params[f"W{i}"].T) * (activations[i] > 0)
    return loss, grads


def train(params, X, y, epochs, lr, seed, batch_size=32, beta=0.9):
    rng = np.random.default_rng(seed)
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            _, grads = loss_and_grads(params, X[batch], y[batch])
            for name in params:
                velocity[name] = beta * velocity[name] + grads[name]
                params[name] -= lr * velocity[name]
    return params


def evaluate(params, X, y):
    with np.errstate(all="ignore"):
        loss = loss_and_grads(params, X, y)[0]
        acc = np.mean(forward(params, X)[-1].argmax(axis=1) == y)
    return loss, acc


def format_stds(values):
    return " ".join(f"{v:9.2e}" for v in values)


def torch_stack_stats(init_fn, depth=DEPTH, width=WIDTH):
    torch.manual_seed(0)
    layers = []
    for _ in range(depth):
        layer = torch.nn.Linear(width, width, bias=False)
        if init_fn is not None:
            init_fn(layer.weight)
        layers += [layer, torch.nn.ReLU()]
    model = torch.nn.Sequential(*layers)
    with torch.no_grad():
        return model(torch.randn(500, width)).std().item()


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    shown_layers = [0, 3, 6, 9]

    stats = {
        (activation_name, init_name): activation_and_gradient_stats(init, activation_name, np.random.default_rng(0))
        for activation_name in ACTIVATIONS
        for init_name, init in INITS.items()
    }

    print(f"=== Step 1: activation std through {DEPTH} layers of width {WIDTH} (layers 1, 4, 7, 10) ===")
    for (activation_name, init_name), (forward_stds, _) in stats.items():
        print(f"  {activation_name:4s} {init_name:10s} | forward  {format_stds([forward_stds[i] for i in shown_layers])}")

    print("\n=== Step 2: gradient std flowing back to layers 1, 4, 7, 10 ===")
    for (activation_name, init_name), (_, backward_stds) in stats.items():
        print(f"  {activation_name:4s} {init_name:10s} | backward {format_stds([backward_stds[i] for i in shown_layers])}")

    print("\n=== Step 3: train a 9-layer ReLU MLP on digits for 5 epochs ===")
    X_train, y_train, X_test, y_test = load_data(seed=0)
    sizes = [64] + [128] * 8 + [10]
    for init_name, init in INITS.items():
        params = init_mlp(sizes, init, np.random.default_rng(1))
        start_loss, _ = evaluate(params, X_train, y_train)
        with np.errstate(all="ignore"):
            params = train(params, X_train, y_train, epochs=5, lr=0.01, seed=2)
        end_loss, test_acc = evaluate(params, X_test, y_test)
        print(f"  {init_name:10s} | initial loss {start_loss:10.3f} | test loss after 5 epochs {end_loss:8.3f} | test acc {test_acc:.3f}")

    print("\n=== Step 4: constant init never breaks symmetry ===")
    constant_init = lambda fan_in, fan_out, rng: np.full((fan_in, fan_out), 0.05)
    params = train(init_mlp([64, 32, 10], constant_init, np.random.default_rng(0)), X_train, y_train, epochs=3, lr=0.01, seed=2)
    first_layer = params["W0"]
    spread = np.abs(first_layer - first_layer[:, :1]).max()
    print(f"  after 3 epochs, max difference between any two hidden units' weights = {spread:.2e}")
    print(f"  32 hidden units behave like 1; test acc {evaluate(params, X_test, y_test)[1]:.3f}")

    print("\n=== Use It: torch.nn.init ===")
    fan_in, fan_out = 512, 256
    weight = torch.empty(fan_out, fan_in)
    torch.nn.init.xavier_normal_(weight)
    print(f"  xavier_normal_   std {weight.std():.4f} vs formula sqrt(2/(in+out)) = {np.sqrt(2 / (fan_in + fan_out)):.4f}")
    torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")
    print(f"  kaiming_normal_  std {weight.std():.4f} vs formula sqrt(2/in)        = {np.sqrt(2 / fan_in):.4f}")
    default_linear = torch.nn.Linear(fan_in, fan_out)
    print(f"  nn.Linear default std {default_linear.weight.std():.4f} vs uniform(+-1/sqrt(in)) std = {1 / np.sqrt(3 * fan_in):.4f}")
    print(f"\n  output std of a {DEPTH}-layer ReLU stack fed with unit-variance input:")
    print(f"    nn.Linear default init : {torch_stack_stats(None):.2e}")
    print(f"    kaiming_normal_ (He)   : {torch_stack_stats(lambda w: torch.nn.init.kaiming_normal_(w, nonlinearity='relu')):.2e}")
    print(f"    xavier_normal_         : {torch_stack_stats(torch.nn.init.xavier_normal_):.2e}")


if __name__ == "__main__":
    main()
