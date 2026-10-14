import io

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons


class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class MiniModule:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, MiniModule):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def named_parameters(self, prefix=""):
        for name, param in self._parameters.items():
            yield prefix + name, param
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix + name + ".")

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def state_dict(self):
        return {name: param.value.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state):
        for name, param in self.named_parameters():
            param.value[...] = state[name]

    def train(self, mode=True):
        object.__setattr__(self, "training", mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for param in self.parameters():
            param.grad[...] = 0.0

    def __call__(self, *args):
        return self.forward(*args)


class MiniLinear(MiniModule):
    def __init__(self, fan_in, fan_out, rng):
        super().__init__()
        bound = 1.0 / np.sqrt(fan_in)
        self.weight = Parameter(rng.uniform(-bound, bound, size=(fan_out, fan_in)))
        self.bias = Parameter(rng.uniform(-bound, bound, size=fan_out))

    def forward(self, x):
        self.x = x
        return x @ self.weight.value.T + self.bias.value

    def backward(self, grad_out):
        self.weight.grad += grad_out.T @ self.x
        self.bias.grad += grad_out.sum(axis=0)
        return grad_out @ self.weight.value


class MiniMLP(MiniModule):
    def __init__(self, hidden, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.fc1 = MiniLinear(2, hidden, rng)
        self.fc2 = MiniLinear(hidden, 2, rng)

    def forward(self, x):
        self.pre = self.fc1(x)
        return self.fc2(np.maximum(0.0, self.pre))

    def backward(self, grad_logits):
        grad_hidden = self.fc2.backward(grad_logits)
        self.fc1.backward(grad_hidden * (self.pre > 0))


def softmax_cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    grad = np.exp(log_probs)
    grad[np.arange(len(labels)), labels] -= 1
    return -np.mean(log_probs[np.arange(len(labels)), labels]), grad / len(labels)


class MoonNet(torch.nn.Module):
    def __init__(self, hidden=32, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("input_mean", torch.zeros(2))
        self.register_buffer("input_std", torch.ones(2))

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.head(x)


def train_mini(X, y, epochs, learning_rate):
    model = MiniMLP(hidden=16, seed=0)
    for epoch in range(epochs):
        model.zero_grad()
        loss, grad = softmax_cross_entropy(model(X), y)
        model.backward(grad)
        for param in model.parameters():
            param.value -= learning_rate * param.grad
        if epoch % 200 == 0:
            print(f"  epoch {epoch:4d} loss {loss:.4f}")
    accuracy = np.mean(model(X).argmax(axis=1) == y)
    return model, accuracy


def train_torch(model, X, y, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        loss = F.cross_entropy(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"  epoch {epoch:3d} loss {loss.item():.4f}")
    return model


def main():
    torch.manual_seed(0)
    X_np, y_np = make_moons(n_samples=400, noise=0.2, random_state=0)
    X_np = X_np * 3 + 5

    print("=== Step 1: a mini module system in NumPy ===")
    mini = MiniMLP(hidden=16, seed=0)
    for name, param in mini.named_parameters():
        print(f"  {name:10s} {param.value.shape}")
    print(f"  total parameters: {sum(p.value.size for p in mini.parameters())}")

    print("\n=== Step 2: train it with parameters() + zero_grad() ===")
    X_centered = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)
    mini, mini_acc = train_mini(X_centered, y_np, epochs=1000, learning_rate=0.5)
    print(f"  training accuracy = {mini_acc:.3f}")

    print("\n=== Step 3: state_dict round trip ===")
    clone = MiniMLP(hidden=16, seed=123)
    clone.load_state_dict(mini.state_dict())
    print(f"  clone matches original: {np.allclose(clone(X_centered), mini(X_centered))}")

    print("\n=== Use It: torch.nn.Module ===")
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np)
    model = MoonNet()
    model.input_mean.copy_(X.mean(dim=0))
    model.input_std.copy_(X.std(dim=0))
    print(model)
    for name, param in model.named_parameters():
        print(f"  param  {name:12s} {tuple(param.shape)}")
    for name, buffer in model.named_buffers():
        print(f"  buffer {name:12s} {tuple(buffer.shape)}")
    print(f"  trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_torch(model, X, y, epochs=400, learning_rate=0.01)
    model.eval()
    with torch.no_grad():
        eval_out_1 = model(X)
        eval_out_2 = model(X)
        accuracy = (eval_out_1.argmax(dim=1) == y).float().mean().item()
    print(f"  eval accuracy = {accuracy:.3f}, eval is deterministic: {torch.equal(eval_out_1, eval_out_2)}")
    model.train()
    with torch.no_grad():
        print(f"  train mode with dropout is deterministic: {torch.equal(model(X), model(X))}")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = MoonNet()
    restored.load_state_dict(torch.load(buffer))
    restored.eval()
    model.eval()
    with torch.no_grad():
        print(f"  restored model matches: {torch.equal(restored(X), model(X))}")
    print(f"  state_dict keys: {list(model.state_dict().keys())}")


if __name__ == "__main__":
    main()
