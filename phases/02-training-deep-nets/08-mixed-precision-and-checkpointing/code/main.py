import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits


def to_bfloat16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounding = 0x7FFF + ((bits >> 16) & 1)
    return (((bits + rounding) >> 16) << 16).astype(np.uint32).view(np.float32)


def half_matmul(a, b):
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)


def init_mlp(sizes, rng):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)
        params[f"b{i}"] = np.zeros(fan_out, dtype=np.float32)
    return params


def loss_and_grads(params, X, y, half=False, loss_scale=1.0):
    dtype = np.float16 if half else np.float32
    matmul = half_matmul if half else np.matmul
    weights = {name: value.astype(dtype) for name, value in params.items()}
    n_layers = len(params) // 2
    activations = [X.astype(dtype)]
    hidden = activations[0]
    for i in range(n_layers):
        z = matmul(hidden, weights[f"W{i}"]) + weights[f"b{i}"]
        hidden = np.maximum(z, 0) if i < n_layers - 1 else z
        activations.append(hidden)
    logits = activations[-1].astype(np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    grad = probs
    grad[np.arange(n), y] -= 1.0
    grad = (grad * loss_scale / n).astype(dtype)
    grads = {}
    for i in reversed(range(n_layers)):
        grads[f"W{i}"] = matmul(activations[i].T, grad)
        grads[f"b{i}"] = grad.astype(np.float32).sum(axis=0).astype(dtype)
        if i > 0:
            grad = matmul(grad, weights[f"W{i}"].T) * (activations[i] > 0)
    return loss, grads


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name in params:
            grad = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)
            params[name] -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(np.float32)


class DynamicLossScaler:
    def __init__(self, init_scale=2.0**16, growth_interval=100):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.good_steps = 0
        self.skipped = 0

    def unscale_or_skip(self, grads):
        unscaled = {name: grad.astype(np.float32) / self.scale for name, grad in grads.items()}
        if not all(np.isfinite(g).all() for g in unscaled.values()):
            self.scale /= 2.0
            self.good_steps = 0
            self.skipped += 1
            return None
        self.good_steps += 1
        if self.good_steps == self.growth_interval:
            self.scale *= 2.0
            self.good_steps = 0
        return unscaled


def train_epochs(state, X, y, epochs, half, batch_size=32):
    params, optimizer, scaler, rng = state["params"], state["optimizer"], state["scaler"], state["rng"]
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            with np.errstate(over="ignore", invalid="ignore"):
                _, grads = loss_and_grads(params, X[batch], y[batch], half=half, loss_scale=scaler.scale if half else 1.0)
            grads = scaler.unscale_or_skip(grads) if half else grads
            if grads is not None:
                optimizer.step(params, grads)
        state["epoch"] += 1
    return state


def new_state(seed):
    rng = np.random.default_rng(seed)
    params = init_mlp([64, 128, 128, 10], rng)
    return {"params": params, "optimizer": Adam(params), "scaler": DynamicLossScaler(), "rng": rng, "epoch": 0}


def save_checkpoint(path, state):
    arrays = {f"param/{k}": v for k, v in state["params"].items()}
    arrays.update({f"m/{k}": v for k, v in state["optimizer"].m.items()})
    arrays.update({f"v/{k}": v for k, v in state["optimizer"].v.items()})
    np.savez(
        path,
        **arrays,
        adam_t=state["optimizer"].t,
        epoch=state["epoch"],
        loss_scale=state["scaler"].scale,
        good_steps=state["scaler"].good_steps,
        rng_state=np.array(json.dumps(state["rng"].bit_generator.state)),
    )


def load_checkpoint(path, seed):
    state = new_state(seed)
    with np.load(path) as data:
        for name in state["params"]:
            state["params"][name] = data[f"param/{name}"].copy()
            state["optimizer"].m[name] = data[f"m/{name}"].copy()
            state["optimizer"].v[name] = data[f"v/{name}"].copy()
        state["optimizer"].t = int(data["adam_t"])
        state["epoch"] = int(data["epoch"])
        state["scaler"].scale = float(data["loss_scale"])
        state["scaler"].good_steps = int(data["good_steps"])
        state["rng"].bit_generator.state = json.loads(str(data["rng_state"]))
    return state


def accuracy(params, X, y):
    hidden = X.astype(np.float32)
    n_layers = len(params) // 2
    for i in range(n_layers):
        hidden = hidden @ params[f"W{i}"] + params[f"b{i}"]
        if i < n_layers - 1:
            hidden = np.maximum(hidden, 0)
    return np.mean(hidden.argmax(axis=1) == y)


def load_data(seed):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    y = digits.target
    order = np.random.default_rng(seed).permutation(len(y))
    X, y = X[order], y[order]
    return X[:1400], y[:1400], X[1400:], y[1400:]


def torch_model():
    return torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    print("=== Step 1: three floating point formats ===")
    fp16 = np.finfo(np.float16)
    fp32 = np.finfo(np.float32)
    print(f"  float32  max {fp32.max:.2e}  eps {fp32.eps:.1e}  smallest normal {fp32.tiny:.1e}")
    print(f"  float16  max {fp16.max:.0f}    eps {fp16.eps:.1e}  smallest normal {fp16.tiny:.1e}")
    print(f"  bfloat16 max {to_bfloat16(3.38e38):.2e}  eps {2.0**-7:.1e}  smallest normal {fp32.tiny:.1e} (same exponent bits as float32)")
    for value in (70000.0, 1.0 + 1e-3, 3.14159265, 1e-8):
        with np.errstate(over="ignore"):
            as_half = float(np.float16(value))
        print(
            f"  {value:<12g} -> float16 {as_half:<12.8g} bfloat16 {float(to_bfloat16(value)):<12.8g}"
            f" float32 {float(np.float32(value)):.8g}"
        )

    print("\n=== Step 2: small gradients underflow in float16, loss scaling rescues them ===")
    rng = np.random.default_rng(0)
    gradients = np.exp(rng.uniform(np.log(1e-10), np.log(1e-2), size=100000)).astype(np.float32)
    for scale in (1.0, 2.0**10, 2.0**16):
        lost = np.mean(np.float16(gradients * scale) == 0)
        print(f"  loss scale {scale:>8.0f} | fraction of gradients that become exactly 0 in float16: {lost:.1%}")

    print("\n=== Step 3: tiny updates vanish without float32 master weights ===")
    weight16, weight32 = np.float16(1.0), np.float32(1.0)
    for _ in range(1000):
        weight16 = np.float16(weight16 + np.float16(1e-4))
        weight32 = np.float32(weight32 + np.float32(1e-4))
    print(f"  1.0 + 1000 updates of 1e-4: float16 weight = {float(weight16):.4f}, float32 weight = {float(weight32):.4f}, exact = 1.1000")

    print("\n=== Step 4: mixed precision training in NumPy (fp16 compute, fp32 master weights, dynamic loss scale) ===")
    X_train, y_train, X_test, y_test = load_data(seed=0)
    for name, half in (("float32", False), ("mixed fp16", True)):
        state = train_epochs(new_state(seed=1), X_train, y_train, epochs=10, half=half)
        extra = f" | final loss scale {state['scaler'].scale:.0f}, skipped steps {state['scaler'].skipped}" if half else ""
        print(f"  {name:10s} | test acc {accuracy(state['params'], X_test, y_test):.3f}{extra}")

    print("\n=== Step 5: checkpoint and resume gives bit-identical results ===")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.npz"
        uninterrupted = train_epochs(new_state(seed=1), X_train, y_train, epochs=6, half=True)
        first_half = train_epochs(new_state(seed=1), X_train, y_train, epochs=3, half=True)
        save_checkpoint(path, first_half)
        print(f"  saved after epoch {first_half['epoch']}: {path.name}, {path.stat().st_size / 1024:.0f} KiB")
        del first_half
        resumed = train_epochs(load_checkpoint(path, seed=99), X_train, y_train, epochs=3, half=True)
        diff = max(np.abs(uninterrupted["params"][k] - resumed["params"][k]).max() for k in resumed["params"])
        print(f"  resumed to epoch {resumed['epoch']} | max |uninterrupted - resumed| = {diff:.1e}")
        partial = new_state(seed=1)
        partial["params"] = load_checkpoint(path, seed=1)["params"]
        partial["epoch"] = 3
        weights_only = train_epochs(partial, X_train, y_train, epochs=3, half=True)
        diff = max(np.abs(uninterrupted["params"][k] - weights_only["params"][k]).max() for k in resumed["params"])
        print(f"  restoring weights only (fresh Adam state and RNG) | max diff = {diff:.1e}")

    print("\n=== Use It: torch.autocast, GradScaler, torch.save ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    X_t = torch.tensor(X_train, device=device)
    y_t = torch.tensor(y_train, device=device)
    model = torch_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    with torch.autocast(device_type=device, dtype=amp_dtype):
        logits = model(X_t[:4])
        loss = torch.nn.functional.cross_entropy(logits, y_t[:4])
    print(f"  device {device}, autocast dtype {amp_dtype}: weights {model[0].weight.dtype}, logits {logits.dtype}, loss {loss.dtype}")
    generator = torch.Generator().manual_seed(0)
    for _ in range(10):
        for batch in torch.randperm(len(y_t), generator=generator).split(32):
            batch = batch.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype):
                loss = torch.nn.functional.cross_entropy(model(X_t[batch]), y_t[batch])
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    with torch.no_grad():
        test_acc = (model(torch.tensor(X_test, device=device)).argmax(1).cpu().numpy() == y_test).mean()
    print(f"  autocast training, 10 epochs | test acc {test_acc:.3f} | GradScaler enabled: {scaler.is_enabled()}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": 10}, path)
        checkpoint = torch.load(path, map_location=device)
        restored = torch_model().to(device)
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        with torch.no_grad():
            same = torch.equal(model(X_t), restored(X_t))
        print(f"  torch.save / torch.load: {path.stat().st_size / 1024:.0f} KiB, restored outputs identical: {same}, epoch {checkpoint['epoch']}")

    x = torch.randn(64, 64, device=device, requires_grad=True)
    reference = model(x).sum()
    reference.backward()
    reference_grad = x.grad.clone()
    x.grad = None
    checkpointed = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False).sum()
    checkpointed.backward()
    print(f"  activation checkpointing: same gradient as normal backward: {torch.allclose(reference_grad, x.grad)}")


if __name__ == "__main__":
    main()
