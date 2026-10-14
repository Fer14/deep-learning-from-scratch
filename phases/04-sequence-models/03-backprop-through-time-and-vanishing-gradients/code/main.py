import torch
import torch.nn as nn
import torch.nn.functional as F


def rnn_forward(x, h0, W_xh, W_hh, b_h, W_hy, b_y):
    hidden_states = [h0]
    for t in range(x.shape[1]):
        hidden_states.append(torch.tanh(x[:, t] @ W_xh + hidden_states[-1] @ W_hh + b_h))
    logits = hidden_states[-1] @ W_hy + b_y
    return logits, hidden_states


def rnn_backward(x, hidden_states, logits, targets, W_xh, W_hh, W_hy):
    batch = x.shape[0]
    d_logits = (torch.softmax(logits, dim=-1) - F.one_hot(targets, logits.shape[-1]).float()) / batch
    grads = {"W_hy": hidden_states[-1].T @ d_logits, "b_y": d_logits.sum(dim=0),
             "W_xh": torch.zeros_like(W_xh), "W_hh": torch.zeros_like(W_hh), "b_h": torch.zeros(W_hh.shape[0])}
    d_h = d_logits @ W_hy.T
    hidden_grad_norms = []
    for t in reversed(range(x.shape[1])):
        hidden_grad_norms.append(d_h.norm().item())
        d_pre = d_h * (1 - hidden_states[t + 1] ** 2)
        grads["W_xh"] += x[:, t].T @ d_pre
        grads["W_hh"] += hidden_states[t].T @ d_pre
        grads["b_h"] += d_pre.sum(dim=0)
        d_h = d_pre @ W_hh.T
    return grads, hidden_grad_norms[::-1]


def random_orthogonal(size, generator):
    q, _ = torch.linalg.qr(torch.randn(size, size, generator=generator))
    return q


def make_memory_task(batch, delay, num_symbols, generator):
    first = torch.randint(0, num_symbols, (batch,), generator=generator)
    noise = torch.randint(num_symbols, 2 * num_symbols, (batch, delay), generator=generator)
    sequence = torch.cat([first[:, None], noise], dim=1)
    return F.one_hot(sequence, 2 * num_symbols).float(), first


class MemoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, last = self.rnn(x)
        return self.head(last[0])


def first_step_signal(model, delay, generator, num_symbols):
    x, y = make_memory_task(256, delay, num_symbols, generator)
    x.requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    model.zero_grad()
    return (x.grad[:, 0].norm() / x.grad[:, -1].norm()).item()


def train_memory_task(delay, generator, steps=400, num_symbols=4):
    torch.manual_seed(0)
    model = MemoryRNN(2 * num_symbols, 32, num_symbols)
    signal = first_step_signal(model, delay, generator, num_symbols)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = make_memory_task(64, delay, num_symbols, generator)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    x, y = make_memory_task(1000, delay, num_symbols, generator)
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    return accuracy, signal


def main():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    batch, steps, input_size, hidden_size, num_classes = 8, 12, 5, 16, 3
    x = torch.randn(batch, steps, input_size, generator=generator)
    targets = torch.randint(0, num_classes, (batch,), generator=generator)
    params = {
        "W_xh": torch.randn(input_size, hidden_size, generator=generator) * 0.5,
        "W_hh": torch.randn(hidden_size, hidden_size, generator=generator) * 0.25,
        "b_h": torch.zeros(hidden_size),
        "W_hy": torch.randn(hidden_size, num_classes, generator=generator) * 0.5,
        "b_y": torch.zeros(num_classes),
    }
    h0 = torch.zeros(batch, hidden_size)
    logits, hidden_states = rnn_forward(x, h0, **params)
    manual, _ = rnn_backward(x, hidden_states, logits, targets, params["W_xh"], params["W_hh"], params["W_hy"])

    leaves = {name: value.clone().requires_grad_(True) for name, value in params.items()}
    auto_logits, _ = rnn_forward(x, h0, **leaves)
    F.cross_entropy(auto_logits, targets).backward()
    print("Manual BPTT vs autograd (max abs difference):")
    for name in params:
        print(f"  d{name:<5} {(manual[name] - leaves[name].grad).abs().max():.1e}")

    print("\nGradient norm ||dL/dh_t|| flowing back through 50 steps, W_hh = radius * orthogonal matrix:")
    long_x = torch.randn(batch, 50, input_size, generator=generator) * 0.1
    rows = {}
    for radius in [0.5, 0.9, 1.0, 1.1, 1.5]:
        trial = dict(params)
        trial["W_hh"] = radius * random_orthogonal(hidden_size, torch.Generator().manual_seed(1))
        trial["W_xh"] = params["W_xh"] * 0.1
        logits, hidden_states = rnn_forward(long_x, h0, **trial)
        _, norms = rnn_backward(long_x, hidden_states, logits, targets, trial["W_xh"], trial["W_hh"], trial["W_hy"])
        rows[radius] = norms
    print(f"  {'radius':>6} " + " ".join(f"t={t:>2}".rjust(9) for t in [49, 40, 30, 20, 10, 0]))
    for radius, norms in rows.items():
        print(f"  {radius:>6} " + " ".join(f"{norms[t]:>9.2e}" for t in [49, 40, 30, 20, 10, 0]))

    print("\nProduct of Jacobians dh_t/dh_0, each factor diag(1 - h_t^2) W_hh^T, with radius 1.0:")
    trial = dict(params)
    trial["W_hh"] = random_orthogonal(hidden_size, torch.Generator().manual_seed(1))
    for input_scale in [0.1, 1.0, 3.0]:
        sequence = torch.randn(1, 20, input_size, generator=generator) * input_scale
        _, hidden_states = rnn_forward(sequence, h0[:1], **trial)
        product = torch.eye(hidden_size)
        norms = []
        for t in range(1, 21):
            product = torch.diag(1 - hidden_states[t][0] ** 2) @ trial["W_hh"].T @ product
            if t % 5 == 0:
                norms.append(torch.linalg.matrix_norm(product, ord=2).item())
        print(f"  input scale {input_scale:>3}: ||dh_t/dh_0|| at t=5,10,15,20: " + ", ".join(f"{n:.2e}" for n in norms))

    print("\nExploding gradients and clipping (nn.RNN with W_hh = 3 x orthogonal, 50 steps):")
    model = nn.RNN(input_size, hidden_size, batch_first=True)
    with torch.no_grad():
        model.weight_hh_l0.copy_(3.0 * random_orthogonal(hidden_size, torch.Generator().manual_seed(2)))
    output, _ = model(torch.randn(batch, 50, input_size, generator=generator) * 0.01)
    output[:, -1].sum().backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    after = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters()))
    print(f"  gradient norm before clipping {total:.2e}, after clip_grad_norm_(max_norm=1.0) {after:.2f}")

    print("\nRemember the first symbol after a delay of noise symbols (vanilla RNN, 400 steps, chance = 0.25):")
    print(f"  {'delay':>5} {'accuracy':>9} {'grad at x_0 / grad at x_T (init)':>34}")
    for delay in [5, 10, 20, 30, 40]:
        accuracy, signal = train_memory_task(delay, generator)
        print(f"  {delay:>5} {accuracy:>9.3f} {signal:>34.2e}")


if __name__ == "__main__":
    main()
