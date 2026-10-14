import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_parameter(*shape, bound):
    return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))


class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        bound = 1.0 / math.sqrt(hidden_size)
        self.W_x = uniform_parameter(input_size, hidden_size, bound=bound)
        self.W_h = uniform_parameter(hidden_size, hidden_size, bound=bound)
        self.b = uniform_parameter(hidden_size, bound=bound)
        self.hidden_size = hidden_size

    def initial_state(self, batch):
        return torch.zeros(batch, self.hidden_size)

    def forward(self, x, state):
        return torch.tanh(x @ self.W_x + state @ self.W_h + self.b)

    def output(self, state):
        return state


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super().__init__()
        bound = 1.0 / math.sqrt(hidden_size)
        self.W_x = uniform_parameter(input_size, 4 * hidden_size, bound=bound)
        self.W_h = uniform_parameter(hidden_size, 4 * hidden_size, bound=bound)
        self.b = uniform_parameter(4 * hidden_size, bound=bound)
        with torch.no_grad():
            self.b[hidden_size:2 * hidden_size] += forget_bias
        self.hidden_size = hidden_size

    def initial_state(self, batch):
        return torch.zeros(batch, self.hidden_size), torch.zeros(batch, self.hidden_size)

    def forward(self, x, state):
        h, c = state
        gates = x @ self.W_x + h @ self.W_h + self.b
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)
        c = forget_gate * c + input_gate * candidate
        h = output_gate * torch.tanh(c)
        return h, c

    def output(self, state):
        return state[0]


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, update_bias=0.0):
        super().__init__()
        bound = 1.0 / math.sqrt(hidden_size)
        self.W_x = uniform_parameter(input_size, 3 * hidden_size, bound=bound)
        self.W_h = uniform_parameter(hidden_size, 3 * hidden_size, bound=bound)
        self.b_x = uniform_parameter(3 * hidden_size, bound=bound)
        self.b_h = uniform_parameter(3 * hidden_size, bound=bound)
        with torch.no_grad():
            self.b_x[hidden_size:2 * hidden_size] += update_bias
        self.hidden_size = hidden_size

    def initial_state(self, batch):
        return torch.zeros(batch, self.hidden_size)

    def forward(self, x, h):
        x_reset, x_update, x_new = (x @ self.W_x + self.b_x).chunk(3, dim=1)
        h_reset, h_update, h_new = (h @ self.W_h + self.b_h).chunk(3, dim=1)
        reset = torch.sigmoid(x_reset + h_reset)
        update = torch.sigmoid(x_update + h_update)
        candidate = torch.tanh(x_new + reset * h_new)
        return (1 - update) * candidate + update * h

    def output(self, state):
        return state


class SequenceClassifier(nn.Module):
    def __init__(self, cell, num_classes):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, num_classes)

    def forward(self, x):
        state = self.cell.initial_state(x.shape[0])
        for t in range(x.shape[1]):
            state = self.cell(x[:, t], state)
        return self.head(self.cell.output(state))


def make_memory_task(batch, delay, num_symbols, generator):
    first = torch.randint(0, num_symbols, (batch,), generator=generator)
    noise = torch.randint(num_symbols, 2 * num_symbols, (batch, delay), generator=generator)
    sequence = torch.cat([first[:, None], noise], dim=1)
    return F.one_hot(sequence, 2 * num_symbols).float(), first


def train(model, delay, generator, steps, num_symbols=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(steps):
        x, y = make_memory_task(64, delay, num_symbols, generator)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    x, y = make_memory_task(1000, delay, num_symbols, generator)
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def check_against_library(generator):
    input_size, hidden_size = 6, 5
    x = torch.randn(3, 7, input_size, generator=generator)

    lstm_cell = LSTMCell(input_size, hidden_size)
    library_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    with torch.no_grad():
        library_lstm.weight_ih_l0.copy_(lstm_cell.W_x.T)
        library_lstm.weight_hh_l0.copy_(lstm_cell.W_h.T)
        library_lstm.bias_ih_l0.copy_(lstm_cell.b)
        library_lstm.bias_hh_l0.zero_()
        state = lstm_cell.initial_state(3)
        outputs = []
        for t in range(7):
            state = lstm_cell(x[:, t], state)
            outputs.append(state[0])
        library_out, (library_h, library_c) = library_lstm(x)
    lstm_diff = (torch.stack(outputs, dim=1) - library_out).abs().max().item()
    cell_diff = (state[1] - library_c[0]).abs().max().item()

    gru_cell = GRUCell(input_size, hidden_size)
    library_gru = nn.GRU(input_size, hidden_size, batch_first=True)
    with torch.no_grad():
        library_gru.weight_ih_l0.copy_(gru_cell.W_x.T)
        library_gru.weight_hh_l0.copy_(gru_cell.W_h.T)
        library_gru.bias_ih_l0.copy_(gru_cell.b_x)
        library_gru.bias_hh_l0.copy_(gru_cell.b_h)
        h = gru_cell.initial_state(3)
        outputs = []
        for t in range(7):
            h = gru_cell(x[:, t], h)
            outputs.append(h)
        library_out, _ = library_gru(x)
    gru_diff = (torch.stack(outputs, dim=1) - library_out).abs().max().item()
    return lstm_diff, cell_diff, gru_diff


def gradient_reach(cell, delay, generator, num_symbols=4):
    model = SequenceClassifier(cell, num_symbols)
    x, y = make_memory_task(256, delay, num_symbols, generator)
    x.requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return (x.grad[:, 0].norm() / x.grad[:, -1].norm()).item()


def main():
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(0)

    lstm_diff, cell_diff, gru_diff = check_against_library(generator)
    print("Your cells vs PyTorch (copied weights, 7 steps):")
    print(f"  LSTM hidden states max diff {lstm_diff:.1e}, final cell state max diff {cell_diff:.1e}")
    print(f"  GRU hidden states max diff  {gru_diff:.1e}")

    input_size, hidden_size = 8, 32
    print(f"\nParameters with input {input_size}, hidden {hidden_size}:")
    for name, cell in [("RNN", VanillaRNNCell(input_size, hidden_size)), ("GRU", GRUCell(input_size, hidden_size)),
                       ("LSTM", LSTMCell(input_size, hidden_size))]:
        print(f"  {name:<5} {sum(p.numel() for p in cell.parameters()):>6,}")

    models = [
        ("RNN", lambda: VanillaRNNCell(input_size, hidden_size)),
        ("GRU z-bias 0", lambda: GRUCell(input_size, hidden_size)),
        ("GRU z-bias 3", lambda: GRUCell(input_size, hidden_size, update_bias=3.0)),
        ("LSTM f-bias 1", lambda: LSTMCell(input_size, hidden_size, forget_bias=1.0)),
        ("LSTM f-bias 3", lambda: LSTMCell(input_size, hidden_size, forget_bias=3.0)),
    ]

    print("\nGradient reaching the first input at initialization, ||dL/dx_0|| / ||dL/dx_T||:")
    print(f"  {'model':<14} {'delay 30':>9} {'delay 80':>9}")
    for name, build in models:
        values = []
        for delay in [30, 80]:
            torch.manual_seed(0)
            values.append(gradient_reach(build(), delay, generator))
        print(f"  {name:<14} {values[0]:>9.1e} {values[1]:>9.1e}")

    print("\nRemember the first symbol after 60 noise symbols (400 training steps each, chance = 0.25):")
    print(f"  {'model':<14} {'accuracy':>9} {'seconds':>8}")
    for name, build in models:
        start = time.perf_counter()
        torch.manual_seed(0)
        accuracy = train(SequenceClassifier(build(), 4), 60, generator, steps=400)
        print(f"  {name:<14} {accuracy:>9.3f} {time.perf_counter() - start:>8.1f}")

    library = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
    x, _ = make_memory_task(4, 10, 4, generator)
    output, (h_n, c_n) = library(x)
    print(f"\nnn.LSTM(num_layers=2): output {tuple(output.shape)}, h_n {tuple(h_n.shape)}, c_n {tuple(c_n.shape)}")


if __name__ == "__main__":
    main()
