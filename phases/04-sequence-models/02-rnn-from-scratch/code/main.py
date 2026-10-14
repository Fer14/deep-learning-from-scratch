import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT = (
    "the river runs past the old mill and the mill turns slowly in the morning light. "
    "the miller walks to the river every morning and listens to the water. "
    "the water is cold and clear and the stones under it are round and smooth. "
    "in the evening the miller sits by the fire and tells stories to the children. "
    "the children listen to the stories and the fire burns low and warm. "
    "the stories are about the river and the mill and the long winter nights. "
    "when the snow falls the river slows and the mill turns slowly. "
    "when the spring comes the river runs fast and the mill sings again. "
    "the miller smiles and the children run to the river to watch the water. "
    "the old mill has stood by the river for a hundred years and it will stand for a hundred more. "
)


class CharVocabulary:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return torch.tensor([self.index[c] for c in text])

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


class ScratchRNN:
    def __init__(self, vocab_size, hidden_size, generator):
        scale_x = 1.0 / math.sqrt(vocab_size)
        scale_h = 1.0 / math.sqrt(hidden_size)
        self.W_xh = (torch.randn(vocab_size, hidden_size, generator=generator) * scale_x).requires_grad_(True)
        self.W_hh = (torch.randn(hidden_size, hidden_size, generator=generator) * scale_h).requires_grad_(True)
        self.b_h = torch.zeros(hidden_size, requires_grad=True)
        self.W_hy = (torch.randn(hidden_size, vocab_size, generator=generator) * scale_h).requires_grad_(True)
        self.b_y = torch.zeros(vocab_size, requires_grad=True)
        self.hidden_size = hidden_size

    def parameters(self):
        return [self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y]

    def step(self, x_t, h):
        return torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)

    def forward(self, inputs, h):
        batch, steps = inputs.shape
        x = F.one_hot(inputs, self.W_xh.shape[0]).float()
        hidden_states = []
        for t in range(steps):
            h = self.step(x[:, t], h)
            hidden_states.append(h)
        hidden = torch.stack(hidden_states, dim=1)
        return hidden @ self.W_hy + self.b_y, h

    def initial_state(self, batch):
        return torch.zeros(batch, self.hidden_size)


def sample_batch(data, batch_size, steps, generator):
    starts = torch.randint(0, len(data) - steps - 1, (batch_size,), generator=generator)
    inputs = torch.stack([data[s:s + steps] for s in starts])
    targets = torch.stack([data[s + 1:s + steps + 1] for s in starts])
    return inputs, targets


def generate(model, vocabulary, prompt, length, temperature, generator):
    h = model.initial_state(1)
    ids = vocabulary.encode(prompt)
    with torch.no_grad():
        logits, h = model.forward(ids[None], h)
        output = list(ids.tolist())
        next_logits = logits[0, -1]
        for _ in range(length):
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1, generator=generator).item()
            output.append(next_id)
            logits, h = model.forward(torch.tensor([[next_id]]), h)
            next_logits = logits[0, -1]
    return vocabulary.decode(output)


def main():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    vocabulary = CharVocabulary(TEXT)
    data = vocabulary.encode(TEXT)
    print(f"Text: {len(TEXT)} characters, vocabulary of {len(vocabulary.chars)} characters")
    print(f"Uniform guessing loss: ln({len(vocabulary.chars)}) = {math.log(len(vocabulary.chars)):.3f}")

    model = ScratchRNN(len(vocabulary.chars), 128, generator)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"RNN with hidden size 128: {parameter_count:,} parameters (the same weights at every time step)")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1, 1201):
        inputs, targets = sample_batch(data, 32, 48, generator)
        logits, _ = model.forward(inputs, model.initial_state(32))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0:
            print(f"  step {step:>4}  loss {loss.item():.3f}")
            if step in (200, 1200):
                print(f"    sample: {generate(model, vocabulary, 'the ', 80, 0.8, generator)!r}")

    for temperature in [0.3, 1.0, 1.5]:
        print(f"\nTemperature {temperature}:")
        print("  " + generate(model, vocabulary, "the miller ", 120, temperature, generator))

    library_rnn = nn.RNN(len(vocabulary.chars), 128, batch_first=True, nonlinearity="tanh")
    with torch.no_grad():
        library_rnn.weight_ih_l0.copy_(model.W_xh.T)
        library_rnn.weight_hh_l0.copy_(model.W_hh.T)
        library_rnn.bias_ih_l0.copy_(model.b_h)
        library_rnn.bias_hh_l0.zero_()
        inputs, _ = sample_batch(data, 4, 20, generator)
        library_hidden, library_last = library_rnn(F.one_hot(inputs, len(vocabulary.chars)).float())
        h = model.initial_state(4)
        scratch_states = []
        for t in range(20):
            h = model.step(F.one_hot(inputs[:, t], len(vocabulary.chars)).float(), h)
            scratch_states.append(h)
        scratch_hidden = torch.stack(scratch_states, dim=1)
    print(f"\nScratch RNN vs nn.RNN hidden states max diff: {(scratch_hidden - library_hidden).abs().max():.1e}")
    print(f"nn.RNN output {tuple(library_hidden.shape)} (all hidden states), h_n {tuple(library_last.shape)} (last state)")


if __name__ == "__main__":
    main()
