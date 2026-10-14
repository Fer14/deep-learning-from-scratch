import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, SOS, EOS = 10, 11, 12
VOCAB_SIZE = 13


def make_batch(batch_size, length, generator, device):
    source = torch.randint(0, 10, (batch_size, length), generator=generator)
    target = source.flip(dims=[1])
    decoder_input = torch.cat([torch.full((batch_size, 1), SOS), target], dim=1)
    decoder_target = torch.cat([target, torch.full((batch_size, 1), EOS)], dim=1)
    return source.to(device), decoder_input.to(device), decoder_target.to(device)


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, source):
        outputs, last = self.rnn(self.embed(source))
        return outputs, last


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, decoder_input, hidden):
        outputs, hidden = self.rnn(self.embed(decoder_input), hidden)
        return self.out(outputs), hidden


class Seq2Seq(nn.Module):
    def __init__(self, embed_size=32, hidden_size=128):
        super().__init__()
        self.encoder = Encoder(embed_size, hidden_size)
        self.decoder = Decoder(embed_size, hidden_size)

    def forward(self, source, decoder_input):
        _, context = self.encoder(source)
        logits, _ = self.decoder(decoder_input, context)
        return logits

    @torch.no_grad()
    def greedy_decode(self, source, max_length):
        _, hidden = self.encoder(source)
        token = torch.full((source.shape[0], 1), SOS, device=source.device)
        outputs = []
        for _ in range(max_length):
            logits, hidden = self.decoder(token, hidden)
            token = logits.argmax(dim=-1)
            outputs.append(token)
        return torch.cat(outputs, dim=1)


def train_teacher_forcing(model, steps, generator, device, lengths=(3, 12)):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(1, steps + 1):
        length = torch.randint(lengths[0], lengths[1] + 1, (1,), generator=generator).item()
        source, decoder_input, decoder_target = make_batch(64, length, generator, device)
        logits = model(source, decoder_input)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            print(f"  step {step:>4}  loss {loss.item():.4f}")


def evaluate(model, length, generator, device, count=500):
    model.eval()
    source, _, decoder_target = make_batch(count, length, generator, device)
    predictions = model.greedy_decode(source, length + 1)
    token_accuracy = (predictions[:, :length] == decoder_target[:, :length]).float().mean().item()
    exact = (predictions == decoder_target).all(dim=1).float().mean().item()
    model.train()
    return token_accuracy, exact


def position_accuracy(model, length, generator, device, count=500):
    model.eval()
    source, _, decoder_target = make_batch(count, length, generator, device)
    predictions = model.greedy_decode(source, length + 1)
    model.train()
    return (predictions[:, :length] == decoder_target[:, :length]).float().mean(dim=0)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(0)

    source, decoder_input, decoder_target = make_batch(1, 6, generator, "cpu")
    print("Task: reverse a sequence of digits. One training example:")
    print(f"  encoder input   {source[0].tolist()}")
    print(f"  decoder input   {decoder_input[0].tolist()}   (11 = <sos>)")
    print(f"  decoder target  {decoder_target[0].tolist()}   (12 = <eos>)")

    model = Seq2Seq().to(device)
    print(f"\nTraining a GRU encoder-decoder ({sum(p.numel() for p in model.parameters()):,} parameters) "
          f"on lengths 3-12 with teacher forcing, device {device}:")
    train_teacher_forcing(model, 2500, generator, device)

    print("\nGreedy decoding accuracy by sequence length:")
    print(f"  {'length':>6} {'token acc':>10} {'exact match':>12}")
    for length in [4, 8, 12, 16, 20]:
        token_accuracy, exact = evaluate(model, length, generator, device)
        note = "  (longer than any training sequence)" if length > 12 else ""
        print(f"  {length:>6} {token_accuracy:>10.3f} {exact:>12.3f}{note}")

    per_position = position_accuracy(model, 12, generator, device)
    print("\nPer-position token accuracy for length 12 (output position 1 = last input digit):")
    print("  " + " ".join(f"{value:.2f}" for value in per_position.tolist()))

    sample = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]], device=device)
    model.eval()
    print(f"\nExample: {sample[0].tolist()} -> {model.greedy_decode(sample, 9)[0].tolist()}")

    sequences = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7, 8]), torch.tensor([9, 0])]
    lengths = torch.tensor([len(s) for s in sequences])
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD).to(device)
    encoder = model.encoder
    with torch.no_grad():
        packed = pack_padded_sequence(encoder.embed(padded), lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, packed_last = encoder.rnn(packed)
        unpacked, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        separate_last = torch.stack([encoder(s[None].to(device))[1][0, 0] for s in sequences])
    print(f"\nPadded batch shape {tuple(padded.shape)}; packed last states match per-sequence runs: "
          f"{torch.allclose(packed_last[0], separate_last, atol=1e-5)}")
    print(f"Unpacked outputs {tuple(unpacked.shape)}, padded positions are zeros: "
          f"{bool((unpacked[1, 3:] == 0).all() and (unpacked[2, 2:] == 0).all())}")


if __name__ == "__main__":
    main()
