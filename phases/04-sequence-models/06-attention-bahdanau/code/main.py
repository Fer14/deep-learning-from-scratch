import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS, EOS = 10, 11
VOCAB_SIZE = 12


def make_batch(batch_size, length, generator, device):
    source = torch.randint(0, 10, (batch_size, length), generator=generator)
    target = source.flip(dims=[1])
    decoder_input = torch.cat([torch.full((batch_size, 1), SOS), target], dim=1)
    decoder_target = torch.cat([target, torch.full((batch_size, 1), EOS)], dim=1)
    return source.to(device), decoder_input.to(device), decoder_target.to(device)


class AdditiveAttention(nn.Module):
    def __init__(self, decoder_size, encoder_size, attention_size):
        super().__init__()
        self.query_proj = nn.Linear(decoder_size, attention_size, bias=False)
        self.key_proj = nn.Linear(encoder_size, attention_size, bias=False)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, projected_keys):
        energy = torch.tanh(self.query_proj(decoder_state)[:, None, :] + projected_keys)
        weights = torch.softmax(self.score(energy).squeeze(-1), dim=1)
        context = torch.bmm(weights[:, None, :], encoder_outputs).squeeze(1)
        return context, weights


class AttentionSeq2Seq(nn.Module):
    def __init__(self, embed_size=32, hidden_size=128):
        super().__init__()
        self.embed_source = nn.Embedding(VOCAB_SIZE, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.init_state = nn.Linear(hidden_size, hidden_size)
        self.embed_target = nn.Embedding(VOCAB_SIZE, embed_size)
        self.attention = AdditiveAttention(hidden_size, hidden_size, 64)
        self.cell = nn.GRUCell(embed_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size + hidden_size + embed_size, VOCAB_SIZE)

    def encode(self, source):
        encoder_outputs, _ = self.encoder(self.embed_source(source))
        state = torch.tanh(self.init_state(encoder_outputs.mean(dim=1)))
        return encoder_outputs, self.attention.key_proj(encoder_outputs), state

    def step(self, token, state, encoder_outputs, projected_keys):
        embedded = self.embed_target(token)
        context, weights = self.attention(state, encoder_outputs, projected_keys)
        state = self.cell(torch.cat([embedded, context], dim=1), state)
        logits = self.out(torch.cat([state, context, embedded], dim=1))
        return logits, state, weights

    def forward(self, source, decoder_input):
        encoder_outputs, projected_keys, state = self.encode(source)
        all_logits = []
        for t in range(decoder_input.shape[1]):
            logits, state, _ = self.step(decoder_input[:, t], state, encoder_outputs, projected_keys)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, source, max_length):
        encoder_outputs, projected_keys, state = self.encode(source)
        token = torch.full((source.shape[0],), SOS, device=source.device)
        outputs, attention_maps = [], []
        for _ in range(max_length):
            logits, state, weights = self.step(token, state, encoder_outputs, projected_keys)
            token = logits.argmax(dim=-1)
            outputs.append(token)
            attention_maps.append(weights)
        return torch.stack(outputs, dim=1), torch.stack(attention_maps, dim=1)


class PlainSeq2Seq(nn.Module):
    def __init__(self, embed_size=32, hidden_size=128):
        super().__init__()
        self.embed_source = nn.Embedding(VOCAB_SIZE, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.embed_target = nn.Embedding(VOCAB_SIZE, embed_size)
        self.decoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, source, decoder_input):
        _, context = self.encoder(self.embed_source(source))
        outputs, _ = self.decoder(self.embed_target(decoder_input), context)
        return self.out(outputs)

    @torch.no_grad()
    def greedy_decode(self, source, max_length):
        _, hidden = self.encoder(self.embed_source(source))
        token = torch.full((source.shape[0], 1), SOS, device=source.device)
        outputs = []
        for _ in range(max_length):
            output, hidden = self.decoder(self.embed_target(token), hidden)
            token = self.out(output).argmax(dim=-1)
            outputs.append(token)
        return torch.cat(outputs, dim=1), None


def train(model, steps, generator, device, lengths=(3, 15)):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / steps)
    for step in range(1, steps + 1):
        length = torch.randint(lengths[0], lengths[1] + 1, (1,), generator=generator).item()
        source, decoder_input, decoder_target = make_batch(64, length, generator, device)
        logits = model(source, decoder_input)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % 500 == 0:
            print(f"    step {step:>4}  loss {loss.item():.4f}")


def exact_match(model, length, generator, device, count=500):
    model.eval()
    source, _, decoder_target = make_batch(count, length, generator, device)
    predictions, _ = model.greedy_decode(source, length + 1)
    model.train()
    return (predictions == decoder_target).all(dim=1).float().mean().item()


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(0)

    models = {}
    for name, build in [("no attention", PlainSeq2Seq), ("Bahdanau attention", AttentionSeq2Seq)]:
        torch.manual_seed(0)
        model = build().to(device)
        print(f"Training {name} ({sum(p.numel() for p in model.parameters()):,} parameters) on lengths 3-15:")
        start = time.perf_counter()
        train(model, 2000, generator, device)
        print(f"    {time.perf_counter() - start:.1f} s")
        models[name] = model

    print("\nExact-match accuracy of greedy decoding:")
    print(f"  {'length':>6} {'no attention':>13} {'attention':>10}")
    for length in [5, 10, 15, 20]:
        scores = [exact_match(models[name], length, generator, device) for name in models]
        note = "  (unseen length)" if length > 15 else ""
        print(f"  {length:>6} {scores[0]:>13.3f} {scores[1]:>10.3f}{note}")

    attention_model = models["Bahdanau attention"]
    attention_model.eval()
    sample = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]], device=device)
    prediction, attention_maps = attention_model.greedy_decode(sample, 11)
    weights = attention_maps[0, :10].cpu()
    print(f"\nInput  {sample[0].tolist()}\nOutput {prediction[0].tolist()}")
    print("Attention weights (rows: output steps, columns: input positions):")
    for row in weights:
        print("  " + " ".join(f"{value:.2f}" for value in row.tolist()))
    focus = weights.argmax(dim=1).tolist()
    print(f"Most attended input position per output step: {focus} (reversal reads positions 9, 8, ..., 0)")

    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(weights, cmap="viridis")
    axis.set_xticks(range(10), [str(v) for v in sample[0].tolist()])
    axis.set_yticks(range(10), [str(v) for v in prediction[0, :10].tolist()])
    axis.set_xlabel("input digit")
    axis.set_ylabel("output digit")
    axis.set_title("Bahdanau attention on a reversal")
    figure.tight_layout()
    output_path = Path(__file__).parent / "attention.png"
    figure.savefig(output_path, dpi=100)
    print(f"Saved {output_path.name}")

    torch.manual_seed(0)
    queries = torch.randn(2, 3, 16)
    keys = torch.randn(2, 7, 16)
    values = torch.randn(2, 7, 16)
    weights = torch.softmax(queries @ keys.transpose(1, 2) / 16 ** 0.5, dim=-1)
    manual = weights @ values
    library = F.scaled_dot_product_attention(queries, keys, values)
    print(f"\nDot-product attention by hand vs F.scaled_dot_product_attention: max diff "
          f"{(manual - library).abs().max():.1e}")


if __name__ == "__main__":
    main()
