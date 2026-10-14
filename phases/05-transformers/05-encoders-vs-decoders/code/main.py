import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

PLUS, EQUALS, BLANK, START = 10, 11, 12, 13
VOCAB_SIZE = 14
PROBLEM_LENGTH = 6
ANSWER_LENGTH = 3


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key_value = nn.Linear(d_model, 2 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def split(self, x):
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.d_head).transpose(1, 2)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        q = self.split(self.query(x))
        k, v = (self.split(t) for t in self.key_value(context).chunk(2, dim=-1))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ v
        return self.output(attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, cross_attention=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = Attention(d_model, num_heads)
        self.cross_norm = nn.LayerNorm(d_model) if cross_attention else None
        self.cross_attention = Attention(d_model, num_heads) if cross_attention else None
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, mask=None, memory=None):
        x = x + self.self_attention(self.norm1(x), mask=mask)
        if self.cross_attention is not None:
            x = x + self.cross_attention(self.cross_norm(x), context=memory)
        return x + self.mlp(self.norm2(x))


class Stack(nn.Module):
    def __init__(self, length, d_model=64, num_heads=4, num_layers=2, cross_attention=False):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.position = nn.Parameter(torch.randn(length, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, num_heads, cross_attention) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, mask=None, memory=None):
        x = self.embed(tokens) + self.position[:tokens.shape[1]]
        for block in self.blocks:
            x = block(x, mask, memory)
        return self.norm(x)


def causal_mask(length, device):
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))


class DecoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = Stack(PROBLEM_LENGTH + ANSWER_LENGTH)
        self.head = nn.Linear(64, VOCAB_SIZE)

    def forward(self, tokens):
        return self.head(self.stack(tokens, mask=causal_mask(tokens.shape[1], tokens.device)))

    def loss(self, problems, answers):
        sequence = torch.cat([problems, answers], dim=1)
        logits = self(sequence[:, :-1])[:, PROBLEM_LENGTH - 1:]
        return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), answers.reshape(-1))

    @torch.no_grad()
    def solve(self, problems):
        sequence = problems
        for _ in range(ANSWER_LENGTH):
            next_token = self(sequence)[:, -1].argmax(dim=-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
        return sequence[:, PROBLEM_LENGTH:]


class EncoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = Stack(PROBLEM_LENGTH + ANSWER_LENGTH)
        self.head = nn.Linear(64, VOCAB_SIZE)

    def forward(self, tokens):
        return self.head(self.stack(tokens))

    def fill(self, problems):
        blanks = torch.full((problems.shape[0], ANSWER_LENGTH), BLANK, device=problems.device)
        return self(torch.cat([problems, blanks], dim=1))[:, PROBLEM_LENGTH:]

    def loss(self, problems, answers):
        return F.cross_entropy(self.fill(problems).reshape(-1, VOCAB_SIZE), answers.reshape(-1))

    @torch.no_grad()
    def solve(self, problems):
        return self.fill(problems).argmax(dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Stack(PROBLEM_LENGTH)
        self.decoder = Stack(ANSWER_LENGTH + 1, cross_attention=True)
        self.head = nn.Linear(64, VOCAB_SIZE)

    def forward(self, problems, decoder_input):
        memory = self.encoder(problems)
        mask = causal_mask(decoder_input.shape[1], decoder_input.device)
        return self.head(self.decoder(decoder_input, mask=mask, memory=memory))

    def loss(self, problems, answers):
        start = torch.full((problems.shape[0], 1), START, device=problems.device)
        logits = self(problems, torch.cat([start, answers[:, :-1]], dim=1))
        return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), answers.reshape(-1))

    @torch.no_grad()
    def solve(self, problems):
        decoder_input = torch.full((problems.shape[0], 1), START, device=problems.device)
        for _ in range(ANSWER_LENGTH):
            next_token = self(problems, decoder_input)[:, -1].argmax(dim=-1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        return decoder_input[:, 1:]


class LibraryEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 64)
        self.source_position = nn.Parameter(torch.randn(PROBLEM_LENGTH, 64) * 0.02)
        self.target_position = nn.Parameter(torch.randn(ANSWER_LENGTH + 1, 64) * 0.02)
        self.transformer = nn.Transformer(64, 4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256,
                                          dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.head = nn.Linear(64, VOCAB_SIZE)

    def forward(self, problems, decoder_input):
        source = self.embed(problems) + self.source_position
        target = self.embed(decoder_input) + self.target_position[:decoder_input.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1], device=problems.device)
        return self.head(self.transformer(source, target, tgt_mask=mask, tgt_is_causal=True))

    loss = EncoderDecoder.loss
    solve = EncoderDecoder.solve


def make_dataset(device):
    a, b = torch.meshgrid(torch.arange(100), torch.arange(100), indexing="ij")
    a, b = a.flatten(), b.flatten()
    total = a + b
    problems = torch.stack([a // 10, a % 10, torch.full_like(a, PLUS), b // 10, b % 10, torch.full_like(a, EQUALS)], dim=1)
    answers = torch.stack([total // 100, (total // 10) % 10, total % 10], dim=1)
    order = torch.randperm(len(a), generator=torch.Generator().manual_seed(0))
    split = int(0.8 * len(a))
    train_idx, test_idx = order[:split], order[split:]
    return (problems[train_idx].to(device), answers[train_idx].to(device),
            problems[test_idx].to(device), answers[test_idx].to(device))


def train(model, problems, answers, steps, generator, batch_size=128):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 100) * (1 - step / steps))
    for _ in range(steps):
        batch = torch.randint(0, len(problems), (batch_size,), generator=generator).to(problems.device)
        loss = model.loss(problems[batch], answers[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    return loss.item()


def to_text(problem, answer):
    symbols = "0123456789+="
    return "".join(symbols[t] for t in problem.tolist()) + "".join(str(t) for t in answer.tolist())


def main():
    warnings.filterwarnings("ignore", message="enable_nested_tensor")
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(0)
    train_problems, train_answers, test_problems, test_answers = make_dataset(device)
    print(f"Two-digit addition: {len(train_problems)} training problems, {len(test_problems)} held-out problems")
    print(f"  example: {to_text(train_problems[0], train_answers[0])}")

    print("\nWhat each position can see (1 = attends):")
    print("  encoder (bidirectional)      decoder (causal)")
    full = torch.ones(5, 5, dtype=torch.int)
    causal = causal_mask(5, "cpu").int()
    for row in range(5):
        print("  " + " ".join(map(str, full[row].tolist())) + " " * 20 + " ".join(map(str, causal[row].tolist())))

    torch.manual_seed(0)
    decoder = DecoderOnly().eval()
    encoder = EncoderOnly().eval()
    sequence = train_problems[:1].cpu().repeat(2, 1)
    sequence = torch.cat([sequence, torch.full((2, 3), BLANK)], dim=1)
    sequence[1, 7:] = torch.tensor([5, 5])
    with torch.no_grad():
        decoder_change = (decoder.cpu()(sequence)[0] - decoder(sequence)[1]).abs().amax(dim=-1)
        encoder_change = (encoder.cpu()(sequence)[0] - encoder(sequence)[1]).abs().amax(dim=-1)
    print("\nChange the last two tokens. Largest output change at each of the 9 positions:")
    print("  decoder: " + " ".join(f"{v:.2f}" for v in decoder_change.tolist()))
    print("  encoder: " + " ".join(f"{v:.2f}" for v in encoder_change.tolist()))

    print(f"\n  {'architecture':<33} {'params':>8} {'final loss':>11} {'test exact':>11} {'seconds':>8}")
    for name, build in [("decoder-only (GPT-style)", DecoderOnly), ("encoder-only (BERT-style)", EncoderOnly),
                        ("encoder-decoder (yours)", EncoderDecoder), ("encoder-decoder (nn.Transformer)", LibraryEncoderDecoder)]:
        torch.manual_seed(0)
        model = build().to(device)
        start = time.perf_counter()
        final_loss = train(model, train_problems, train_answers, 1000, generator)
        model.eval()
        predictions = model.solve(test_problems)
        exact = (predictions == test_answers).all(dim=1).float().mean().item()
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:<33} {params:>8,} {final_loss:>11.4f} {exact:>11.3f} {time.perf_counter() - start:>8.1f}")
        if name.startswith("decoder-only"):
            examples = model.solve(test_problems[:3])
            shown = ", ".join(to_text(p, a) for p, a in zip(test_problems[:3], examples))
            print(f"    generated: {shown}")


if __name__ == "__main__":
    main()
