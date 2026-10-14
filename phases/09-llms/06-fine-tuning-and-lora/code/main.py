import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

CORPUS = """the old harbor town woke up before the sun. the baker lit the oven and the smell of warm bread drifted down to the water. the fishermen pulled their boats into the cold sea and the gulls followed them, crying for scraps.
on the hill above the town stood a lighthouse. the keeper was a quiet woman who wrote everything in a small blue notebook. she wrote the wind, the waves, the color of the sky and the names of the boats that came home.
every morning the keeper climbed the stairs of the lighthouse. there were one hundred and twelve stairs, and she counted them every time. at the top she cleaned the great lamp and looked out at the sea.
the baker had a son who wanted to be a sailor. he watched the boats leave the harbor and he dreamed of islands far away. his father wanted him to bake bread, but the boy wanted to read the stars.
one evening a storm came from the north. the wind pushed the waves over the wall of the harbor and the rain fell like stones. the keeper lit the great lamp and the light turned slowly over the dark water.
a small boat was still out at sea. the fishermen on the dock could not see it, but the keeper saw a tiny light between the waves. she wrote in her notebook: a boat in the storm, north of the rocks, one light.
the baker's son ran up the hill to the lighthouse. he climbed the one hundred and twelve stairs and stood next to the keeper. together they watched the tiny light move closer to the rocks.
the keeper turned the great lamp toward the boat. the light showed the safe path between the rocks, and the boat followed it slowly into the harbor. the fishermen cheered and the gulls flew in circles above the dock.
the next morning the sea was calm and the sky was clear. the baker made a large loaf of bread for the keeper and his son carried it up the hill. the keeper smiled and wrote in her notebook: calm sea, clear sky, warm bread.
after that day the boy came to the lighthouse every evening. the keeper taught him the names of the stars and the shapes of the clouds. she taught him how to read the wind and how to clean the great lamp.
the boy still baked bread in the morning, but in the evening he climbed the stairs and watched the sea. he wrote his own notes in a small green notebook: the wind from the west, the waves are low, three boats came home.
years later the keeper grew old and her hands began to shake. the boy, now a young man, became the new keeper of the lighthouse. he kept the blue notebook next to his green one and read it on quiet nights.
the town changed slowly. new houses were built on the hill and new boats came to the harbor. but the lighthouse stayed the same, and every night the light turned over the dark water.
the new keeper had a daughter who loved numbers. she counted the stairs, the boats, the gulls and the waves. she wrote the numbers in a small red notebook and drew lines between them.
one winter the daughter noticed that the storms came more often when the wind turned from the north. she showed her father the lines in her notebook, and he showed her the old blue notebook. the same pattern was there, written many years before.
the keeper and his daughter wrote a letter to the fishermen. when the wind turns north, they wrote, stay in the harbor and wait for the light. the fishermen read the letter and pinned it on the wall of the bakery.
the bakery still smells of warm bread in the morning. the gulls still cry above the dock. and on the hill the lighthouse still stands, with three small notebooks on the table at the top of the stairs.
"""

WINDS = ["north", "south", "east", "west"]
WAVES = ["low", "high", "calm", "rough"]
COUNTS = ["one boat", "two boats", "three boats", "no boats"]
SKIES = ["clear sky", "grey sky", "rain", "fog"]


def generate_logbook(num_entries, seed):
    rng = np.random.default_rng(seed)
    lines = []
    for day in range(1, num_entries + 1):
        lines.append(
            f"day {day % 30 + 1}: wind from the {rng.choice(WINDS)}, waves {rng.choice(WAVES)}, "
            f"{rng.choice(SKIES)}, {rng.choice(COUNTS)} came home."
        )
    return "\n".join(lines) + "\n"


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

    def forward(self, x):
        batch, length, embed_dim = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(batch, length, embed_dim)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.out = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, block_size) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, ids):
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.ln_final(self.blocks(x)))


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.T @ self.lora_b.T) * self.scale

    def merged(self):
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None).to(self.base.weight.device)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.scale * self.lora_b @ self.lora_a)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


LORA_TARGETS = ["attn.qkv", "attn.proj", "fc", "out"]


def get_submodule_parent(model, name):
    parent_name, _, child_name = name.rpartition(".")
    return model.get_submodule(parent_name), child_name


def add_lora(model, rank, alpha):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block_index in range(len(model.blocks)):
        for target in LORA_TARGETS:
            parent, child = get_submodule_parent(model, f"blocks.{block_index}.{target}")
            setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha))
    return model


def merge_lora(model):
    for block_index in range(len(model.blocks)):
        for target in LORA_TARGETS:
            parent, child = get_submodule_parent(model, f"blocks.{block_index}.{target}")
            setattr(parent, child, getattr(parent, child).merged())
    return model


def get_batch(data, block_size, batch_size, device, generator):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts]).to(device)
    return x, y


def train(model, data, steps, lr, device, seed):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        x, y = get_batch(data, model.block_size, 32, device, generator)
        loss = F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, data, device):
    generator = torch.Generator().manual_seed(99)
    losses = []
    for _ in range(10):
        x, y = get_batch(data, model.block_size, 32, device, generator)
        losses.append(F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1)).item())
    return float(np.mean(losses))


@torch.no_grad()
def greedy(model, prompt, length, char_to_id, chars, device):
    ids = torch.tensor([[char_to_id[c] for c in prompt]], device=device)
    for _ in range(length):
        next_id = model(ids[:, -model.block_size :])[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return "".join(chars[i] for i in ids[0].tolist())


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LoRAParametrization(nn.Module):
    def __init__(self, lora_a, lora_b, scale):
        super().__init__()
        self.lora_a = nn.Parameter(lora_a.detach().clone())
        self.lora_b = nn.Parameter(lora_b.detach().clone())
        self.scale = scale

    def forward(self, weight):
        return weight + self.scale * self.lora_b @ self.lora_a


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logbook = generate_logbook(400, seed=0)
    chars = sorted(set(CORPUS) | set(logbook))
    char_to_id = {c: i for i, c in enumerate(chars)}
    story = torch.tensor([char_to_id[c] for c in CORPUS], dtype=torch.long)
    task = torch.tensor([char_to_id[c] for c in logbook], dtype=torch.long)
    task_train, task_val = task[: int(0.8 * len(task))], task[int(0.8 * len(task)) :]
    print(f"device {device} | story {len(story)} chars | logbook {len(task)} chars | vocab {len(chars)}")
    print(f"logbook example: {logbook.splitlines()[0]!r}")

    print("\nstep 1: pretrain on the story")
    base = GPT(len(chars), 48, 64, 4, 2).to(device)
    train(base, story, steps=400, lr=3e-3, device=device, seed=0)
    total_params = sum(p.numel() for p in base.parameters())
    print(f"  base model {total_params:,} parameters")
    print(f"  story loss {evaluate(base, story, device):.3f} | logbook val loss {evaluate(base, task_val, device):.3f}")
    print(f"  sample: {greedy(base, 'day 5: ', 60, char_to_id, chars, device)!r}")

    print("\nstep 2: full fine-tuning")
    full = copy.deepcopy(base)
    train(full, task_train, steps=300, lr=1e-3, device=device, seed=1)
    print(f"  trainable {count_trainable(full):,} ({count_trainable(full) / total_params:.0%})")
    print(f"  story loss {evaluate(full, story, device):.3f} | logbook val loss {evaluate(full, task_val, device):.3f}")
    print(f"  sample: {greedy(full, 'day 5: ', 60, char_to_id, chars, device)!r}")

    print("\nstep 3: LoRA fine-tuning (rank 4 on attention and MLP linears)")
    lora = add_lora(copy.deepcopy(base), rank=4, alpha=8)
    print(f"  output unchanged at init (B = 0): {torch.allclose(lora(task_val[:48].unsqueeze(0).to(device)), base(task_val[:48].unsqueeze(0).to(device)))}")
    train(lora, task_train, steps=300, lr=1e-3, device=device, seed=1)
    print(f"  trainable {count_trainable(lora):,} ({count_trainable(lora) / total_params:.1%})")
    print(f"  story loss {evaluate(lora, story, device):.3f} | logbook val loss {evaluate(lora, task_val, device):.3f}")
    print(f"  sample: {greedy(lora, 'day 5: ', 60, char_to_id, chars, device)!r}")

    print("\nstep 4: rank sweep")
    for rank in [1, 16]:
        model = add_lora(copy.deepcopy(base), rank=rank, alpha=2 * rank)
        train(model, task_train, steps=300, lr=1e-3, device=device, seed=1)
        print(f"  rank {rank:2d} | trainable {count_trainable(model):6,} | story {evaluate(model, story, device):.3f} | logbook val {evaluate(model, task_val, device):.3f}")

    print("\nstep 5: merge the adapters into the weights")
    probe = task_val[:48].unsqueeze(0).to(device)
    with torch.no_grad():
        before = lora(probe)
        adapter_a = lora.blocks[0].attn.qkv.lora_a.detach().clone()
        adapter_b = lora.blocks[0].attn.qkv.lora_b.detach().clone()
        adapter_scale = lora.blocks[0].attn.qkv.scale
        merged = merge_lora(copy.deepcopy(lora))
        after = merged(probe)
    print(f"  max |adapter output - merged output| = {(before - after).abs().max().item():.2e}")
    print(f"  merged model has {sum(p.numel() for p in merged.parameters()):,} parameters, same as the base")

    print("\nuse it: torch.nn.utils.parametrize applies the same low-rank update to an existing nn.Linear")
    layer = copy.deepcopy(base.blocks[0].attn.qkv)
    parametrize.register_parametrization(layer, "weight", LoRAParametrization(adapter_a, adapter_b, adapter_scale))
    x = torch.randn(4, 64, device=device)
    with torch.no_grad():
        ours = lora.blocks[0].attn.qkv(x)
        library = layer(x)
    print(f"  max |LoRALinear - parametrized nn.Linear| = {(ours - library).abs().max().item():.2e}")
    print(f"  parametrized weight shape {tuple(layer.weight.shape)}, original stored at layer.parametrizations.weight.original")


if __name__ == "__main__":
    main()
