import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 11
BOS = 10
RESPONSE_LENGTH = 8


def true_reward(responses):
    return (responses[:, 1:] > responses[:, :-1]).float().sum(dim=1)


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
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, block_size=RESPONSE_LENGTH + 1):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, block_size) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, VOCAB_SIZE)

    def hidden(self, ids):
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        return self.ln_final(self.blocks(x))

    def forward(self, ids):
        return self.head(self.hidden(ids))


class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(backbone.head.in_features, 1)

    def forward(self, responses):
        ids = with_bos(responses)
        return self.value_head(self.backbone.hidden(ids)[:, -1]).squeeze(-1)


def with_bos(responses):
    bos = torch.full((responses.shape[0], 1), BOS, dtype=torch.long, device=responses.device)
    return torch.cat([bos, responses], dim=1)


def sequence_log_prob(model, responses):
    ids = with_bos(responses)
    logits = model(ids[:, :-1])
    logits = logits.clone()
    logits[..., BOS] = float("-inf")
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=1)


@torch.no_grad()
def sample(model, num_samples, device, generator):
    ids = torch.full((num_samples, 1), BOS, dtype=torch.long, device=device)
    for _ in range(RESPONSE_LENGTH):
        logits = model(ids)[:, -1]
        logits[:, BOS] = float("-inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[:, 1:]


def supervised_fine_tune(model, demonstrations, steps, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        batch = demonstrations[torch.randint(len(demonstrations), (128,), generator=generator)].to(device)
        loss = -sequence_log_prob(model, batch).mean() / RESPONSE_LENGTH
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def collect_preferences(policy, num_pairs, device, generator):
    first = sample(policy, num_pairs, device, generator)
    second = sample(policy, num_pairs, device, generator)
    prob_first_wins = torch.sigmoid(2.0 * (true_reward(first) - true_reward(second)))
    first_wins = torch.rand(num_pairs, generator=generator, device=device) < prob_first_wins
    chosen = torch.where(first_wins.unsqueeze(1), first, second)
    rejected = torch.where(first_wins.unsqueeze(1), second, first)
    return chosen, rejected


def train_reward_model(reward_model, chosen, rejected, steps):
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    for _ in range(steps):
        index = torch.randint(len(chosen), (128,), generator=generator)
        margin = reward_model(chosen[index]) - reward_model(rejected[index])
        loss = -F.logsigmoid(margin).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return reward_model


@torch.no_grad()
def preference_accuracy(scorer, chosen, rejected):
    not_tied = true_reward(chosen) != true_reward(rejected)
    return (scorer(chosen[not_tied]) > scorer(rejected[not_tied])).float().mean().item()


@torch.no_grad()
def evaluate_policy(policy, reference, reward_model, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    responses = sample(policy, 1000, device, generator)
    kl = (sequence_log_prob(policy, responses) - sequence_log_prob(reference, responses)).mean().item()
    return true_reward(responses).mean().item(), reward_model(responses).mean().item(), kl, responses


def rlhf(policy, reference, reward_model, beta, steps, device):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(2)
    for _ in range(steps):
        responses = sample(policy, 256, device, generator)
        log_prob = sequence_log_prob(policy, responses)
        with torch.no_grad():
            kl_per_sample = log_prob - sequence_log_prob(reference, responses)
            total_reward = reward_model(responses) - beta * kl_per_sample
            advantage = (total_reward - total_reward.mean()) / (total_reward.std() + 1e-6)
        loss = -(advantage * log_prob).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return policy


def dpo_loss(policy, reference, chosen, rejected, beta):
    policy_chosen = sequence_log_prob(policy, chosen)
    policy_rejected = sequence_log_prob(policy, rejected)
    with torch.no_grad():
        reference_chosen = sequence_log_prob(reference, chosen)
        reference_rejected = sequence_log_prob(reference, rejected)
    logits = beta * ((policy_chosen - reference_chosen) - (policy_rejected - reference_rejected))
    return -F.logsigmoid(logits).mean(), logits


def dpo(policy, reference, chosen, rejected, beta, steps):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(3)
    for _ in range(steps):
        index = torch.randint(len(chosen), (256,), generator=generator)
        loss, _ = dpo_loss(policy, reference, chosen[index], rejected[index], beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return policy


def show(responses, count=4):
    return "  ".join("".join(str(t) for t in row.tolist()) for row in responses[:count])


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    print("task: responses are 8 digits; hidden human preference = number of rising steps (0 to 7)")

    print("\nstep 1: supervised fine-tuning on random demonstrations")
    demonstrations = torch.randint(0, 10, (5000, RESPONSE_LENGTH))
    sft = supervised_fine_tune(TinyGPT().to(device), demonstrations, steps=300, device=device)
    sft.eval()
    generator = torch.Generator(device=device).manual_seed(0)
    sft_samples = sample(sft, 1000, device, generator)
    print(f"  SFT true reward {true_reward(sft_samples).mean().item():.2f} (uniform digits: 7 * 0.45 = 3.15) | samples {show(sft_samples)}")

    print("\nstep 2: collect noisy pairwise preferences from the SFT policy (Bradley-Terry labeler)")
    chosen, rejected = collect_preferences(sft, 9000, device, generator)
    train_chosen, train_rejected = chosen[:8000], rejected[:8000]
    test_chosen, test_rejected = chosen[8000:], rejected[8000:]
    print(f"  8000 train pairs | on non-tied test pairs the noisy labeler picked the truly better response {preference_accuracy(true_reward, test_chosen, test_rejected):.1%} of the time")

    print("\nstep 3: reward model (SFT backbone + scalar head, loss -log sigmoid(r_chosen - r_rejected))")
    torch.manual_seed(1)
    reward_model = train_reward_model(RewardModel(copy.deepcopy(sft)).to(device), train_chosen, train_rejected, steps=1000)
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)
    print(f"  reward model picks the truly better response on {preference_accuracy(reward_model, test_chosen, test_rejected):.1%} of non-tied test pairs")
    probes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0], [3, 3, 3, 3, 3, 3, 3, 3]], device=device)
    print(f"  scores: rising {reward_model(probes)[0]:.2f} | falling {reward_model(probes)[1]:.2f} | flat {reward_model(probes)[2]:.2f}")

    reference = copy.deepcopy(sft)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    results = {"SFT": evaluate_policy(sft, reference, reward_model, device, seed=10)}

    print("\nstep 4: RLHF with policy gradient and a KL penalty to the SFT model")
    for beta in [0.1, 0.0]:
        torch.manual_seed(2)
        policy = rlhf(copy.deepcopy(sft), reference, reward_model, beta=beta, steps=150, device=device)
        results[f"RLHF beta={beta}"] = evaluate_policy(policy, reference, reward_model, device, seed=10)

    print("step 5: DPO directly on the preference pairs")
    for beta in [0.1, 0.5]:
        torch.manual_seed(3)
        policy = dpo(copy.deepcopy(sft), reference, train_chosen, train_rejected, beta=beta, steps=300)
        results[f"DPO beta={beta}"] = evaluate_policy(policy, reference, reward_model, device, seed=10)

    print(f"\n  {'policy':14s} {'true reward':>11s} {'RM reward':>9s} {'KL to SFT':>9s}  samples")
    for name, (reward, rm_reward, kl, responses) in results.items():
        print(f"  {name:14s} {reward:11.2f} {rm_reward:9.2f} {kl:9.2f}  {show(responses, 3)}")

    print("\nuse it: DPO is logistic regression on the implicit reward margin")
    _, logits = dpo_loss(policy, reference, test_chosen[:512], test_rejected[:512], beta=0.5)
    ours = -F.logsigmoid(logits).mean()
    library = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    print(f"  -logsigmoid loss {ours.item():.6f} | binary_cross_entropy_with_logits {library.item():.6f}")
    print(f"  DPO implicit reward ranks held-out pairs correctly {(logits > 0).float().mean().item():.1%} of the time")


if __name__ == "__main__":
    main()
