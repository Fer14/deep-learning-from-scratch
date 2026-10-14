import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class BatchedCartPole:
    gravity, cart_mass, pole_mass, half_length, force_mag, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x_limit, theta_limit = 2.4, 12 * math.pi / 180

    def __init__(self, n_envs, generator):
        self.n_envs, self.generator = n_envs, generator

    def reset(self):
        self.state = (torch.rand(self.n_envs, 4, generator=self.generator) - 0.5) * 0.1
        self.alive = torch.ones(self.n_envs, dtype=torch.bool)
        return self.state.clone()

    def step(self, actions):
        x, x_dot, theta, theta_dot = self.state.unbind(1)
        force = torch.where(actions == 1, self.force_mag, -self.force_mag)
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.half_length
        cos, sin = torch.cos(theta), torch.sin(theta)
        temp = (force + pole_mass_length * theta_dot**2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.half_length * (4.0 / 3.0 - self.pole_mass * cos**2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos / total_mass
        self.state = torch.stack([x + self.dt * x_dot, x_dot + self.dt * x_acc,
                                  theta + self.dt * theta_dot, theta_dot + self.dt * theta_acc], dim=1)
        rewards = self.alive.float()
        fell = (self.state[:, 0].abs() > self.x_limit) | (self.state[:, 2].abs() > self.theta_limit)
        self.alive = self.alive & ~fell
        return self.state.clone(), rewards


def make_mlp(out_dim, hidden=32):
    return nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))


def log_prob_of(logits, actions):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return log_probs.gather(-1, actions[..., None]).squeeze(-1)


def sample_actions(logits, generator):
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def collect(policy, env, horizon, generator):
    states, actions, rewards = [], [], []
    state = env.reset()
    for _ in range(horizon):
        with torch.no_grad():
            action = sample_actions(policy(state), generator)
        next_state, reward = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return torch.stack(states, 1), torch.stack(actions, 1), torch.stack(rewards, 1)


def rewards_to_go(rewards, gamma):
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0])
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running
    return returns


def advantage_weights(variant, states, rewards, value, gamma):
    if variant == "total return":
        return rewards_to_go(rewards, gamma)[:, :1].expand_as(rewards)
    returns = rewards_to_go(rewards, gamma)
    if variant == "reward-to-go":
        return returns
    return returns - value(states).squeeze(-1).detach()


def surrogate_loss(policy, states, actions, weights, mask):
    log_probs = log_prob_of(policy(states), actions)
    return -(log_probs * weights * mask).sum(1).mean()


def fit_value(value, optimizer, states, returns, mask, steps=1):
    for _ in range(steps):
        prediction = value(states).squeeze(-1)
        loss = (((prediction - returns) ** 2) * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def train(variant, iterations, n_envs=32, horizon=200, gamma=0.99, seed=0):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    env = BatchedCartPole(n_envs, generator)
    policy, value = make_mlp(2), make_mlp(1)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    value_opt = torch.optim.Adam(value.parameters(), lr=1e-2)
    history = []
    for iteration in range(iterations):
        states, actions, rewards = collect(policy, env, horizon, generator)
        mask = rewards
        if variant == "reward-to-go + baseline":
            fit_value(value, value_opt, states, rewards_to_go(rewards, gamma), mask)
        weights = advantage_weights(variant, states, rewards, value, gamma)
        loss = surrogate_loss(policy, states, actions, weights, mask)
        policy_opt.zero_grad()
        loss.backward()
        policy_opt.step()
        history.append(rewards.sum(1).mean().item())
        if iteration % 20 == 0 or iteration == iterations - 1:
            print(f"  [{variant:24s}] iter {iteration:3d} | mean episode length {history[-1]:6.1f}")
    return policy, history


def gradient_noise(policy, value, variant, batches, gamma=0.99):
    generator = torch.Generator().manual_seed(7)
    env = BatchedCartPole(16, generator)
    grads = []
    for _ in range(batches):
        states, actions, rewards = collect(policy, env, 200, generator)
        weights = advantage_weights(variant, states, rewards, value, gamma)
        policy.zero_grad()
        surrogate_loss(policy, states, actions, weights, rewards).backward()
        grads.append(torch.cat([p.grad.flatten() for p in policy.parameters()]))
    grads = torch.stack(grads)
    return (grads.std(0).norm() / grads.mean(0).norm()).item()


def variance_experiment(seed=3, gamma=0.99):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    policy, value = make_mlp(2), make_mlp(1)
    env = BatchedCartPole(128, generator)
    states, actions, rewards = collect(policy, env, 200, generator)
    value_opt = torch.optim.Adam(value.parameters(), lr=1e-2)
    loss = fit_value(value, value_opt, states, rewards_to_go(rewards, gamma), rewards, steps=300)
    print(f"  fixed random policy (mean length {rewards.sum(1).mean():.1f}), value net fitted, MSE {loss:.2f}")
    for variant in ["total return", "reward-to-go", "reward-to-go + baseline"]:
        print(f"  {variant:24s}: gradient noise-to-signal ratio {gradient_noise(policy, value, variant, 30):.2f}")


def main():
    torch.set_num_threads(1)
    iterations = 120
    results = {}
    print("Training REINFORCE with three choices of weight for log pi(a|s)")
    for variant in ["total return", "reward-to-go", "reward-to-go + baseline"]:
        start = time.perf_counter()
        policy, history = train(variant, iterations)
        results[variant] = (policy, history)
        print(f"  -> mean length over the last 10 iterations {np.mean(history[-10:]):6.1f}, "
              f"first iteration above 195: {next((i for i, h in enumerate(history) if h > 195), None)}, "
              f"{time.perf_counter() - start:.1f}s\n")

    print("Variance of the gradient estimate, same policy, 30 batches of 16 episodes")
    variance_experiment()
    print()

    print("Use It: torch.distributions.Categorical gives the same log-probabilities")
    policy = results["reward-to-go + baseline"][0]
    generator = torch.Generator().manual_seed(1)
    states = BatchedCartPole(5, generator).reset()
    logits = policy(states)
    actions = sample_actions(logits, generator)
    dist = Categorical(logits=logits)
    print(f"  hand-written log pi(a|s): {log_prob_of(logits, actions).detach().numpy().round(4)}")
    print(f"  Categorical.log_prob:     {dist.log_prob(actions).detach().numpy().round(4)}")
    print(f"  policy entropy per state: {dist.entropy().detach().numpy().round(3)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, (_, history) in results.items():
        ax.plot(history, label=variant)
    ax.set(xlabel="iteration (32 episodes each)", ylabel="mean episode length", title="REINFORCE on cart-pole")
    ax.legend()
    fig.tight_layout()
    path = Path(__file__).parent / "policy_gradient.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
