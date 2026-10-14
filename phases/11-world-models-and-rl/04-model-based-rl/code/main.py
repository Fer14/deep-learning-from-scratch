import copy
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

X_LIMIT, THETA_LIMIT = 2.4, 12 * math.pi / 180


def cartpole_step(state, action):
    gravity, cart_mass, pole_mass, half_length, force_mag, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x, x_dot, theta, theta_dot = state.unbind(-1)
    force = torch.where(action == 1, force_mag, -force_mag)
    total_mass = cart_mass + pole_mass
    cos, sin = torch.cos(theta), torch.sin(theta)
    temp = (force + pole_mass * half_length * theta_dot**2 * sin) / total_mass
    theta_acc = (gravity * sin - cos * temp) / (half_length * (4.0 / 3.0 - pole_mass * cos**2 / total_mass))
    x_acc = temp - pole_mass * half_length * theta_acc * cos / total_mass
    return torch.stack([x + dt * x_dot, x_dot + dt * x_acc, theta + dt * theta_dot, theta_dot + dt * theta_acc], -1)


def has_fallen(state):
    return (state[..., 0].abs() > X_LIMIT) | (state[..., 2].abs() > THETA_LIMIT)


def collect_random(n_envs, steps, generator):
    scale = torch.tensor([1.5, 1.0, 0.15, 1.0])
    state = (torch.rand(n_envs, 4, generator=generator) * 2 - 1) * scale
    states, actions, next_states = [], [], []
    for _ in range(steps):
        action = torch.randint(0, 2, (n_envs,), generator=generator)
        next_state = cartpole_step(state, action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        state = torch.where(has_fallen(next_state)[:, None], (torch.rand(n_envs, 4, generator=generator) * 2 - 1) * scale, next_state)
    return torch.cat(states), torch.cat(actions), torch.cat(next_states)


class DynamicsModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        self.register_buffer("state_mean", torch.zeros(4))
        self.register_buffer("state_std", torch.ones(4))
        self.register_buffer("delta_mean", torch.zeros(4))
        self.register_buffer("delta_std", torch.ones(4))

    def set_normalization(self, states, next_states):
        deltas = next_states - states
        self.state_mean.copy_(states.mean(0))
        self.state_std.copy_(states.std(0) + 1e-6)
        self.delta_mean.copy_(deltas.mean(0))
        self.delta_std.copy_(deltas.std(0) + 1e-6)

    def forward(self, state, action):
        inputs = torch.cat([(state - self.state_mean) / self.state_std, (action.float()[..., None] * 2 - 1)], -1)
        return self.net(inputs)

    def predict(self, state, action):
        return state + self(state, action) * self.delta_std + self.delta_mean


def train_model(model, states, actions, next_states, steps, generator, batch_size=256):
    model.set_normalization(states, next_states)
    targets = ((next_states - states) - model.delta_mean) / model.delta_std
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(steps):
        idx = torch.randint(0, len(states), (batch_size,), generator=generator)
        loss = ((model(states[idx], actions[idx]) - targets[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    return loss.item()


def rollout_error(model, generator, horizons=(1, 5, 10, 20, 30), n=200):
    state = (torch.rand(n, 4, generator=generator) - 0.5) * 0.1
    actions = torch.randint(0, 2, (n, max(horizons)), generator=generator)
    true_state, model_state, errors = state, state, {}
    with torch.no_grad():
        for t in range(max(horizons)):
            true_state = cartpole_step(true_state, actions[:, t])
            model_state = model.predict(model_state, actions[:, t])
            if t + 1 in horizons:
                errors[t + 1] = (true_state[:, 2] - model_state[:, 2]).abs().mean().item() * 180 / math.pi
    return errors


def planning_reward(state):
    alive = (~has_fallen(state)).float()
    return alive * (1.0 - (state[..., 2] / THETA_LIMIT) ** 2 - 0.1 * (state[..., 0] / X_LIMIT) ** 2)


def cem_plan(dynamics, state, generator, horizon=15, candidates=128, elites=16, iterations=2):
    probs = torch.full((horizon,), 0.5)
    for _ in range(iterations):
        actions = (torch.rand(candidates, horizon, generator=generator) < probs).long()
        sim = state.expand(candidates, 4)
        total, alive = torch.zeros(candidates), torch.ones(candidates)
        for t in range(horizon):
            sim = dynamics(sim, actions[:, t])
            alive = alive * (~has_fallen(sim)).float()
            total = total + alive * planning_reward(sim)
        best = actions[total.topk(elites).indices].float()
        probs = 0.2 * probs + 0.8 * best.mean(0)
    return int(probs[0] > 0.5)


def run_episodes(choose_action, episodes, generator, max_steps=200):
    lengths, trajectories = [], []
    for _ in range(episodes):
        state = (torch.rand(4, generator=generator) - 0.5) * 0.1
        visited = []
        for step in range(max_steps):
            with torch.no_grad():
                action = choose_action(state)
            next_state = cartpole_step(state, torch.tensor(action))
            visited.append((state, action, next_state))
            state = next_state
            if has_fallen(state):
                break
        lengths.append(len(visited))
        trajectories.append(visited)
    return sum(lengths) / len(lengths), trajectories


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(0)

    print("1) Collect random-action data from the real environment")
    states, actions, next_states = collect_random(n_envs=20, steps=20, generator=generator)
    print(f"  {len(states)} transitions (DQN in the previous lesson used 18000)")

    print("\n2) Fit a neural dynamics model s_t+1 = s_t + f(s_t, a_t)")
    model = DynamicsModel()
    start = time.perf_counter()
    loss = train_model(model, states, actions, next_states, steps=1500, generator=generator)
    print(f"  final normalized MSE {loss:.4f} ({time.perf_counter() - start:.1f}s)")
    errors = rollout_error(model, generator)
    print("  pole-angle error when the model is rolled out open-loop on its own predictions:")
    for horizon, error in errors.items():
        print(f"    after {horizon:2d} steps: {error:6.3f} degrees")

    print("\n3) Plan with the cross-entropy method (CEM) inside the model, re-plan every step (MPC)")
    policies = {
        "random actions": lambda s: int(torch.randint(0, 2, (1,), generator=generator)),
        "CEM on the true simulator": lambda s: cem_plan(cartpole_step, s, generator),
        "CEM on the learned model": lambda s: cem_plan(model.predict, s, generator),
    }
    results = {}
    for name, policy in policies.items():
        start = time.perf_counter()
        mean_length, trajectories = run_episodes(policy, 3, generator)
        results[name] = (mean_length, trajectories)
        print(f"  {name:26s}: mean episode length {mean_length:6.1f} / 200 ({time.perf_counter() - start:.1f}s)")

    print("\n4) How much data does the model need? Fewer random transitions, same planner")
    data_sweep = {len(states): (errors, results["CEM on the learned model"][0])}
    for n_envs, steps in [(1, 8), (2, 8), (4, 10)]:
        small = collect_random(n_envs, steps, generator)
        small_model = DynamicsModel()
        train_model(small_model, *small, steps=800, generator=generator)
        small_errors = rollout_error(small_model, generator)
        small_length, _ = run_episodes(lambda s: cem_plan(small_model.predict, s, generator), 2, generator)
        data_sweep[len(small[0])] = (small_errors, small_length)
    for size, (size_errors, length) in sorted(data_sweep.items()):
        print(f"  {size:4d} transitions: 20-step angle error {size_errors[20]:7.3f} deg, CEM episode length {length:6.1f}")

    print("\n5) Use It: an ensemble evaluated in one call with torch.func.vmap")
    members = [model]
    for seed in [1, 2, 3]:
        member = DynamicsModel()
        train_model(member, states, actions, next_states, 1000, torch.Generator().manual_seed(seed))
        members.append(member)
    params, buffers = stack_module_state(members)
    base = copy.deepcopy(members[0]).to("meta")

    def call_member(p, b, s, a):
        return functional_call(base, (p, b), (s, a), {}) * b["delta_std"] + b["delta_mean"] + s

    ensemble = vmap(call_member, in_dims=(0, 0, None, None))
    probe_in = (torch.rand(256, 4, generator=generator) - 0.5) * 0.2
    probe_out = probe_in + torch.tensor([0.0, 4.0, 0.0, 6.0])
    probe_actions = torch.randint(0, 2, (256,), generator=generator)
    with torch.no_grad():
        loop = torch.stack([m.predict(probe_in, probe_actions) for m in members])
        batched = ensemble(params, buffers, probe_in, probe_actions)
        spread_in = batched.std(0).mean().item()
        spread_out = ensemble(params, buffers, probe_out, probe_actions).std(0).mean().item()
    print(f"  vmap output matches a Python loop over members: {torch.allclose(loop, batched, atol=1e-5)}")
    print(f"  ensemble disagreement on familiar states {spread_in:.5f} vs unseen fast-spinning states {spread_out:.5f}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for size, (size_errors, _) in sorted(data_sweep.items()):
        ax.plot(list(size_errors), list(size_errors.values()), "o-", label=f"{size} transitions")
    ax.set_yscale("log")
    ax.set(xlabel="open-loop horizon (steps)", ylabel="mean |angle error| (deg)", title="Compounding model error")
    ax.legend()
    fig.tight_layout()
    path = Path(__file__).parent / "model_error.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
