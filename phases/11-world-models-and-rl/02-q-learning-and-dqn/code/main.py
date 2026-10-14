import copy
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUT = ["S...#", ".#..P", "...#.", ".#...", "...#G"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = ["^", ">", "v", "<"]


class GridWorld:
    def __init__(self, layout, slip=0.2, step_reward=-0.04):
        self.layout = layout
        self.cells = [(r, c) for r, row in enumerate(layout) for c, ch in enumerate(row) if ch != "#"]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.n_states, self.n_actions = len(self.cells), 4
        self.slip, self.step_reward = slip, step_reward
        find = lambda ch: self.index[next(cell for cell in self.cells if layout[cell[0]][cell[1]] == ch)]
        self.start = find("S")
        self.terminal = {find("G"): 1.0, find("P"): -1.0}

    def move(self, s, action):
        r, c = self.cells[s][0] + MOVES[action][0], self.cells[s][1] + MOVES[action][1]
        if 0 <= r < len(self.layout) and 0 <= c < len(self.layout[0]) and self.layout[r][c] != "#":
            return self.index[(r, c)]
        return s

    def step(self, s, action, rng):
        roll = rng.random()
        if roll < self.slip / 2:
            action = (action + 1) % 4
        elif roll < self.slip:
            action = (action + 3) % 4
        s_next = self.move(s, action)
        reward = self.step_reward + self.terminal.get(s_next, 0.0)
        return s_next, reward, s_next in self.terminal

    def optimal_q(self, gamma, sweeps=500):
        Q = np.zeros((self.n_states, self.n_actions))
        side = self.slip / 2
        for _ in range(sweeps):
            V = np.where([s in self.terminal for s in range(self.n_states)], 0.0, Q.max(1))
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    total = 0.0
                    for actual, prob in [(a, 1 - self.slip), ((a + 1) % 4, side), ((a + 3) % 4, side)]:
                        s_next = self.move(s, actual)
                        total += prob * (self.step_reward + self.terminal.get(s_next, 0.0) + gamma * V[s_next])
                    Q[s, a] = total
        return Q


def epsilon_greedy(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


def tabular_q_learning(env, episodes, gamma, alpha, rng):
    Q = np.zeros((env.n_states, env.n_actions))
    lengths = []
    for episode in range(episodes):
        epsilon = max(0.05, 1.0 - episode / (0.6 * episodes))
        s, steps = env.start, 0
        for steps in range(1, 201):
            a = epsilon_greedy(Q[s], epsilon, rng)
            s_next, reward, done = env.step(s, a, rng)
            target = reward + (0.0 if done else gamma * Q[s_next].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
            if done:
                break
        lengths.append(steps)
    return Q, lengths


class CartPole:
    gravity, cart_mass, pole_mass, half_length, force_mag, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x_limit, theta_limit, max_steps = 2.4, 12 * math.pi / 180, 200

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.astype(np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.half_length
        cos, sin = math.cos(theta), math.sin(theta)
        temp = (force + pole_mass_length * theta_dot**2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.half_length * (4.0 / 3.0 - self.pole_mass * cos**2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos / total_mass
        x, x_dot = x + self.dt * x_dot, x_dot + self.dt * x_acc
        theta, theta_dot = theta + self.dt * theta_dot, theta_dot + self.dt * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        fell = abs(x) > self.x_limit or abs(theta) > self.theta_limit
        truncated = self.steps >= self.max_steps
        return self.state.astype(np.float32), 1.0, fell, truncated


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.pointer = capacity, 0, 0

    def add(self, state, action, reward, next_state, done):
        i = self.pointer
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.dones[i] = next_state, done
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.states[idx]), torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.next_states[idx]),
                torch.from_numpy(self.dones[idx]))


def make_q_network(state_dim, n_actions, hidden=64):
    return nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_actions))


def huber(errors, delta=1.0):
    abs_errors = errors.abs()
    quadratic = 0.5 * errors**2
    linear = delta * (abs_errors - 0.5 * delta)
    return torch.where(abs_errors <= delta, quadratic, linear).mean()


def dqn_loss(online, target, batch, gamma):
    states, actions, rewards, next_states, dones = batch
    q_taken = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = target(next_states).max(dim=1).values
        td_target = rewards + gamma * (1.0 - dones) * next_q
    return huber(q_taken - td_target), q_taken, td_target


def evaluate(network, episodes, seed):
    env = CartPole(seed)
    lengths = []
    for _ in range(episodes):
        state, done, truncated, steps = env.reset(), False, False, 0
        while not (done or truncated):
            with torch.no_grad():
                action = int(network(torch.from_numpy(state)[None]).argmax())
            state, _, done, truncated = env.step(action)
            steps += 1
        lengths.append(steps)
    return float(np.mean(lengths))


def train_dqn(total_steps, gamma=0.99, batch_size=64, target_every=500, warmup=500, seed=0):
    rng = np.random.default_rng(seed)
    env = CartPole(seed)
    online = make_q_network(4, 2)
    target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=5e-4)
    buffer = ReplayBuffer(20000, 4)
    state, episode_return, returns, log = env.reset(), 0.0, [], []
    best_length, best_weights = -1.0, None
    for step in range(1, total_steps + 1):
        epsilon = max(0.05, 1.0 - step / (0.4 * total_steps))
        if rng.random() < epsilon:
            action = int(rng.integers(2))
        else:
            with torch.no_grad():
                action = int(online(torch.from_numpy(state)[None]).argmax())
        next_state, reward, done, truncated = env.step(action)
        buffer.add(state, action, reward, next_state, float(done))
        state, episode_return = next_state, episode_return + reward
        if done or truncated:
            returns.append(episode_return)
            state, episode_return = env.reset(), 0.0
        if step >= warmup:
            loss, _, _ = dqn_loss(online, target, buffer.sample(batch_size, rng), gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % target_every == 0:
            target.load_state_dict(online.state_dict())
        if step % 2000 == 0:
            recent = np.mean(returns[-10:]) if returns else 0.0
            greedy_length = evaluate(online, 5, seed=100)
            log.append((step, recent, greedy_length))
            if greedy_length >= best_length:
                best_length, best_weights = greedy_length, copy.deepcopy(online.state_dict())
            print(f"  step {step:5d} | epsilon {epsilon:.2f} | last 10 episodes {recent:6.1f} | greedy policy {greedy_length:6.1f}")
    best = make_q_network(4, 2)
    best.load_state_dict(best_weights)
    return online, best, target, buffer, returns, log


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    gamma = 0.95

    print("1) Tabular Q-learning on the slippery gridworld (no model of P is used)")
    env = GridWorld(LAYOUT)
    Q_star = env.optimal_q(gamma)
    Q, lengths = tabular_q_learning(env, 3000, gamma, alpha=0.1, rng=rng)
    visited = [s for s in range(env.n_states) if s not in env.terminal]
    agree = np.mean(Q[visited].argmax(1) == Q_star[visited].argmax(1))
    print(f"  mean episode length: first 100 = {np.mean(lengths[:100]):.1f}, last 100 = {np.mean(lengths[-100:]):.1f}")
    print(f"  max_a Q(start): learned {Q[env.start].max():+.3f} vs value iteration {Q_star[env.start].max():+.3f}")
    print(f"  greedy action matches the optimal one in {agree:.0%} of states")
    rows = []
    for r, row in enumerate(LAYOUT):
        rows.append("  " + " ".join("#" if ch == "#" else ch if ch in "GP" else ARROWS[Q[env.index[(r, c)]].argmax()]
                                    for c, ch in enumerate(row)))
    print("\n".join(rows))

    print("\n2) DQN on a from-scratch cart-pole (4 continuous state numbers, 2 actions)")
    start = time.perf_counter()
    online, best, target, buffer, returns, log = train_dqn(total_steps=18000)
    print(f"  trained in {time.perf_counter() - start:.1f}s over {len(returns)} episodes")
    print(f"  20 fresh episodes, final network:  mean length {evaluate(online, 20, seed=123):.1f} / 200")
    print(f"  20 fresh episodes, best checkpoint: mean length {evaluate(best, 20, seed=123):.1f} / 200")

    print("\n3) Use It: F.smooth_l1_loss is the same Huber loss")
    batch = buffer.sample(256, rng)
    loss, q_taken, td_target = dqn_loss(online, target, batch, 0.99)
    library_loss = F.smooth_l1_loss(q_taken, td_target)
    print(f"  hand-written Huber {loss.item():.6f} | F.smooth_l1_loss {library_loss.item():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(np.convolve(lengths, np.ones(50) / 50, mode="valid"))
    axes[0].set(title="Tabular Q-learning: episode length", xlabel="episode", ylabel="steps (50-ep avg)")
    axes[1].plot(returns, alpha=0.4, label="training episodes")
    steps, _, greedy = zip(*log)
    episode_at = np.interp(steps, np.cumsum(returns), np.arange(len(returns)))
    axes[1].plot(episode_at, greedy, "o-", label="greedy evaluation")
    axes[1].set(title="DQN on cart-pole", xlabel="episode", ylabel="return")
    axes[1].legend()
    fig.tight_layout()
    path = Path(__file__).parent / "learning_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
