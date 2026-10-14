from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYOUT = [
    "S...#",
    ".#..P",
    "...#.",
    ".#...",
    "...#G",
]
ACTIONS = ["up", "right", "down", "left"]
ARROWS = ["^", ">", "v", "<"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridWorld:
    def __init__(self, layout, slip=0.2, step_reward=-0.04, goal_reward=1.0, pit_reward=-1.0):
        self.layout = layout
        self.rows, self.cols = len(layout), len(layout[0])
        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if layout[r][c] != "#"]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.n_states, self.n_actions = len(self.cells), len(ACTIONS)
        self.start = self.index[self.find("S")]
        self.terminal = {self.index[self.find("G")]: goal_reward, self.index[self.find("P")]: pit_reward}
        self.slip, self.step_reward = slip, step_reward
        self.P, self.R = self.build_model()

    def find(self, char):
        return next((r, c) for r in range(self.rows) for c in range(self.cols) if self.layout[r][c] == char)

    def move(self, cell, action):
        r, c = cell[0] + MOVES[action][0], cell[1] + MOVES[action][1]
        if 0 <= r < self.rows and 0 <= c < self.cols and self.layout[r][c] != "#":
            return (r, c)
        return cell

    def outcomes(self, action):
        side = self.slip / 2
        return [(action, 1 - self.slip), ((action + 1) % 4, side), ((action + 3) % 4, side)]

    def build_model(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        for s, cell in enumerate(self.cells):
            for a in range(self.n_actions):
                if s in self.terminal:
                    P[s, a, s] = 1.0
                    continue
                for actual, prob in self.outcomes(a):
                    s_next = self.index[self.move(cell, actual)]
                    P[s, a, s_next] += prob
                    R[s, a] += prob * (self.step_reward + self.terminal.get(s_next, 0.0))
        return P, R

    def step(self, s, a, rng):
        s_next = rng.choice(self.n_states, p=self.P[s, a])
        reward = self.step_reward + self.terminal.get(s_next, 0.0)
        return s_next, reward, s_next in self.terminal


def policy_evaluation(env, policy, gamma, tol=1e-8):
    V = np.zeros(env.n_states)
    sweeps = 0
    while True:
        sweeps += 1
        Q = env.R + gamma * env.P @ V
        V_new = (policy * Q).sum(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, sweeps
        V = V_new


def greedy(env, V, gamma):
    Q = env.R + gamma * env.P @ V
    policy = np.zeros((env.n_states, env.n_actions))
    policy[np.arange(env.n_states), Q.argmax(axis=1)] = 1.0
    return policy, Q


def value_iteration(env, gamma, tol=1e-8):
    V = np.zeros(env.n_states)
    history = []
    while True:
        Q = env.R + gamma * env.P @ V
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta < tol:
            return V, history


def policy_iteration(env, gamma):
    policy = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)
    for iteration in range(1, 100):
        V, sweeps = policy_evaluation(env, policy, gamma)
        new_policy, _ = greedy(env, V, gamma)
        changed = int((new_policy.argmax(1) != policy.argmax(1)).sum())
        print(f"  iteration {iteration}: evaluation took {sweeps:4d} sweeps, {changed:2d} states changed action")
        if np.array_equal(new_policy, policy):
            return V, policy, iteration
        policy = new_policy


def monte_carlo_return(env, policy, gamma, episodes, rng, max_steps=200):
    returns = []
    for _ in range(episodes):
        s, total, discount = env.start, 0.0, 1.0
        for _ in range(max_steps):
            a = rng.choice(env.n_actions, p=policy[s])
            s, reward, done = env.step(s, a, rng)
            total += discount * reward
            discount *= gamma
            if done:
                break
        returns.append(total)
    return np.mean(returns), np.std(returns) / np.sqrt(episodes)


def render(env, V, policy=None):
    lines = []
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            char = env.layout[r][c]
            if char == "#":
                row.append("  ####  ")
                continue
            s = env.index[(r, c)]
            mark = char if char in "GP" else (ARROWS[policy[s].argmax()] if policy is not None else " ")
            row.append(f"{V[s]:+.2f} {mark} ")
        lines.append("  " + "".join(row))
    return "\n".join(lines)


def plot_values(env, V, policy, path):
    grid = np.full((env.rows, env.cols), np.nan)
    for (r, c), s in env.index.items():
        grid[r, c] = V[s]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="viridis")
    for (r, c), s in env.index.items():
        char = env.layout[r][c]
        label = char if char in "GP" else ARROWS[policy[s].argmax()]
        ax.text(c, r, f"{label}\n{V[s]:.2f}", ha="center", va="center", color="white", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Optimal values and greedy policy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    gamma = 0.95
    env = GridWorld(LAYOUT)
    print(f"GridWorld: {env.n_states} states, {env.n_actions} actions, slip={env.slip}, gamma={gamma}")
    print(f"P has shape {env.P.shape}, every row sums to 1: {np.allclose(env.P.sum(-1), 1)}")

    print("\n1) Evaluate the uniform random policy")
    random_policy = np.full((env.n_states, env.n_actions), 0.25)
    V_random, sweeps = policy_evaluation(env, random_policy, gamma)
    print(f"  converged in {sweeps} sweeps, V(start) = {V_random[env.start]:+.3f}")
    print(render(env, V_random))

    print("\n2) Value iteration")
    V_star, history = value_iteration(env, gamma)
    pi_star, Q_star = greedy(env, V_star, gamma)
    print(f"  converged in {len(history)} sweeps, max change after 1/5/20 sweeps: "
          f"{history[0]:.3f} / {history[4]:.3f} / {history[19]:.5f}")
    print(render(env, V_star, pi_star))

    print("\n3) Policy iteration")
    V_pi, pi_pi, iterations = policy_iteration(env, gamma)
    print(f"  stable after {iterations} improvements, same policy as value iteration: "
          f"{np.array_equal(pi_pi.argmax(1), pi_star.argmax(1))}, max |V diff| = {np.abs(V_pi - V_star).max():.1e}")

    print("\n4) Check the Bellman numbers with Monte Carlo rollouts from the start state")
    for name, policy, V in [("random", random_policy, V_random), ("optimal", pi_star, V_star)]:
        mean, stderr = monte_carlo_return(env, policy, gamma, 2000, rng)
        print(f"  {name:8s} Bellman V(start) = {V[env.start]:+.3f} | Monte Carlo = {mean:+.3f} +- {stderr:.3f}")

    print("\n5) Use It: policy evaluation as one linear solve, V = (I - gamma P_pi)^-1 R_pi")
    P_pi = np.einsum("sa,sat->st", pi_star, env.P)
    R_pi = (pi_star * env.R).sum(1)
    V_solve = np.linalg.solve(np.eye(env.n_states) - gamma * P_pi, R_pi)
    print(f"  max |V_solve - V_star| = {np.abs(V_solve - V_star).max():.1e}")

    print("\n6) The reward defines the task: change the cost of living")
    for step_reward in [0.0, -0.04, -0.6]:
        env_r = GridWorld(LAYOUT, step_reward=step_reward)
        V_r, _ = value_iteration(env_r, gamma)
        pi_r, _ = greedy(env_r, V_r, gamma)
        print(f"  step reward {step_reward:+.2f}:")
        print(render(env_r, V_r, pi_r))

    path = Path(__file__).parent / "values.png"
    plot_values(env, V_star, pi_star, path)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
