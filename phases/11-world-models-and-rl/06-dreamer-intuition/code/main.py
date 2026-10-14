import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

OBS_DIM, ACT_DIM, DETER, STOCH, HIDDEN = 2, 2, 64, 8, 64


class PointMassWorld:
    def __init__(self, n_envs, generator, friction=0.8, power=0.05, noise=0.01):
        self.n_envs, self.generator = n_envs, generator
        self.friction, self.power, self.noise = friction, power, noise

    def reset(self):
        self.position = torch.rand(self.n_envs, 2, generator=self.generator) * 1.6 - 0.8
        self.velocity = torch.zeros(self.n_envs, 2)
        return self.observe()

    def observe(self):
        return self.position + self.noise * torch.randn(self.n_envs, 2, generator=self.generator)

    def step(self, action):
        self.velocity = self.friction * self.velocity + self.power * action.clamp(-1, 1)
        self.position = (self.position + self.velocity).clamp(-1, 1)
        return self.observe(), -self.position.norm(dim=1)


def collect(policy, n_episodes, length, generator):
    env = PointMassWorld(n_episodes, generator)
    obs, state = env.reset(), None
    observations, actions, rewards = [obs], [], []
    for _ in range(length):
        action, state = policy(obs, state)
        obs, reward = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
    return torch.stack(observations, 1), torch.stack(actions, 1), torch.stack(rewards, 1)


def mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, out_dim))


def gaussian(params, min_std=0.1):
    mean, raw_std = params.chunk(2, -1)
    return mean, F.softplus(raw_std) + min_std


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(STOCH + ACT_DIM, DETER)
        self.prior_net = mlp(DETER, 2 * STOCH)
        self.posterior_net = mlp(DETER + OBS_DIM, 2 * STOCH)
        self.decoder = mlp(DETER + STOCH, OBS_DIM)
        self.reward_head = mlp(DETER + STOCH, 1)

    def initial(self, batch):
        return torch.zeros(batch, DETER), torch.zeros(batch, STOCH)

    def img_step(self, h, z, action, generator=None):
        h = self.cell(torch.cat([z, action], -1), h)
        mean, std = gaussian(self.prior_net(h))
        return h, mean, std

    def obs_step(self, h, z, action, obs, generator=None):
        h, prior_mean, prior_std = self.img_step(h, z, action)
        post_mean, post_std = gaussian(self.posterior_net(torch.cat([h, obs], -1)))
        z = post_mean + post_std * torch.randn(post_mean.shape, generator=generator)
        return h, z, (prior_mean, prior_std), (post_mean, post_std)

    def observe(self, observations, actions, generator=None):
        batch, length = actions.shape[:2]
        h, z = self.initial(batch)
        prev_action = torch.zeros(batch, ACT_DIM)
        hs, zs, priors, posts = [], [], [], []
        for t in range(length + 1):
            h, z, prior, post = self.obs_step(h, z, prev_action, observations[:, t], generator)
            hs.append(h)
            zs.append(z)
            priors.append(prior)
            posts.append(post)
            if t < length:
                prev_action = actions[:, t]
        stack = lambda pairs, i: torch.stack([p[i] for p in pairs], 1)
        return torch.stack(hs, 1), torch.stack(zs, 1), (stack(priors, 0), stack(priors, 1)), (stack(posts, 0), stack(posts, 1))


def gaussian_kl(mean_q, std_q, mean_p, std_p):
    return (torch.log(std_p / std_q) + (std_q**2 + (mean_q - mean_p) ** 2) / (2 * std_p**2) - 0.5).sum(-1)


def world_model_loss(rssm, observations, actions, rewards, generator, free_nats=1.0):
    hs, zs, (prior_mean, prior_std), (post_mean, post_std) = rssm.observe(observations, actions, generator)
    features = torch.cat([hs, zs], -1)
    reconstruction = ((rssm.decoder(features) - observations) ** 2).sum(-1).mean()
    reward_loss = ((rssm.reward_head(features[:, 1:]).squeeze(-1) - rewards) ** 2).mean()
    kl = gaussian_kl(post_mean, post_std, prior_mean, prior_std).mean()
    loss = 100 * reconstruction + 100 * reward_loss + torch.clamp(kl, min=free_nats)
    return loss, reconstruction, reward_loss, kl, hs, zs


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(DETER + STOCH, 2 * ACT_DIM)

    def forward(self, features, generator=None, deterministic=False):
        mean, std = gaussian(self.net(features), min_std=0.05)
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(mean + std * torch.randn(mean.shape, generator=generator))


def lambda_returns(rewards, values, gamma, lam):
    returns = torch.zeros_like(rewards)
    next_return = values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        next_return = rewards[:, t] + gamma * ((1 - lam) * values[:, t + 1] + lam * next_return)
        returns[:, t] = next_return
    return returns


def imagine(rssm, actor, h, z, horizon, generator):
    features, actions = [torch.cat([h, z], -1)], []
    for _ in range(horizon):
        action = actor(features[-1].detach(), generator)
        h, mean, std = rssm.img_step(h, z, action)
        z = mean + std * torch.randn(mean.shape, generator=generator)
        features.append(torch.cat([h, z], -1))
        actions.append(action)
    return torch.stack(features, 1), torch.stack(actions, 1)


def train_behavior(rssm, actor, critic, actor_opt, critic_opt, start_h, start_z, generator, horizon=15, gamma=0.95, lam=0.95):
    for p in rssm.parameters():
        p.requires_grad_(False)
    features, _ = imagine(rssm, actor, start_h, start_z, horizon, generator)
    rewards = rssm.reward_head(features[:, 1:]).squeeze(-1)
    values = critic(features).squeeze(-1)
    returns = lambda_returns(rewards, values, gamma, lam)
    actor_loss = -returns.mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    critic_loss = ((critic(features[:, :-1].detach()).squeeze(-1) - returns.detach()) ** 2).mean()
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    for p in rssm.parameters():
        p.requires_grad_(True)
    return returns.mean().item(), critic_loss.item()


def make_agent_policy(rssm, actor, generator, explore):
    def policy(obs, state):
        with torch.no_grad():
            if state is None:
                h, z = rssm.initial(len(obs))
                prev_action = torch.zeros(len(obs), ACT_DIM)
            else:
                h, z, prev_action = state
            h, z, _, (post_mean, _) = rssm.obs_step(h, z, prev_action, obs, generator)
            features = torch.cat([h, post_mean], -1)
            action = actor(features, generator, deterministic=not explore)
            if explore:
                action = (action + 0.3 * torch.randn(action.shape, generator=generator)).clamp(-1, 1)
        return action, (h, z, action)
    return policy


def random_policy(generator):
    def policy(obs, state):
        return torch.rand(len(obs), ACT_DIM, generator=generator) * 2 - 1, state
    return policy


def score(policy, generator, n=64, length=40):
    _, _, rewards = collect(policy, n, length, generator)
    return rewards.mean().item(), -rewards[:, -1].mean().item()


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(0)
    rssm, actor, critic = RSSM(), Actor(), mlp(DETER + STOCH, 1)
    model_opt = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=8e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    data = collect(random_policy(generator), 64, 40, generator)
    print(f"Seed data: {data[0].shape[0]} random episodes of {data[1].shape[1]} steps. The agent only sees noisy position, never velocity.")
    print(f"  random policy:         mean reward {score(random_policy(generator), generator)[0]:+.3f}")

    history = []
    start = time.perf_counter()
    for round_index in range(5):
        observations, actions, rewards = data
        for step in range(100):
            idx = torch.randint(0, len(actions), (32,), generator=generator)
            t0 = int(torch.randint(0, actions.shape[1] - 15 + 1, (1,), generator=generator))
            loss, reconstruction, reward_loss, kl, hs, zs = world_model_loss(
                rssm, observations[idx, t0:t0 + 16], actions[idx, t0:t0 + 15], rewards[idx, t0:t0 + 15], generator)
            model_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            model_opt.step()
            start_h, start_z = hs.detach().flatten(0, 1), zs.detach().flatten(0, 1)
            imagined_return, critic_loss = train_behavior(rssm, actor, critic, actor_opt, critic_opt, start_h, start_z, generator)
        real_reward, final_distance = score(make_agent_policy(rssm, actor, generator, explore=False), generator)
        history.append(real_reward)
        print(f"  round {round_index} | obs MSE {reconstruction:.5f} | reward MSE {reward_loss:.5f} | KL {kl:5.2f} | "
              f"imagined return {imagined_return:+.3f} | real reward {real_reward:+.3f} | final dist {final_distance:.3f}")
        new = collect(make_agent_policy(rssm, actor, generator, explore=True), 16, 40, generator)
        data = tuple(torch.cat([old, fresh]) for old, fresh in zip(data, new))
    print(f"  {len(data[0])} real episodes in total, {time.perf_counter() - start:.1f}s")

    def do_nothing(obs, state):
        return torch.zeros(len(obs), ACT_DIM), state

    print("\nHow good is that? Compare with a hand-tuned PD controller (velocity from frame differences)")

    def pd_controller(obs, state):
        prev = obs if state is None else state
        velocity = obs - prev
        return (-4 * obs - 20 * velocity).clamp(-1, 1), obs

    print(f"  PD controller:         mean reward {score(pd_controller, generator)[0]:+.3f}")
    print(f"  do nothing:            mean reward {score(do_nothing, generator)[0]:+.3f}")
    print(f"  Dreamer-style agent:   mean reward {history[-1]:+.3f}")

    print("\nUse It: torch.distributions.kl_divergence matches the hand-written Gaussian KL")
    with torch.no_grad():
        hs, zs, (pm, ps), (qm, qs) = rssm.observe(data[0][:4], data[1][:4], generator)
        hand = gaussian_kl(qm, qs, pm, ps).mean()
        library = kl_divergence(Normal(qm, qs), Normal(pm, ps)).sum(-1).mean()
    print(f"  hand-written {hand:.6f} | kl_divergence {library:.6f}")

    with torch.no_grad():
        observations, actions, _ = collect(make_agent_policy(rssm, actor, generator, explore=False), 1, 40, generator)
        hs, zs, _, _ = rssm.observe(observations[:, :11], actions[:, :10], generator)
        h, z = hs[:, -1], zs[:, -1]
        dreamed = []
        for t in range(10, 40):
            h, mean, std = rssm.img_step(h, z, actions[:, t])
            z = mean
            dreamed.append(rssm.decoder(torch.cat([h, z], -1)))
        dreamed = torch.cat(dreamed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(observations[0, :, 0], observations[0, :, 1], "o-", markersize=3, label="real trajectory")
    axes[0].plot(dreamed[:, 0], dreamed[:, 1], "x--", label="open-loop dream from step 10")
    axes[0].plot(0, 0, "r*", markersize=12, label="goal")
    axes[0].set(xlim=(-1, 1), ylim=(-1, 1), title="Agent in the real world vs its imagination")
    axes[0].legend(fontsize=8)
    axes[1].plot(history, "o-")
    axes[1].set(xlabel="round", ylabel="real mean reward", title="Learning in imagination")
    fig.tight_layout()
    path = Path(__file__).parent / "dreamer.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
