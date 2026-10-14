import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

SIZE, N_ACTIONS, GOAL = 16, 5, torch.tensor([0.5, 0.5])
PUSHES = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])


class BallWorld:
    def __init__(self, n_envs, generator, accel=0.01, friction=0.8):
        self.n_envs, self.generator, self.accel, self.friction = n_envs, generator, accel, friction
        coords = (torch.arange(SIZE) + 0.5) / SIZE
        self.grid_y, self.grid_x = torch.meshgrid(coords, coords, indexing="ij")

    def reset(self):
        self.position = torch.rand(self.n_envs, 2, generator=self.generator) * 0.8 + 0.1
        self.velocity = (torch.rand(self.n_envs, 2, generator=self.generator) - 0.5) * 0.04
        return self.render()

    def step(self, actions):
        self.velocity = self.friction * self.velocity + self.accel * PUSHES[actions]
        self.position = self.position + self.velocity
        low, high = self.position < 0.05, self.position > 0.95
        self.velocity = torch.where(low | high, -self.velocity, self.velocity)
        self.position = self.position.clamp(0.05, 0.95)
        return self.render(), self.reward()

    def reward(self):
        return -(self.position - GOAL).norm(dim=1)

    def render(self):
        x, y = self.position[:, 0, None, None], self.position[:, 1, None, None]
        image = torch.exp(-((self.grid_x - x) ** 2 + (self.grid_y - y) ** 2) / (2 * 0.06**2))
        return image.flatten(1)


def collect_rollouts(n_episodes, length, generator):
    env = BallWorld(n_episodes, generator)
    frames, actions, rewards = [env.reset()], [], []
    action = torch.randint(0, N_ACTIONS, (n_episodes,), generator=generator)
    for _ in range(length):
        change = torch.rand(n_episodes, generator=generator) < 0.3
        action = torch.where(change, torch.randint(0, N_ACTIONS, (n_episodes,), generator=generator), action)
        frame, reward = env.step(action)
        frames.append(frame)
        actions.append(action)
        rewards.append(reward)
    return torch.stack(frames, 1), torch.stack(actions, 1), torch.stack(rewards, 1)


class VAE(nn.Module):
    def __init__(self, latent=4, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(SIZE * SIZE, hidden), nn.ReLU(), nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, SIZE * SIZE))

    def encode(self, frames):
        mean, log_var = self.encoder(frames).chunk(2, dim=-1)
        return mean, log_var

    def forward(self, frames, generator=None):
        mean, log_var = self.encode(frames)
        z = mean + torch.exp(0.5 * log_var) * torch.randn(mean.shape, generator=generator)
        return self.decoder(z), mean, log_var


def vae_loss(vae, frames, generator, beta=1.0):
    logits, mean, log_var = vae(frames, generator)
    reconstruction = F.binary_cross_entropy_with_logits(logits, frames, reduction="sum") / len(frames)
    kl = -0.5 * (1 + log_var - mean**2 - log_var.exp()).sum() / len(frames)
    return reconstruction + beta * kl, reconstruction, kl


class MDNRNN(nn.Module):
    def __init__(self, latent=4, hidden=64, mixtures=3):
        super().__init__()
        self.latent, self.mixtures = latent, mixtures
        self.rnn = nn.GRU(latent + N_ACTIONS, hidden, batch_first=True)
        self.mdn = nn.Linear(hidden, mixtures * (1 + 2 * latent))
        self.reward_head = nn.Linear(hidden + latent, 1)

    def mixture_params(self, h):
        raw = self.mdn(h)
        logit_pi = raw[..., :self.mixtures]
        mu, log_sigma = raw[..., self.mixtures:].reshape(*h.shape[:-1], self.mixtures, 2, self.latent).unbind(-2)
        return logit_pi, mu, log_sigma.clamp(-5, 2)

    def forward(self, z, actions, h0=None):
        inputs = torch.cat([z, F.one_hot(actions, N_ACTIONS).float()], -1)
        h, h_last = self.rnn(inputs, h0)
        return h, h_last

    def predict_reward(self, h, z_next):
        return self.reward_head(torch.cat([h, z_next], -1)).squeeze(-1)


def mdn_nll(logit_pi, mu, log_sigma, target):
    target = target.unsqueeze(-2)
    log_normal = -0.5 * ((target - mu) / log_sigma.exp()) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
    log_mix = torch.log_softmax(logit_pi, -1) + log_normal.sum(-1)
    return -torch.logsumexp(log_mix, -1).mean()


def sample_mdn(logit_pi, mu, log_sigma, temperature, generator):
    probs = torch.softmax(logit_pi / temperature, -1)
    k = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    chosen_mu = mu.gather(-2, k[:, None, None].expand(-1, 1, mu.shape[-1])).squeeze(-2)
    chosen_sigma = log_sigma.gather(-2, k[:, None, None].expand(-1, 1, mu.shape[-1])).squeeze(-2).exp()
    return chosen_mu + chosen_sigma * math.sqrt(temperature) * torch.randn(chosen_mu.shape, generator=generator)


class LinearController:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.n_params = n_inputs * N_ACTIONS + N_ACTIONS

    def act(self, params, features):
        weights = params[:, :self.n_inputs * N_ACTIONS].reshape(-1, self.n_inputs, N_ACTIONS)
        bias = params[:, self.n_inputs * N_ACTIONS:]
        return (torch.einsum("pi,pia->pa", features, weights) + bias).argmax(-1)


def dream_rollouts(params, model, controller, start_z, steps, temperature, generator):
    population, n_starts = params.shape[0], start_z.shape[0]
    params = params.repeat_interleave(n_starts, 0)
    z = start_z.repeat(population, 1)
    h = torch.zeros(1, z.shape[0], model.rnn.hidden_size)
    total = torch.zeros(z.shape[0])
    for _ in range(steps):
        action = controller.act(params, torch.cat([z, h[0]], -1))
        out, h = model(z[:, None], action[:, None], h)
        z = sample_mdn(*model.mixture_params(out[:, 0]), temperature, generator)
        total = total + model.predict_reward(out[:, 0], z)
    return total.reshape(population, n_starts).mean(1)


def evolution_strategy(fitness, n_params, generations, population, sigma, lr, generator):
    theta = torch.zeros(n_params)
    for generation in range(generations):
        noise = torch.randn(population // 2, n_params, generator=generator)
        noise = torch.cat([noise, -noise])
        scores = fitness(theta + sigma * noise)
        ranks = scores.argsort().argsort().float()
        utilities = ranks / (len(ranks) - 1) - 0.5
        theta = theta + lr / (population * sigma) * (utilities[:, None] * noise).sum(0)
        if generation % 10 == 0 or generation == generations - 1:
            print(f"    generation {generation:3d} | mean dream return {scores.mean():7.2f} | best {scores.max():7.2f}")
    return theta


def run_real(policy_fn, n_envs, steps, generator):
    env = BallWorld(n_envs, generator)
    frame, state, total = env.reset(), None, torch.zeros(n_envs)
    for _ in range(steps):
        action, state = policy_fn(frame, state, env)
        frame, reward = env.step(action)
        total = total + reward
    return (total / steps).mean().item(), env.position


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(0)

    print("1) Collect random rollouts in the ball world (16x16 frames)")
    frames, actions, rewards = collect_rollouts(n_episodes=300, length=30, generator=generator)
    print(f"  frames {tuple(frames.shape)}, actions {tuple(actions.shape)}, rewards {tuple(rewards.shape)}")

    print("\n2) V: a VAE compresses each frame into a 4-number latent")
    vae = VAE()
    optimizer = torch.optim.Adam(vae.parameters(), lr=2e-3)
    flat = frames.flatten(0, 1)
    start = time.perf_counter()
    for step in range(2500):
        batch = flat[torch.randint(0, len(flat), (256,), generator=generator)]
        loss, reconstruction, kl = vae_loss(vae, batch, generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == 2499:
            print(f"    step {step:4d} | recon {reconstruction:7.2f} | KL {kl:5.2f}")
    print(f"  trained in {time.perf_counter() - start:.1f}s")
    with torch.no_grad():
        z_all = vae.encode(frames)[0]

    print("\n3) M: an MDN-RNN predicts the next latent and the reward")
    model = MDNRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = time.perf_counter()
    for step in range(1200):
        idx = torch.randint(0, len(z_all), (64,), generator=generator)
        z, a, r = z_all[idx], actions[idx], rewards[idx]
        h, _ = model(z[:, :-1], a)
        nll = mdn_nll(*model.mixture_params(h), z[:, 1:])
        reward_loss = F.mse_loss(model.predict_reward(h, z[:, 1:]), r)
        loss = nll + 10 * reward_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0 or step == 1199:
            print(f"    step {step:4d} | next-z NLL {nll:6.2f} | reward MSE {reward_loss:.5f}")
    print(f"  trained in {time.perf_counter() - start:.1f}s")

    print("\n4) C: a linear controller on [z, h], trained by evolution strategies entirely in the dream")
    controller = LinearController(n_inputs=4 + 64)
    start_z = z_all[:32, 0]
    start = time.perf_counter()
    with torch.no_grad():
        theta = evolution_strategy(
            lambda p: dream_rollouts(p, model, controller, start_z, 30, 1.0, generator),
            controller.n_params, generations=60, population=64, sigma=0.5, lr=1.0, generator=generator)
    print(f"  evolved {controller.n_params} parameters in {time.perf_counter() - start:.1f}s without touching the real world")

    print("\n5) Transfer: run the dream-trained controller in the real ball world")

    def world_model_policy(frame, h, env):
        with torch.no_grad():
            z = vae.encode(frame)[0]
            h = torch.zeros(1, len(frame), 64) if h is None else h
            action = controller.act(theta.expand(len(frame), -1), torch.cat([z, h[0]], -1))
            _, h = model(z[:, None], action[:, None], h)
        return action, h

    def random_policy(frame, state, env):
        return torch.randint(0, N_ACTIONS, (len(frame),), generator=generator), state

    def oracle_policy(frame, state, env):
        direction = GOAL - env.position - 3 * env.velocity
        scores = PUSHES @ direction.T
        return scores.argmax(0), state

    for name, policy in [("random actions", random_policy), ("world-model agent", world_model_policy),
                         ("oracle (knows the true state)", oracle_policy)]:
        score, final_positions = run_real(policy, 64, 40, generator)
        final_distance = (final_positions - GOAL).norm(dim=1).mean()
        print(f"  {name:30s}: mean reward {score:+.3f} | final distance to goal {final_distance:.3f}")

    print("\n6) Use It: torch.distributions builds the same mixture density")
    idx = torch.arange(8)
    with torch.no_grad():
        h, _ = model(z_all[idx, :-1], actions[idx])
        logit_pi, mu, log_sigma = model.mixture_params(h)
        mixture = MixtureSameFamily(Categorical(logits=logit_pi), Independent(Normal(mu, log_sigma.exp()), 1))
        print(f"  hand-written MDN NLL {mdn_nll(logit_pi, mu, log_sigma, z_all[idx, 1:]):.5f} | "
              f"MixtureSameFamily {-mixture.log_prob(z_all[idx, 1:]).mean():.5f}")

    with torch.no_grad():
        recon = torch.sigmoid(vae.decoder(vae.encode(frames[0, :8])[0]))
        dream_z, h = z_all[:1, 0], torch.zeros(1, 1, 64)
        dream_frames = []
        for _ in range(8):
            action = controller.act(theta[None], torch.cat([dream_z, h[0]], -1))
            out, h = model(dream_z[:, None], action[:, None], h)
            dream_z = sample_mdn(*model.mixture_params(out[:, 0]), 0.5, generator)
            dream_frames.append(torch.sigmoid(vae.decoder(dream_z))[0])
    fig, axes = plt.subplots(3, 8, figsize=(10, 4))
    for i in range(8):
        for row, image in enumerate([frames[0, i], recon[i], dream_frames[i]]):
            axes[row, i].imshow(image.reshape(SIZE, SIZE), cmap="gray")
            axes[row, i].axis("off")
    for row, title in enumerate(["real", "VAE recon", "dream"]):
        axes[row, 0].set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    path = Path(__file__).parent / "world_model.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
