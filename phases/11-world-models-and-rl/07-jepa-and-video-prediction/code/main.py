import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

SIZE, EMBED, AHEAD = 16, 32, 3


def make_videos(n_videos, length, generator, kick_prob=0.15, noise=0.35):
    position = torch.rand(n_videos, 2, generator=generator) * 0.7 + 0.15
    angle = torch.rand(n_videos, generator=generator) * 2 * torch.pi
    velocity = 0.05 * torch.stack([angle.cos(), angle.sin()], 1)
    coords = (torch.arange(SIZE) + 0.5) / SIZE
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    frames, positions, velocities = [], [], []
    for _ in range(length):
        image = torch.exp(-((grid_x - position[:, 0, None, None]) ** 2 + (grid_y - position[:, 1, None, None]) ** 2) / (2 * 0.07**2))
        clutter = noise * torch.rand(n_videos, SIZE, SIZE, generator=generator)
        frames.append((image + clutter).clamp(0, 1).flatten(1))
        positions.append(position)
        velocities.append(velocity)
        kick = torch.rand(n_videos, 1, generator=generator) < kick_prob
        new_angle = torch.rand(n_videos, generator=generator) * 2 * torch.pi
        velocity = torch.where(kick, 0.05 * torch.stack([new_angle.cos(), new_angle.sin()], 1), velocity)
        position = position + velocity
        bounce = (position < 0.1) | (position > 0.9)
        velocity = torch.where(bounce, -velocity, velocity)
        position = position.clamp(0.1, 0.9)
    return torch.stack(frames, 1), torch.stack(positions, 1), torch.stack(velocities, 1)


def encoder():
    return nn.Sequential(nn.Linear(2 * SIZE * SIZE, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, EMBED))


def pairs(frames, t):
    return torch.cat([frames[:, t - 1], frames[:, t]], -1)


def sample_batch(frames, batch_size, generator):
    idx = torch.randint(0, frames.shape[0], (batch_size,), generator=generator)
    t = int(torch.randint(1, frames.shape[1] - AHEAD, (1,), generator=generator))
    return pairs(frames[idx], t), pairs(frames[idx], t + AHEAD), frames[idx, t + AHEAD]


def train_jepa(frames, steps, generator, ema_decay=0.99, collapse=False):
    context_encoder = encoder()
    target_encoder = copy.deepcopy(context_encoder)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    library_ema = AveragedModel(context_encoder, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    predictor = nn.Sequential(nn.Linear(EMBED, 128), nn.GELU(), nn.Linear(128, EMBED))
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()), lr=1e-3, weight_decay=0.01)
    for step in range(steps):
        context, target, _ = sample_batch(frames, 256, generator)
        prediction = predictor(context_encoder(context))
        if collapse:
            target_embedding = context_encoder(target)
        else:
            with torch.no_grad():
                target_embedding = target_encoder(target)
        loss = F.mse_loss(prediction, target_embedding)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not collapse:
            with torch.no_grad():
                for p_target, p_online in zip(target_encoder.parameters(), context_encoder.parameters()):
                    p_target.mul_(ema_decay).add_((1 - ema_decay) * p_online)
            library_ema.update_parameters(context_encoder)
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                spread = context_encoder(context).std(0).mean().item()
            print(f"    step {step:4d} | latent prediction loss {loss.item():.5f} | embedding std across batch {spread:.4f}")
    return context_encoder, target_encoder, library_ema, predictor


def train_pixel_predictor(frames, steps, generator):
    context_encoder = encoder()
    decoder = nn.Sequential(nn.Linear(EMBED, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, SIZE * SIZE))
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=0.01)
    for step in range(steps):
        context, _, future_frame = sample_batch(frames, 256, generator)
        loss = F.mse_loss(torch.sigmoid(decoder(context_encoder(context))), future_frame)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"    step {step:4d} | pixel MSE {loss.item():.5f}")
    return context_encoder, decoder


def features(model, frames, t):
    with torch.no_grad():
        return model(pairs(frames, t)).numpy()


def linear_probe(model, train, test, t=5):
    (train_frames, train_pos, train_vel), (test_frames, test_pos, test_vel) = train, test
    scores = {}
    for name, train_y, test_y in [("position", train_pos[:, t], test_pos[:, t]), ("velocity", train_vel[:, t], test_vel[:, t])]:
        probe = Ridge(alpha=1e-2).fit(features(model, train_frames, t), train_y.numpy())
        scores[name] = probe.score(features(model, test_frames, t), test_y.numpy())
    return scores


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(0)
    train = make_videos(3000, 12, generator)
    test = make_videos(1000, 12, generator)
    print(f"Videos: {tuple(train[0].shape)} (videos, frames, pixels). Ball + random kicks + per-frame pixel clutter.")
    print(f"Task: from frames (t-1, t), predict {AHEAD} steps ahead.\n")

    start = time.perf_counter()
    print("1) Pixel-space predictor: encoder -> decoder -> future frame")
    pixel_encoder, decoder = train_pixel_predictor(train[0], 1500, generator)

    print("\n2) JEPA: predict the target encoder's embedding of the future, target updated by EMA")
    jepa_encoder, target_encoder, library_ema, predictor = train_jepa(train[0], 1500, generator)

    print("\n3) Naive latent prediction: same encoder for target, gradients flow into both sides")
    collapsed_encoder, _, _, _ = train_jepa(train[0], 600, generator, collapse=True)
    print(f"  trained all three in {time.perf_counter() - start:.1f}s")

    print("\n4) Linear probes on frozen embeddings of (t-1, t): can a linear map read out the ball's state?")
    torch.manual_seed(1)
    random_encoder = encoder()
    print(f"  {'encoder':26s} {'position R^2':>13s} {'velocity R^2':>13s}")
    for name, model in [("random init", random_encoder), ("pixel predictor", pixel_encoder),
                        ("naive (collapsed)", collapsed_encoder), ("JEPA", jepa_encoder)]:
        scores = linear_probe(model, train, test)
        print(f"  {name:26s} {scores['position']:13.3f} {scores['velocity']:13.3f}")

    print("\n5) What the pixel predictor spends its effort on")
    with torch.no_grad():
        context, _, future = sample_batch(test[0], 512, generator)
        predicted = torch.sigmoid(decoder(pixel_encoder(context)))
        clutter_mean = 0.35 / 2
        baseline = F.mse_loss(torch.full_like(future, clutter_mean), future).item()
        print(f"  MSE of predicting a flat clutter-mean image: {baseline:.5f}")
        print(f"  MSE of the trained pixel predictor:          {F.mse_loss(predicted, future).item():.5f}")
        print("  most of the remaining error is unpredictable clutter; JEPA's loss never asks for it")

    print("\n6) Use It: torch.optim.swa_utils.AveragedModel keeps the same EMA")
    gap = max((a - b).abs().max().item() for a, b in zip(target_encoder.parameters(), library_ema.module.parameters()))
    print(f"  max |manual EMA - AveragedModel EMA| over all weights: {gap:.2e}")

    fig, axes = plt.subplots(3, 6, figsize=(9, 4.8))
    for i in range(6):
        axes[0, i].imshow(context[i, SIZE * SIZE:].reshape(SIZE, SIZE), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(future[i].reshape(SIZE, SIZE), cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(predicted[i].reshape(SIZE, SIZE), cmap="gray", vmin=0, vmax=1)
        for row in range(3):
            axes[row, i].axis("off")
    for row, title in enumerate(["frame t", f"frame t+{AHEAD}", "pixel prediction"]):
        axes[row, 0].set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    path = Path(__file__).parent / "video_prediction.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
