import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 8000
DURATION = 1.0
UNITS = {"a": (750, 1200), "e": (450, 1900), "i": (300, 2400), "o": (500, 850), "u": (320, 650)}
UNIT_NAMES = list(UNITS)
BLANK = 0
FRAME_LENGTH = 256
HOP_LENGTH = 160
NUM_MELS = 40


def synthesize_unit(name, num_samples, pitch, rng):
    t = np.arange(num_samples) / SAMPLE_RATE
    first, second = UNITS[name]
    first *= rng.uniform(0.85, 1.15)
    second *= rng.uniform(0.85, 1.15)
    harmonics = np.arange(1, int(3600 / pitch) + 1) * pitch
    envelope = np.exp(-((harmonics - first) ** 2) / (2 * 110**2)) + 0.7 * np.exp(-((harmonics - second) ** 2) / (2 * 140**2)) + 0.02
    phases = rng.uniform(0, 2 * np.pi, len(harmonics))
    wave = (envelope[:, None] * np.sin(2 * np.pi * harmonics[:, None] * t[None, :] + phases[:, None])).sum(0)
    fade = np.minimum(1, np.minimum(np.arange(num_samples), np.arange(num_samples)[::-1]) / 80)
    return wave / (np.abs(wave).max() + 1e-8) * fade


def synthesize_utterance(rng):
    length = rng.integers(2, 6)
    units = list(rng.integers(0, len(UNIT_NAMES), size=length))
    pitch = rng.uniform(90, 200)
    audio = np.zeros(int(SAMPLE_RATE * DURATION))
    position = int(rng.uniform(0.02, 0.08) * SAMPLE_RATE)
    for unit in units:
        num_samples = int(rng.uniform(0.09, 0.15) * SAMPLE_RATE)
        audio[position : position + num_samples] += synthesize_unit(UNIT_NAMES[unit], num_samples, pitch, rng) * rng.uniform(0.6, 1.0)
        position += num_samples + int(rng.uniform(-0.01, 0.02) * SAMPLE_RATE)
    audio += rng.normal(0, 0.3, len(audio))
    return audio.astype(np.float32), [u + 1 for u in units]


def frame_signal(audio, frame_length, hop_length):
    num_frames = 1 + (audio.shape[-1] - frame_length) // hop_length
    index = torch.arange(frame_length)[None, :] + hop_length * torch.arange(num_frames)[:, None]
    return audio[..., index]


def hann_window(length):
    n = torch.arange(length, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * n / length)


def stft_power(audio):
    frames = frame_signal(audio, FRAME_LENGTH, HOP_LENGTH) * hann_window(FRAME_LENGTH)
    return torch.fft.rfft(frames, dim=-1).abs() ** 2


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(num_mels, num_fft_bins, sample_rate):
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), num_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_frequencies = np.linspace(0, sample_rate / 2, num_fft_bins)
    filters = np.zeros((num_mels, num_fft_bins))
    for m in range(num_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (bin_frequencies - left) / (center - left)
        falling = (right - bin_frequencies) / (right - center)
        filters[m] = np.maximum(0, np.minimum(rising, falling))
    return torch.tensor(filters, dtype=torch.float32)


def log_mel_spectrogram(audio, filters):
    return torch.log(stft_power(audio) @ filters.T + 1e-6)


class SpeechRecognizer(nn.Module):
    def __init__(self, num_mels=NUM_MELS, hidden=96, num_classes=len(UNIT_NAMES) + 1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(num_mels, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GELU(),
        )
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, features):
        h = self.convs(features.transpose(1, 2)).transpose(1, 2)
        return F.log_softmax(self.out(h), dim=-1)


def ctc_loss_by_hand(log_probs, targets, target_lengths):
    batch, num_frames, _ = log_probs.shape
    max_length = targets.shape[1]
    extended = torch.full((batch, 2 * max_length + 1), BLANK, dtype=torch.long, device=log_probs.device)
    extended[:, 1::2] = targets
    states = extended.shape[1]
    can_skip = torch.zeros(batch, states, dtype=torch.bool, device=log_probs.device)
    can_skip[:, 2:] = (extended[:, 2:] != BLANK) & (extended[:, 2:] != extended[:, :-2])
    impossible = -1e9
    neg_inf = torch.tensor(impossible, device=log_probs.device)
    emissions = log_probs.gather(2, extended.unsqueeze(1).expand(batch, num_frames, states))
    alpha = torch.full((batch, states), impossible, device=log_probs.device)
    alpha[:, 0] = emissions[:, 0, 0]
    alpha[:, 1] = emissions[:, 0, 1]
    for t in range(1, num_frames):
        stay = alpha
        advance = F.pad(alpha, (1, 0), value=impossible)[:, :-1]
        skip = torch.where(can_skip, F.pad(alpha, (2, 0), value=impossible)[:, :-2], neg_inf)
        alpha = torch.logsumexp(torch.stack([stay, advance, skip]), dim=0) + emissions[:, t]
    rows = torch.arange(batch, device=log_probs.device)
    final = torch.logsumexp(torch.stack([alpha[rows, 2 * target_lengths], alpha[rows, 2 * target_lengths - 1]]), dim=0)
    return (-final / target_lengths).mean()


def greedy_decode(log_probs):
    best = log_probs.argmax(-1).tolist()
    sequences = []
    for row in best:
        collapsed = [k for i, k in enumerate(row) if i == 0 or k != row[i - 1]]
        sequences.append([k for k in collapsed if k != BLANK])
    return sequences


def edit_distance(a, b):
    table = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        previous, table[0] = table[0], i
        for j in range(1, len(b) + 1):
            current = table[j]
            table[j] = min(table[j] + 1, table[j - 1] + 1, previous + (a[i - 1] != b[j - 1]))
            previous = current
    return table[-1]


def to_text(units):
    return "".join(UNIT_NAMES[u - 1] for u in units)


def make_dataset(count, rng, filters):
    audios, labels = zip(*[synthesize_utterance(rng) for _ in range(count)])
    audio = torch.tensor(np.stack(audios))
    features = log_mel_spectrogram(audio, filters)
    features = (features - features.mean()) / features.std()
    lengths = torch.tensor([len(l) for l in labels])
    targets = torch.zeros(count, int(lengths.max()), dtype=torch.long)
    for i, l in enumerate(labels):
        targets[i, : len(l)] = torch.tensor(l)
    return audio, features, targets, lengths, list(labels)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filters = mel_filterbank(NUM_MELS, FRAME_LENGTH // 2 + 1, SAMPLE_RATE)

    print(f"device {device} | {SAMPLE_RATE} Hz | {DURATION:.1f} s clips | units {UNIT_NAMES} with formants {list(UNITS.values())}")
    print("\nstep 1: waveform -> frames -> STFT power -> mel filterbank -> log")
    train_audio, train_features, train_targets, train_lengths, train_labels = make_dataset(2000, rng, filters)
    test_audio, test_features, test_targets, test_lengths, test_labels = make_dataset(300, rng, filters)
    print(f"  audio {tuple(train_audio.shape)} -> frames {tuple(frame_signal(train_audio[:1], FRAME_LENGTH, HOP_LENGTH).shape)} -> log-mel {tuple(train_features.shape)}")
    print(f"  {FRAME_LENGTH}-sample window = {1000 * FRAME_LENGTH / SAMPLE_RATE:.0f} ms, hop {1000 * HOP_LENGTH / SAMPLE_RATE:.0f} ms, {FRAME_LENGTH // 2 + 1} FFT bins -> {NUM_MELS} mel bands")
    print(f"  example transcript {to_text(train_labels[0])!r}")

    print("\nstep 2: train a convolutional acoustic model with a hand-written CTC loss")
    model = SpeechRecognizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(501):
        batch = torch.randint(len(train_features), (64,))
        log_probs = model(train_features[batch].to(device))
        loss = ctc_loss_by_hand(log_probs, train_targets[batch].to(device), train_lengths[batch].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"  step {step:3d} | ctc loss {loss.item():.3f}")
    model.eval()

    print("\nstep 3: greedy CTC decoding on new utterances")
    with torch.no_grad():
        test_log_probs = model(test_features.to(device)).cpu()
    predictions = greedy_decode(test_log_probs)
    errors = sum(edit_distance(p, t) for p, t in zip(predictions, test_labels))
    total = sum(len(t) for t in test_labels)
    exact = np.mean([p == t for p, t in zip(predictions, test_labels)])
    print(f"  unit error rate {errors / total:.3f} | exact transcripts {exact:.3f}")
    for p, t in list(zip(predictions, test_labels))[:5]:
        print(f"  predicted {to_text(p):6s} truth {to_text(t)}")
    frames = test_log_probs[0].argmax(-1).tolist()
    print(f"  frame-level argmax for the first clip: {''.join('-' if k == BLANK else UNIT_NAMES[k - 1] for k in frames)}")

    print("\nuse it: torch.stft and F.ctc_loss")
    library_power = torch.stft(test_audio[:4], FRAME_LENGTH, HOP_LENGTH, window=hann_window(FRAME_LENGTH), center=False, return_complex=True).abs() ** 2
    print(f"  max |hand STFT power - torch.stft power| = {(stft_power(test_audio[:4]) - library_power.transpose(1, 2)).abs().max().item():.2e}")
    with torch.no_grad():
        log_probs = model(test_features[:64].to(device))
        ours = ctc_loss_by_hand(log_probs, test_targets[:64].to(device), test_lengths[:64].to(device))
        input_lengths = torch.full((64,), log_probs.shape[1], dtype=torch.long)
        library = F.ctc_loss(log_probs.transpose(0, 1), test_targets[:64].to(device), input_lengths, test_lengths[:64], blank=BLANK)
    print(f"  hand CTC {ours.item():.6f} | F.ctc_loss {library.item():.6f}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5))
    axes[0].plot(np.arange(len(test_audio[0])) / SAMPLE_RATE, test_audio[0].numpy(), linewidth=0.5)
    axes[0].set_title(f"waveform: {to_text(test_labels[0])}")
    axes[1].imshow(test_features[0].T.numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("log-mel spectrogram")
    axes[2].imshow(test_log_probs[0].exp().T.numpy(), origin="lower", aspect="auto", cmap="Greys")
    axes[2].set_yticks(range(len(UNIT_NAMES) + 1), ["blank"] + UNIT_NAMES)
    axes[2].set_title("CTC output probabilities per frame")
    fig.tight_layout()
    out_path = Path(__file__).parent / "speech.png"
    fig.savefig(out_path, dpi=110)
    print(f"\nsaved {out_path.name}")


if __name__ == "__main__":
    main()
