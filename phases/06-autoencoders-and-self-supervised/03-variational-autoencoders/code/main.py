from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from torch.distributions import Normal, kl_divergence

OUT_DIR = Path(__file__).parent


def load_data(seed):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(X))
    return (torch.tensor(X[order[300:]]), torch.tensor(X[order[:300]]),
            digits.target[order[300:]], digits.target[order[:300]])


def init_linear(n_in, n_out, generator):
    weight = torch.randn(n_in, n_out, generator=generator) * np.sqrt(2.0 / n_in)
    return [weight.requires_grad_(), torch.zeros(n_out).requires_grad_()]


def mlp(layers, h):
    for index, (weight, bias) in enumerate(layers):
        h = h @ weight + bias
        if index < len(layers) - 1:
            h = torch.relu(h)
    return h


class ScratchVAE:
    def __init__(self, latent_dim, generator, hidden=128):
        self.latent_dim = latent_dim
        self.encoder = [init_linear(64, hidden, generator), init_linear(hidden, 2 * latent_dim, generator)]
        self.decoder = [init_linear(latent_dim, hidden, generator), init_linear(hidden, 64, generator)]

    def parameters(self):
        return [p for layer in self.encoder + self.decoder for p in layer]

    def encode(self, x):
        stats = mlp(self.encoder, x)
        return stats[:, :self.latent_dim], stats[:, self.latent_dim:]

    def decode_logits(self, z):
        return mlp(self.decoder, z)


def reparameterize(mu, logvar, generator):
    eps = torch.randn(mu.shape, generator=generator)
    return mu + torch.exp(0.5 * logvar) * eps


def gaussian_kl(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar).sum(dim=1)


def vae_terms(model, x, generator):
    mu, logvar = model.encode(x)
    z = reparameterize(mu, logvar, generator)
    logits = model.decode_logits(z)
    reconstruction = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)
    return reconstruction, gaussian_kl(mu, logvar)


def train_vae(model, X_train, beta, epochs, lr, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), batch_size):
            batch = X_train[order[start:start + batch_size]]
            reconstruction, kl = vae_terms(model, batch, generator)
            loss = (reconstruction + beta * kl).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def train_plain_autoencoder(X_train, latent_dim, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    encoder = [init_linear(64, 128, generator), init_linear(128, latent_dim, generator)]
    decoder = [init_linear(latent_dim, 128, generator), init_linear(128, 64, generator)]
    optimizer = torch.optim.Adam([p for layer in encoder + decoder for p in layer], lr=3e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), 128):
            batch = X_train[order[start:start + 128]]
            logits = mlp(decoder, mlp(encoder, batch))
            loss = F.binary_cross_entropy_with_logits(logits, batch, reduction="none").sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return encoder, decoder


def digit_likeness(classifier, images):
    probabilities = classifier.predict_proba(images)
    return float(np.mean(probabilities.max(axis=1) > 0.9)), np.bincount(probabilities.argmax(axis=1), minlength=10)


def save_grid(images, path, title, nrow):
    ncol = len(images) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 0.8, nrow * 0.8))
    for ax, image in zip(axes.flat, images):
        ax.imshow(image.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    X_train, X_test, y_train, y_test = load_data(seed=0)
    classifier = LogisticRegression(max_iter=2000).fit(X_train.numpy(), y_train)
    print(f"reference classifier test accuracy on real digits: {classifier.score(X_test.numpy(), y_test):.3f}")
    real_confident, _ = digit_likeness(classifier, X_test.numpy())
    print(f"real test digits with classifier confidence > 0.9: {real_confident:.2f}")

    print("\n=== Step 1: the reparameterization trick lets gradients reach mu and sigma ===")
    mu = torch.tensor([0.5], requires_grad=True)
    logvar = torch.tensor([0.0], requires_grad=True)
    z = reparameterize(mu, logvar, torch.Generator().manual_seed(0))
    (z ** 2).sum().backward()
    print(f"  z = {z.item():.3f}, dL/dmu = {mu.grad.item():.3f}, dL/dlogvar = {logvar.grad.item():.3f}")

    print("\n=== Step 2: train a VAE with an 8D latent (beta = 1) ===")
    vae = train_vae(ScratchVAE(8, torch.Generator().manual_seed(1)), X_train, beta=1.0, epochs=200, lr=2e-3, batch_size=128, seed=0)
    with torch.no_grad():
        reconstruction, kl = vae_terms(vae, X_test, torch.Generator().manual_seed(5))
    print(f"  test: reconstruction {reconstruction.mean():.2f} nats + KL {kl.mean():.2f} nats = negative ELBO {(reconstruction + kl).mean():.2f}")

    print("\n=== Step 3: sample from the prior, VAE vs plain autoencoder ===")
    prior_generator = torch.Generator().manual_seed(11)
    z_prior = torch.randn(500, 8, generator=prior_generator)
    ae_encoder, ae_decoder = train_plain_autoencoder(X_train, 8, epochs=200, seed=1)
    with torch.no_grad():
        vae_samples = torch.sigmoid(vae.decode_logits(z_prior)).numpy()
        ae_codes = mlp(ae_encoder, X_train)
        print(f"  plain AE code std per dim: {ae_codes.std(dim=0).mean():.2f}, mean |code|: {ae_codes.abs().mean():.2f} (not N(0, 1))")
        ae_samples = torch.sigmoid(mlp(ae_decoder, z_prior)).numpy()
    for name, samples in [("VAE", vae_samples), ("plain AE", ae_samples)]:
        confident, counts = digit_likeness(classifier, samples)
        print(f"  {name:8s} prior samples with confidence > 0.9: {confident:.2f} | predicted class counts {counts.tolist()}")
    save_grid(vae_samples[:60], OUT_DIR / "vae_samples.png", "VAE: decode z ~ N(0, I)", nrow=6)

    print("\n=== Step 4: the beta tradeoff ===")
    for beta in [0.1, 1.0, 4.0]:
        model = train_vae(ScratchVAE(8, torch.Generator().manual_seed(1)), X_train, beta=beta, epochs=120, lr=2e-3, batch_size=128, seed=0)
        with torch.no_grad():
            reconstruction, kl = vae_terms(model, X_test, torch.Generator().manual_seed(5))
            samples = torch.sigmoid(model.decode_logits(z_prior)).numpy()
        confident, _ = digit_likeness(classifier, samples)
        print(f"  beta {beta:3.1f}: reconstruction {reconstruction.mean():6.2f} | KL {kl.mean():5.2f} | confident samples {confident:.2f}")

    print("\n=== Step 5: a 2D latent you can walk through ===")
    vae2 = train_vae(ScratchVAE(2, torch.Generator().manual_seed(1)), X_train, beta=1.0, epochs=200, lr=2e-3, batch_size=128, seed=0)
    quantiles = torch.tensor(np.linspace(0.05, 0.95, 12), dtype=torch.float32)
    grid_values = Normal(0.0, 1.0).icdf(quantiles)
    grid = torch.stack(torch.meshgrid(grid_values, grid_values.flip(0), indexing="xy"), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        manifold = torch.sigmoid(vae2.decode_logits(grid)).numpy()
        mu_test, _ = vae2.encode(X_test)
    mean, std = mu_test.mean(dim=0), mu_test.std(dim=0)
    print(f"  test codes: mean ({mean[0]:+.2f}, {mean[1]:+.2f}), std ({std[0]:.2f}, {std[1]:.2f})")
    save_grid(manifold, OUT_DIR / "vae_manifold_2d.png", "2D VAE: decoder over a grid of z", nrow=12)
    print("  saved vae_samples.png and vae_manifold_2d.png")

    print("\n=== Use It: torch.distributions ===")
    with torch.no_grad():
        mu, logvar = vae.encode(X_test[:5])
    posterior = Normal(mu, torch.exp(0.5 * logvar))
    prior = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    library_kl = kl_divergence(posterior, prior).sum(dim=1)
    print(f"  scratch KL  {[round(v, 4) for v in gaussian_kl(mu, logvar).tolist()]}")
    print(f"  library KL  {[round(v, 4) for v in library_kl.tolist()]}")
    print(f"  rsample() is differentiable: {Normal(mu.requires_grad_(), torch.ones_like(mu)).rsample().requires_grad}")


if __name__ == "__main__":
    main()
