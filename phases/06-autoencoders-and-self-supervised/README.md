# Phase 06: Autoencoders and Self-Supervised Learning

> No labels? Make the data its own teacher.

This phase is about learning from unlabeled data. You start with the autoencoder, which compresses an input through a bottleneck and rebuilds it, and check that its linear version is exactly PCA. Denoising autoencoders replace the bottleneck with corruption and reveal a vector field that points back to the data, the seed of diffusion models. Variational autoencoders make the code space a proper probability distribution you can sample from. Then the phase moves from reconstruction to the modern self-supervised objectives: contrastive learning (SimCLR) teaches invariance by matching two augmented views, a comparison lesson shows how to score and use embeddings, and masked autoencoders hide most of an image and let a transformer fill it in. Every lesson runs on scikit-learn's 8x8 digits and trains in seconds on a CPU. Models are built from raw weight tensors (or with the transformer blocks from Phase 05), and the scripts pin PyTorch to one CPU thread because tiny models run faster that way.

**Prerequisites:** Phase 01 (Neural Networks), Phase 02 (Training Deep Nets), Phase 03 (CNNs), Phase 05 (Transformers)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Autoencoders](01-autoencoders/) | A scratch MLP autoencoder, compared with PCA, with a 2D code you can plot |
| 02 | [Denoising Autoencoders](02-denoising-autoencoders/) | Gaussian and masking denoisers, plus the denoising vector field on a 2D ring |
| 03 | [Variational Autoencoders](03-variational-autoencoders/) | A VAE with the reparameterization trick, closed-form KL, prior sampling and a beta sweep |
| 04 | [Contrastive Learning](04-contrastive-learning/) | SimCLR with augmentations, a conv encoder, NT-Xent from scratch and linear probes |
| 05 | [Representation Learning and Embeddings](05-representation-learning-and-embeddings/) | Embedding tables, kNN / precision@k / linear-probe scoring, cosine retrieval and transfer |
| 06 | [Masked Autoencoders](06-masked-autoencoders/) | An asymmetric ViT MAE with patch masking, a mask-ratio sweep and few-label fine-tuning |

Run the first lesson:

```bash
uv run python phases/06-autoencoders-and-self-supervised/01-autoencoders/code/main.py
```
