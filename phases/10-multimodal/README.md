# Phase 10: Multimodal

> Every modality is just another way to make vectors. Align them, attend across them, or turn them all into tokens.

This phase connects vision, language and audio. You start with CLIP-style contrastive learning, which puts images and captions into one embedding space and turns classification into a text query. You then generate text from images with an encoder-decoder captioner, answer questions about images with a LLaVA-style vision-language model, and go the other way with a text-conditioned diffusion model and classifier-free guidance. An audio lesson builds log-mel spectrograms and a CTC speech recognizer from scratch, and the phase closes with an any-to-any transformer that reads and writes both text and image tokens. All data is generated in the scripts or comes from scikit-learn's bundled digits: synthetic shapes rendered with NumPy, captions like "a handwritten seven", and formant-synthesized speech sounds.

**Prerequisites:** Phase 05 (Transformers), Phase 06 (Autoencoders and Self-Supervised), Phase 08 (Diffusion Models), Phase 09 (LLMs)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Image-Text Embeddings (CLIP)](01-image-text-embeddings-clip/) | Two encoders trained with symmetric InfoNCE, zero-shot digit classification and text-to-image retrieval |
| 02 | [Image Captioning](02-image-captioning/) | A CNN encoder with a hand-written cross-attention decoder, greedy captions and a compositional generalization test |
| 03 | [Vision-Language Models](03-vision-language-models/) | A frozen vision encoder, an MLP projector and a tiny GPT answering questions about shapes, with a blind baseline |
| 04 | [Text-to-Image Conditioning](04-text-to-image-conditioning/) | A FiLM-conditioned diffusion model with caption dropout and classifier-free guidance, graded by a classifier |
| 05 | [Audio and Speech Models](05-audio-and-speech-models/) | Log-mel features and a CTC loss written by hand, training a convolutional recognizer on synthetic speech |
| 06 | [Any-to-Any Models](06-any-to-any-models/) | A k-means image tokenizer and one transformer that does both text-to-image and image-to-text |

Run the first lesson:

```bash
uv run python phases/10-multimodal/01-image-text-embeddings-clip/code/main.py
```
