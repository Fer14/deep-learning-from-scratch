# Phase 09: LLMs

> A large language model is a next-token predictor, scaled up, tuned to follow instructions, and served efficiently.

This phase builds the full life cycle of a language model at a size that trains in seconds. You start with how text becomes integers (byte pair encoding) and the oldest language models (n-gram counts and perplexity), then build a GPT from scratch with hand-written causal attention. You fit a scaling law on a family of tiny GPTs, learn the decoding strategies that turn distributions into text, adapt a pretrained model with full fine-tuning and with LoRA, align a policy to hidden preferences with RLHF and DPO, and finish with the KV cache that makes generation fast. Every model trains on a small story or synthetic text written inside the script: no downloads, no pretrained weights.

**Prerequisites:** Phase 02 (Training Deep Nets), Phase 04 (Sequence Models), Phase 05 (Transformers)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Tokenization and BPE](01-tokenization-and-bpe/) | A byte-level BPE tokenizer with regex pre-tokenization, encode/decode and a compression curve |
| 02 | [Language Modeling and N-grams](02-language-modeling-and-n-grams/) | Smoothed character n-gram models, perplexity, interpolation and a neural bigram |
| 03 | [GPT from Scratch](03-gpt-from-scratch/) | A decoder-only transformer with hand-written causal attention, trained on a short story |
| 04 | [Scaling Laws and Training at Scale](04-scaling-laws-and-training-at-scale/) | A power-law fit over six GPT sizes, with warmup, cosine decay, accumulation and bfloat16 |
| 05 | [Sampling and Decoding](05-sampling-and-decoding/) | Greedy, temperature, top-k, top-p and beam search, with a quality-diversity table |
| 06 | [Fine-Tuning and LoRA](06-fine-tuning-and-lora/) | Full fine-tuning versus hand-written LoRA adapters, a rank sweep and adapter merging |
| 07 | [RLHF and DPO](07-rlhf-and-dpo/) | A Bradley-Terry reward model, policy-gradient RLHF with a KL penalty, and DPO on a toy preference task |
| 08 | [Inference: The KV Cache](08-inference-kv-cache/) | Prefill and cached decode steps, a speed comparison and cache memory arithmetic |

Run the first lesson:

```bash
uv run python phases/09-llms/01-tokenization-and-bpe/code/main.py
```
