# Phase 05: Transformers

> Drop the recurrence. Let every token look at every other token, stack the same block, and scale.

This phase builds the transformer one component at a time. You start with scaled dot-product self-attention and see why it scales by `sqrt(d)` and why it treats a sequence as a set, then split it into multiple heads and match `nn.MultiheadAttention` weight for weight. Positional encodings (sinusoidal, learned and rotary) give attention a sense of order. You assemble the full pre-norm and post-norm blocks, check them against `nn.TransformerEncoderLayer`, and use the same block to build encoder-only, decoder-only and encoder-decoder models that learn two-digit addition. The phase closes with the Vision Transformer on digits, and with the techniques that make attention affordable on long sequences: tiled attention with the online softmax, sliding windows, linear attention and grouped-query attention. Every lesson writes the mechanism by hand with tensor ops and then compares it with the PyTorch version.

**Prerequisites:** Phase 02 (Training Deep Nets), Phase 03 (CNNs), Phase 04 (Sequence Models)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Self-Attention from Scratch](01-self-attention-from-scratch/) | Scaled dot-product attention with masks, the `sqrt(d)` experiment, and a counting task a per-token MLP cannot solve |
| 02 | [Multi-Head Attention](02-multi-head-attention/) | Batched heads that match `nn.MultiheadAttention`, and heads that specialize into previous- and next-token lookups |
| 03 | [Positional Encodings](03-positional-encodings/) | Sinusoidal, learned and rotary encodings, their properties, and a reversal task that fails without them |
| 04 | [The Transformer Block](04-the-transformer-block/) | LayerNorm, attention, feed-forward and residuals in pre- and post-norm form, matching `nn.TransformerEncoderLayer`, plus a sorter |
| 05 | [Encoders vs Decoders](05-encoders-vs-decoders/) | Encoder-only, decoder-only and encoder-decoder models with cross-attention, trained on two-digit addition alongside `nn.Transformer` |
| 06 | [Vision Transformer](06-vision-transformer/) | Patch embeddings, a `[CLS]` token and a ViT on 8x8 digits, compared with torchvision's ViT and a small CNN |
| 07 | [Efficient Attention](07-efficient-attention/) | Exact tiled attention with the online softmax, sliding windows, linear attention, benchmarks and grouped-query attention |

Run the first lesson:

```bash
uv run python phases/05-transformers/01-self-attention-from-scratch/code/main.py
```
