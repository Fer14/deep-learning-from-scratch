# Phase 04: Sequence Models

> Read one token at a time, carry a memory forward, and learn what to keep.

This phase is about data where order matters. You start by turning words into dense vectors with skip-gram and negative sampling, then build a recurrent network by hand and train it as a character-level language model. You derive backpropagation through time yourself and watch gradients vanish and explode, which motivates the gated cells (LSTM and GRU) you implement next and match against PyTorch weight for weight. The phase closes with encoder-decoder models: a seq2seq network that reverses digit sequences until its fixed-size bottleneck breaks, and Bahdanau attention, which fixes it and leads straight into transformers. Every lesson uses small embedded text or synthetic sequence tasks, builds the mechanism with plain tensor ops first, and then compares it with the `torch.nn` version.

**Prerequisites:** Phase 01 (Neural Networks), Phase 02 (Training Deep Nets)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Word Embeddings](01-word-embeddings/) | Skip-gram with negative sampling on a generated corpus, nearest neighbors, analogies, and `nn.Embedding` |
| 02 | [RNN from Scratch](02-rnn-from-scratch/) | A tanh RNN cell trained as a character language model, temperature sampling, and a match with `nn.RNN` |
| 03 | [Backprop Through Time and Vanishing Gradients](03-backprop-through-time-and-vanishing-gradients/) | Hand-written BPTT checked against autograd, gradient decay and explosion vs spectral radius, and clipping |
| 04 | [LSTM and GRU](04-lstm-and-gru/) | LSTM and GRU cells that match PyTorch exactly, and the effect of gate bias initialization on a 60-step memory task |
| 05 | [Seq2Seq](05-seq2seq/) | A GRU encoder-decoder with teacher forcing and greedy decoding on digit reversal, plus packed sequences |
| 06 | [Attention (Bahdanau)](06-attention-bahdanau/) | Additive attention with a bidirectional encoder, accuracy vs length against plain seq2seq, and the alignment matrix |

Run the first lesson:

```bash
uv run python phases/04-sequence-models/01-word-embeddings/code/main.py
```
