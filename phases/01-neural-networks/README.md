# Phase 01: Neural Networks

> A neuron draws a line. Stack them with a bend in between, send the error back, and they learn any shape.

This phase builds neural networks from the single neuron up. You start with the perceptron and its XOR failure, then fix it with hidden layers and see why nonlinear activations are what make depth useful. You implement the activations and losses networks actually use, derive backpropagation on paper and check it numerically, and assemble a modular NumPy framework that trains an MLP on handwritten digits. A lesson on universal approximation shows what networks can represent and where that guarantee stops. The phase closes by rebuilding PyTorch's `nn.Module` bookkeeping yourself before using the real thing. Every lesson builds the idea in NumPy first and then checks it against PyTorch.

**Prerequisites:** Phase 00 (Tensors and Autodiff)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [The Perceptron](01-the-perceptron/) | Rosenblatt's mistake-driven learner, logic gates, and the XOR failure |
| 02 | [Multilayer Perceptrons](02-multilayer-perceptrons/) | A hand-wired XOR network, the linear-collapse proof, and a batched MLP forward pass |
| 03 | [Activation Functions](03-activation-functions/) | Sigmoid, tanh, ReLU, leaky ReLU, GELU and softmax with derivatives, plus a 20-layer gradient experiment |
| 04 | [Loss Functions for Networks](04-loss-functions-for-networks/) | MSE, MAE, Huber and stable cross-entropy from logits, with the `p - y` gradient verified |
| 05 | [Backpropagation by Hand](05-backpropagation-by-hand/) | The full backward pass of a two-layer network, a gradient check, and a match with autograd |
| 06 | [An MLP from Scratch in NumPy](06-an-mlp-from-scratch-in-numpy/) | Modular layers and minibatch SGD that reach about 97% on the 8x8 digits dataset |
| 07 | [Universal Approximation Intuition](07-universal-approximation-intuition/) | Sigmoid bumps, ReLU interpolants and trained networks approximating a curve, and where they fail |
| 08 | [PyTorch nn.Module](08-pytorch-nn-module/) | A mini module system with parameter registration and state_dict, then a real `nn.Module` with buffers and dropout |

Run the first lesson:

```bash
uv run python phases/01-neural-networks/01-the-perceptron/code/main.py
```
