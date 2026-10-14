# Phase 00: Tensors and Autodiff

> Every deep learning framework is two ideas: arrays that move fast, and numbers that remember how they were made.

This phase builds the machinery under every model in the course. You start with tensors: shapes, strides, views, and the broadcasting rules that decide which shapes can be combined. Then you turn formulas into computational graphs and differentiate them by hand in forward and reverse mode. With that in place you write an autograd engine for scalars, in the spirit of micrograd, and train a tiny network with it, then rewrite it to operate on whole NumPy arrays so it can train an MLP on real digit images in under a second. The phase ends by mapping everything you built onto PyTorch's tensors and `torch.autograd`, and by learning how CPUs and GPUs actually execute that code.

**Prerequisites:** Python and NumPy, [Classic ML from Scratch](https://github.com/Fer14/classic-ml-from-scratch) Phase 01 (Linear Algebra) and Phase 02 (Calculus and Optimization)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Tensors and Broadcasting](01-tensors-and-broadcasting/) | Shape and stride inspection, the broadcasting rules as a function, and `unbroadcast` for gradients |
| 02 | [Computational Graphs](02-computational-graphs/) | An explicit graph with forward-mode and reverse-mode differentiation, timed against each other |
| 03 | [Autograd from Scratch](03-autograd-from-scratch/) | A scalar `Value` autograd engine and a tiny MLP trained with it on two moons |
| 04 | [A Vectorized Autodiff Engine](04-vectorized-autodiff-engine/) | A NumPy `Tensor` engine with broadcasting, matmul and cross-entropy that trains a digit classifier |
| 05 | [PyTorch Tensors and Autograd](05-pytorch-tensors-and-autograd/) | Gradient accumulation, vector-Jacobian products, `no_grad`, a custom `autograd.Function` and a training loop |
| 06 | [Devices and GPU Basics](06-devices-and-gpu-basics/) | A correct GPU timer, CPU vs GPU and dtype benchmarks, transfer costs and memory accounting |

Run the first lesson:

```bash
uv run python phases/00-tensors-and-autodiff/01-tensors-and-broadcasting/code/main.py
```
