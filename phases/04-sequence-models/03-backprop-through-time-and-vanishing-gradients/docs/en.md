# Backprop Through Time and Vanishing Gradients

> The gradient that reaches step 1 has been multiplied by the same matrix fifty times. That number either dies or explodes.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Neural Networks), RNN from Scratch
**Time:** ~55 minutes

## Learning Objectives

- Derive and implement backpropagation through time for a vanilla RNN by hand and match autograd
- Explain why `dL/dh_t` behaves like a product of Jacobians and how the spectral radius of `W_hh` and tanh saturation control it
- Measure vanishing and exploding gradients numerically and tame explosions with gradient clipping
- Show the practical consequence: a vanilla RNN that cannot learn a dependency longer than a few dozen steps

## The Problem

The RNN in the previous lesson memorized a short text, so it clearly learns something across time. But how far back can it look? Suppose the first word of a long sentence decides the last one (a subject and its verb, an opening and a closing bracket). For the network to learn that, the error at the last step has to flow back to the first step and tell `W_xh` how to encode it.

That error signal passes through every intermediate step, and each step multiplies it by a Jacobian. Bengio, Simard and Frasconi (1994) and Hochreiter (1991) showed that this product shrinks or grows exponentially with the distance. When it shrinks, the network simply never receives a learning signal for long-range dependencies. When it grows, a single batch can produce a gigantic update. This lesson makes both effects visible, and it is the reason LSTMs, GRUs and eventually attention were invented.

## The Concept

### The forward pass

```
a_t = x_t W_xh + h_{t-1} W_hh + b_h
h_t = tanh(a_t)
logits = h_T W_hy + b_y           (one prediction from the last state)
```

### The backward pass

Start at the output. With softmax and cross-entropy, `dL/dlogits = (p - y) / batch`. Then:

```
dL/dh_T     = dL/dlogits W_hy^T
dL/da_t     = dL/dh_t * (1 - h_t^2)                   (tanh derivative)
dL/dW_xh   += x_t^T dL/da_t
dL/dW_hh   += h_{t-1}^T dL/da_t
dL/db_h    += sum over batch of dL/da_t
dL/dh_{t-1} = dL/da_t W_hh^T
```

Walk `t` from `T` down to 1. The weight gradients **accumulate** over time steps because the same weights are used at every step. That is all backpropagation through time (BPTT) is: ordinary backprop on the unrolled graph, summing the contributions of tied weights.

### The product of Jacobians

Each step back multiplies the gradient by the Jacobian of `h_t` with respect to `h_{t-1}`:

```
dh_t / dh_{t-1} = diag(1 - h_t^2) W_hh^T

dL/dh_k = dL/dh_T  *  prod_{t=k+1}^{T} diag(1 - h_t^2) W_hh^T
```

Two factors decide the size of that product:

- **`W_hh`**. If its largest singular value is below 1, every step shrinks the gradient; above 1, it can grow. With `W_hh = r * Q` for an orthogonal `Q`, the gradient scales roughly like `r^(T-k)`.
- **tanh saturation**. `1 - h_t^2 <= 1`, with equality only at `h_t = 0`. Whenever units saturate (large inputs, large weights), the diagonal factor pushes the product down further.

So for a vanilla RNN, the gradient can only survive long distances in a narrow regime: `W_hh` close to orthogonal with radius 1 and activations that stay near zero. Training does not stay in that regime.

```
radius < 1:   0.5^10 = 0.001       vanishes
radius = 1:   stays ~ constant      (only without saturation)
radius > 1:   1.5^10 = 58           explodes
```

### Clipping

Explosions are easy to treat: if the total gradient norm exceeds a threshold, scale every gradient down so the norm equals the threshold. The direction is kept, the step size is bounded. Vanishing gradients have no such fix inside a vanilla RNN; they need a different architecture.

## Build It

### Step 1: Forward pass that keeps every state

```python
def rnn_forward(x, h0, W_xh, W_hh, b_h, W_hy, b_y):
    hidden_states = [h0]
    for t in range(x.shape[1]):
        hidden_states.append(torch.tanh(x[:, t] @ W_xh + hidden_states[-1] @ W_hh + b_h))
    logits = hidden_states[-1] @ W_hy + b_y
    return logits, hidden_states
```

Backprop needs every `h_t`, which is why RNN training memory grows linearly with sequence length.

### Step 2: BPTT by hand

```python
def rnn_backward(x, hidden_states, logits, targets, W_xh, W_hh, W_hy):
    batch = x.shape[0]
    d_logits = (torch.softmax(logits, dim=-1) - F.one_hot(targets, logits.shape[-1]).float()) / batch
    grads = {"W_hy": hidden_states[-1].T @ d_logits, "b_y": d_logits.sum(dim=0),
             "W_xh": torch.zeros_like(W_xh), "W_hh": torch.zeros_like(W_hh), "b_h": torch.zeros(W_hh.shape[0])}
    d_h = d_logits @ W_hy.T
    hidden_grad_norms = []
    for t in reversed(range(x.shape[1])):
        hidden_grad_norms.append(d_h.norm().item())
        d_pre = d_h * (1 - hidden_states[t + 1] ** 2)
        grads["W_xh"] += x[:, t].T @ d_pre
        grads["W_hh"] += hidden_states[t].T @ d_pre
        grads["b_h"] += d_pre.sum(dim=0)
        d_h = d_pre @ W_hh.T
    return grads, hidden_grad_norms[::-1]
```

`hidden_states[t + 1]` is `h_t` and `hidden_states[t]` is `h_{t-1}`, because the list starts with `h_0`. Against autograd on a 12-step sequence:

```
dW_xh  1.2e-07
dW_hh  6.0e-08
db_h   3.0e-08
dW_hy  3.0e-08
db_y   1.5e-08
```

### Step 3: Watch the gradient travel back

Set `W_hh = r * Q` with `Q` orthogonal, use small inputs (so tanh stays nearly linear), run 50 steps, and record `||dL/dh_t||`:

```
radius      t=49      t=40      t=30      t=20      t=10      t= 0
   0.5  5.23e-01  1.02e-03  9.94e-07  9.70e-10  9.46e-13  9.22e-16
   0.9  5.21e-01  2.00e-01  6.95e-02  2.41e-02  8.35e-03  2.90e-03
   1.0  5.29e-01  5.04e-01  4.85e-01  4.72e-01  4.65e-01  4.61e-01
   1.1  6.05e-01  6.00e-01  7.27e-01  1.21e+00  2.81e+00  7.18e+00
   1.5  6.91e-01  9.85e-01  3.00e+00  4.97e+00  1.64e+01  6.13e+02
```

The geometric decay with `r = 0.5` is almost exactly `0.5` per step: 15 orders of magnitude over 50 steps. Only `r = 1` keeps the gradient flat.

### Step 4: Saturation kills it even at radius 1

Now keep `r = 1` and increase the input scale, so tanh saturates. The script multiplies the per-step Jacobians and prints the norm of the product:

```
input scale 0.1: ||dh_t/dh_0|| at t=5,10,15,20: 9.72e-01, 8.40e-01, 6.58e-01, 4.93e-01
input scale 1.0: ||dh_t/dh_0|| at t=5,10,15,20: 3.58e-01, 4.10e-02, 3.14e-03, 3.47e-04
input scale 3.0: ||dh_t/dh_0|| at t=5,10,15,20: 2.95e-02, 7.74e-04, 2.66e-06, 9.99e-09
```

A perfectly norm-preserving `W_hh` does not help once the activations are pushed toward +-1.

### Step 5: Explosion and clipping

With `W_hh = 3Q` and 50 steps, the gradient norm reaches about `2e6`. `torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)` rescales it to exactly 1 and returns the norm before clipping, which is worth logging in any RNN training run.

### Step 6: The consequence for learning

A task that needs long memory: the first input is one of four symbols, followed by `delay` random noise symbols; at the end the network must output the first symbol. A vanilla `nn.RNN` with 32 hidden units, 400 Adam steps, clipping at 1:

```
delay  accuracy   grad at x_0 / grad at x_T (init)
    5     1.000                           6.19e-02
   10     1.000                           3.96e-03
   20     1.000                           1.38e-05
   30     1.000                           4.79e-08
   40     0.242                           1.59e-10
```

The last column is the ratio between the gradient that reaches the first input and the gradient at the last input, at initialization. It falls by a factor of 10 to 20 every 5 steps. Adam rescales tiny gradients, so the network still learns up to a delay of 30, but somewhere between 30 and 40 the signal is lost in the noise and accuracy drops to chance (0.25).

## Use It

In PyTorch you never write BPTT: `loss.backward()` on the output of `nn.RNN` (or of your own loop) does it. What you do control:

- **Clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` after `backward()` and before `optimizer.step()`. Use it in every recurrent model.
- **Truncation**: for long streams, split the sequence into chunks and call `h = h.detach()` between chunks. Gradients then flow at most one chunk back, which bounds memory and cost but also caps the dependencies the model can learn.
- **Initialization**: `nn.init.orthogonal_(rnn.weight_hh_l0)` starts `W_hh` with all singular values equal to 1, the best case for gradient flow.

None of these fixes vanishing gradients in a vanilla RNN. The next lesson changes the architecture so that the path through time is additive instead of multiplicative, the same trick residual networks used across depth.

## Exercises

1. **Easy.** Rerun Step 3 with radius 0.95 and 1.05 and predict the gradient at `t = 0` before running it, using `r^49`.
2. **Medium.** Train the memory task at delay 40 with `nn.init.orthogonal_` on `W_hh` and ReLU instead of tanh (`nn.RNN(..., nonlinearity="relu")`). Does it learn now? What happens to stability?
3. **Hard.** Extend `rnn_backward` to a loss at every time step (as in the character model). Verify it against autograd, then measure how much each time step's loss contributes to `dW_hh` for a 50-step sequence. Which steps dominate the gradient?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Backpropagation through time | "BPTT" | Backprop on the unrolled RNN; gradients of tied weights are summed over all time steps |
| Vanishing gradient | "The RNN forgets" | `dL/dh_k` shrinks exponentially with `T - k`, so early inputs get no learning signal |
| Exploding gradient | "Training blows up" | The same product grows exponentially, producing huge updates |
| Jacobian product | "Chain rule over time" | `prod diag(1 - h_t^2) W_hh^T`; its norm controls how much gradient survives |
| Spectral radius | "Largest eigenvalue" | The largest absolute eigenvalue of `W_hh`; above 1 tends to explode, below 1 tends to vanish |
| Saturation | "Dead tanh" | Activations near +-1 where the derivative is near 0, shrinking the gradient further |
| Gradient clipping | "Norm clipping" | Rescaling the total gradient to at most a chosen norm; fixes explosions, not vanishing |
| Truncated BPTT | "Detaching the state" | Backpropagating only a fixed number of steps by detaching the hidden state between chunks |

## Further Reading

- [Bengio, Simard and Frasconi, Learning Long-Term Dependencies with Gradient Descent is Difficult (1994)](https://ieeexplore.ieee.org/document/279181) -- the classic analysis of vanishing gradients in RNNs
- [Pascanu, Mikolov and Bengio, On the difficulty of training Recurrent Neural Networks (2012)](https://arxiv.org/abs/1211.5063) -- the spectral-radius argument and gradient norm clipping
- [Dive into Deep Learning, Section 9.7: Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) -- the derivation with full and truncated variants
- [Karpathy, min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086) -- a 100-line NumPy RNN with a hand-written BPTT loop
