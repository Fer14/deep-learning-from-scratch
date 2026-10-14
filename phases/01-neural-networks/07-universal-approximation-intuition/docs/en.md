# Universal Approximation Intuition

> Enough bumps can draw any curve. The theorem says the network exists; it does not say you will find it, or that it will generalize.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 00 (Tensors and Autodiff), An MLP from Scratch in NumPy
**Time:** ~40 minutes

## Learning Objectives

- Build a "bump" from two sigmoid units and approximate a curve with a sum of bumps
- Construct an exact piecewise-linear interpolant with a sum of ReLUs
- State what the universal approximation theorem guarantees and what it does not (learnability, efficiency, extrapolation)
- Train one-hidden-layer networks of increasing width and watch the approximation error fall

## The Problem

You have built MLPs and trained one on digits. A natural question is: how do you know a network can represent the function you need at all? Maybe your problem needs some exotic architecture.

The universal approximation theorem (Cybenko 1989, Hornik 1991) answers the representational part: a network with a single hidden layer and a nonlinear activation can approximate any continuous function on a bounded region as closely as you want, given enough hidden units. That result is often quoted as "neural networks can learn anything", which is wrong in three important ways. This lesson builds the intuition for why the theorem is true with your own hands, and then shows where it stops helping.

## The Concept

### Two sigmoids make a bump

A steep sigmoid is almost a step. Subtract a step shifted to the right and you get a rectangle:

```
sigmoid(k (x - a))          ____________
                           |
                    _______|

- sigmoid(k (x - b))                  _____
                                     |
                    _________________|

= bump on [a, b]            _______
                           |       |
                    _______|       |______
                           a       b
```

Two hidden units plus one output weight (the height) give one bump. That is a tiny network:

```
x --> [ sigmoid(k(x - a)) ] --( +h )--.
                                       +--> output
x --> [ sigmoid(k(x - b)) ] --( -h )--'
```

### Bumps make any curve

Split the interval into `N` pieces, put a bump on each with height `f(center)`, and add them up. That is a staircase approximation of `f`, like a Riemann sum. As `N` grows and the bumps get narrower, the staircase converges to `f`. A network with `2N` hidden sigmoid units represents it exactly. This is the core of the universal approximation argument in one dimension.

### ReLUs make piecewise-linear curves

ReLU networks do the same with ramps. Starting from a line through the first knot, each `ReLU(x - k_i)` term changes the slope at knot `k_i`:

```
f_hat(x) = f(k0) + s0 (x - k0) + sum_i (s_i - s_{i-1}) ReLU(x - k_i)
```

where `s_i` is the slope between knots `i` and `i+1`. The result passes exactly through every knot and is linear in between. With more knots, the error shrinks quadratically for smooth functions.

### What the theorem does not say

| The theorem says | It does not say |
|------------------|-----------------|
| Some weights exist that approximate `f` | Gradient descent will find them |
| One hidden layer is enough | One hidden layer is efficient; some functions need exponentially many units shallow but few units deep |
| Approximation on a bounded region | Anything about inputs outside that region |
| Approximation of a known `f` | Generalization from a finite, noisy sample |

Depth matters in practice because deep networks can reuse features: a function built by composing simple pieces can need far fewer units deep than wide. And extrapolation is where the theorem is silent: a tanh network trained on `[-2, 2]` saturates outside it and goes flat.

## Build It

The target throughout is `f(x) = sin(3x) + 0.3 x^2` on `[-2, 2]`.

### Step 1: A bump

```python
def bump(x, left, right, height, steepness=50.0):
    return height * (sigmoid(steepness * (x - left)) - sigmoid(steepness * (x - right)))
```

`bump(x, -0.5, 0.5, 1)` is about 1 inside the interval and 0 outside.

### Step 2: A sum of bumps

```python
def bump_approximation(x, n_bumps):
    edges = np.linspace(-2.1, 2.1, n_bumps + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    heights = target_function(centers)
    steepness = 10.0 * n_bumps
    return sum(bump(x, left, right, h, steepness) for left, right, h in zip(edges[:-1], edges[1:], heights))
```

The steepness grows with `N` so each step stays sharp relative to its width.

```
   4 bumps (  8 sigmoid units) max error = 1.1717
  16 bumps ( 32 sigmoid units) max error = 0.3811
  64 bumps (128 sigmoid units) max error = 0.0951
```

### Step 3: ReLU interpolation

```python
def relu_interpolation(x, knots):
    values = target_function(knots)
    slopes = np.diff(values) / np.diff(knots)
    output = values[0] + slopes[0] * (x - knots[0])
    for index in range(1, len(slopes)):
        output = output + (slopes[index] - slopes[index - 1]) * np.maximum(0.0, x - knots[index])
    return output
```

```
   3 ReLU units: max error = 1.0022
   7 ReLU units: max error = 0.2503
  31 ReLU units: max error = 0.0186
```

ReLUs converge much faster than sigmoid staircases here because each piece can slope, not just step.

### Step 4: Let gradient descent choose

Hand construction proves existence. Training is what you actually do. The script trains a one-hidden-layer tanh network with Adam (written by hand in a few lines; Phase 02 explains it) for 3000 full-batch steps:

```
   2 tanh units: MSE = 0.11278
   8 tanh units: MSE = 0.00021
  32 tanh units: MSE = 0.00000
```

Eight trained units beat 128 hand-placed sigmoids, because training places and shapes each unit where it helps most.

### Step 5: Extrapolation

```
  x = [3. 4.] true = [3.112 4.263] predicted = [2.286 2.349]
```

Inside `[-2, 2]` the 32-unit network is nearly perfect. Outside, its tanh units saturate and the output goes flat while the true function keeps rising. The theorem never promised otherwise. The script saves `approximations.png` with both experiments.

## Use It

```python
net = torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(3000):
    loss = F.mse_loss(net(xt), yt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The torch version shows the same pattern: MSE around 0.11 with 2 units, around `1e-4` with 8, and near zero with 32. The exact numbers differ slightly because the initializations differ. In practice you will rarely think about the theorem; you will think about its gaps. When a model fails, ask which one bit you: not enough capacity, an optimizer that cannot find good weights, or data that does not cover where you are predicting.

## Exercises

1. **Easy.** Change the target to `np.abs(x)` and find the smallest number of ReLU units that represents it exactly on `[-2, 2]`. (Hint: one.)
2. **Medium.** Replace the tanh hidden layer in `train_one_hidden_layer` with ReLU. Compare the MSE for 2, 8 and 32 units and the shape of the extrapolation at `x = 3, 4`. Why does a ReLU network extrapolate linearly?
3. **Hard.** Build a 2D bump: a "tower" that is 1 inside the square `[-0.5, 0.5]^2` and 0 outside, using sigmoid hidden units for each edge and a second sigmoid layer that fires only when all four agree. Visualize it with `plt.imshow` and explain why two layers make 2D approximation much easier than one.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Universal approximation theorem | "Neural nets can learn anything" | A one-hidden-layer network with a nonlinear activation can approximate any continuous function on a compact set to any accuracy, given enough units |
| Bump | "A localized feature" | The difference of two shifted steep sigmoids, a unit that responds only on an interval |
| Piecewise-linear | "Made of straight segments" | A function that is linear between knots; exactly what ReLU networks compute |
| Capacity | "Model size" | The set of functions a model can represent; grows with width and depth |
| Expressivity vs learnability | "Can vs will" | Whether good weights exist versus whether training actually finds them |
| Extrapolation | "Predicting outside the data" | Evaluating a model where it saw no training data; approximation theorems say nothing about it |
| Depth efficiency | "Why go deep" | Some functions need exponentially more units in a shallow network than in a deep one |

## Further Reading

- [Cybenko (1989), Approximation by superpositions of a sigmoidal function](https://link.springer.com/article/10.1007/BF02551274) - the original theorem for sigmoid networks
- [Hornik (1991), Approximation capabilities of multilayer feedforward networks](https://www.sciencedirect.com/science/article/abs/pii/089360809190009T) - generalizes the result to any non-constant bounded activation
- [Michael Nielsen, Neural Networks and Deep Learning, Chapter 4](http://neuralnetworksanddeeplearning.com/chap4.html) - the visual bump construction this lesson follows
- [Telgarsky (2016), Benefits of depth in neural networks](https://arxiv.org/abs/1602.04485) - functions that deep networks represent compactly and shallow ones cannot
