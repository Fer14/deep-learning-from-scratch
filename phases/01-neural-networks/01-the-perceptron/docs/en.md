# The Perceptron

> One neuron, one line: a weighted vote that learns from its own mistakes.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 00 (Tensors and Autodiff)
**Time:** ~45 minutes

## Learning Objectives

- Describe an artificial neuron as a weighted sum, a bias and an activation
- Implement Rosenblatt's mistake-driven perceptron learning rule in NumPy
- Explain why the perceptron converges on linearly separable data and cycles forever on XOR
- Connect the perceptron to `torch.nn.Linear`, the building block of every network in this course

## The Problem

Every deep network you will build in this course, from a digit classifier to a GPT, is made of the same small unit: take some numbers, multiply each by a weight, add them up, add a bias, and pass the result through a function. If you do not understand what one of those units can and cannot do on its own, the rest of deep learning looks like magic.

The perceptron (Rosenblatt, 1958) is that unit in its oldest form. It was the first machine that learned its weights from data, and it caused the first AI hype cycle. It also caused the first AI winter, when Minsky and Papert showed in 1969 that a single perceptron cannot learn something as simple as XOR. Both halves of that story matter: the learning rule shows you that weights can be found automatically, and the XOR failure is the reason networks have hidden layers.

## The Concept

### A neuron is a weighted vote

```
 x1 ---- w1 ---.
                \
 x2 ---- w2 -----( sum + b )---> step ---> y_hat in {-1, +1}
                /
 x3 ---- w3 ---'
```

```
z     = w1 x1 + w2 x2 + ... + wd xd + b = w^T x + b
y_hat = +1 if z >= 0 else -1
```

Each weight says how much a feature votes for the positive class; the bias shifts the threshold. The set of points where `z = 0` is a hyperplane (a line in 2D), so a perceptron is a linear classifier.

### Hand-wiring a logic gate

You can pick weights by hand. With `w = [1, 1]` and `b = -1.5`, the neuron only fires when both inputs are 1: that is AND. With `b = -0.5` it fires when at least one input is 1: that is OR. Each gate is a line through the unit square.

```
 x2                      x2
 1 |  o     *            1 |  *     *
   |     \                 |  \
 0 |  o    o\            0 |  o\    *
   +---------- x1          +---------- x1
      0     1                 0     1
      AND                      OR
```

### The learning rule

Rosenblatt's rule only reacts to mistakes. Using labels in `{-1, +1}`, a point is misclassified when `y (w^T x + b) <= 0`. Then:

```
w <- w + lr * y * x
b <- b + lr * y
```

If a positive point was predicted negative, you add `x` to `w`, which rotates the boundary toward classifying it correctly. Correct points cause no update at all.

### Convergence and the XOR wall

The perceptron convergence theorem says: if some hyperplane separates the data with margin `gamma`, and all points lie within radius `R`, the algorithm makes at most `(R / gamma)^2` mistakes. On separable data it always stops.

XOR is not separable:

```
 x2
 1 |  *     o        * = class +1
   |                 o = class -1
 0 |  o     *
   +---------- x1
      0     1
```

No single line puts both stars on one side and both circles on the other. The perceptron keeps fixing one corner and breaking another, and never stops. The fix, which you build in the next lesson, is to stack neurons into layers.

## Build It

### Step 1: A single neuron

```python
def step(z):
    return np.where(z >= 0, 1.0, -1.0)


w, b = np.array([1.0, 1.0]), -1.5
for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    z = np.dot(w, x) + b
    print(f"  x={x} z={z:+.1f} output={int(step(z))}")
```

Only `[1, 1]` gives a positive `z`: the neuron computes AND.

### Step 2: The perceptron learner

```python
class PerceptronScratch:
    def __init__(self, learning_rate=1.0, max_epochs=50):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.mistakes_per_epoch = []

    def predict(self, X):
        return step(X @ self.weights + self.bias)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.mistakes_per_epoch = []
        for _ in range(self.max_epochs):
            mistakes = 0
            for x_i, y_i in zip(X, y):
                if y_i * (x_i @ self.weights + self.bias) <= 0:
                    self.weights += self.learning_rate * y_i * x_i
                    self.bias += self.learning_rate * y_i
                    mistakes += 1
            self.mistakes_per_epoch.append(mistakes)
            if mistakes == 0:
                break
        return self
```

The `<= 0` matters: at initialization every score is 0, and you want those points to count as mistakes so learning can start.

### Step 3: Separable data

Two well separated Gaussian blobs, 100 points each:

```
  mistakes per epoch: [2, 0]
  converged after 2 epochs
  training accuracy = 1.000
```

Two mistakes were enough to find a separating line; the second epoch confirms it.

### Step 4: Logic gates and XOR

```python
for gate in ["AND", "OR", "NAND", "XOR"]:
    Xg, yg = logic_gate_data(gate)
    gate_model = PerceptronScratch(max_epochs=100).fit(Xg, yg)
```

```
  AND  accuracy=1.00 converged=True epochs=9
  OR   accuracy=1.00 converged=True epochs=6
  NAND accuracy=1.00 converged=True epochs=9
  XOR  accuracy=0.50 converged=False epochs=100
```

For XOR the mistake count stays at 4 every epoch: the weights cycle and the algorithm never settles.

## Use It

scikit-learn ships the same algorithm, and in PyTorch the weighted sum is `torch.nn.Linear`:

```python
from sklearn.linear_model import Perceptron

sk = Perceptron(eta0=1.0, max_iter=50, tol=None, shuffle=False).fit(X, y)
print(sk.score(X, y))

neuron = torch.nn.Linear(2, 1)
with torch.no_grad():
    neuron.weight.copy_(torch.tensor(model.weights, dtype=torch.float32).unsqueeze(0))
    neuron.bias.fill_(model.bias)
    torch_pred = torch.where(neuron(X_tensor).squeeze(1) >= 0, 1.0, -1.0)
```

Both agree with your scratch model on every point. Note the shape: `nn.Linear` stores its weight as `(out_features, in_features)` and computes `x @ W.T + b`. Modern networks keep the `Linear` part and replace the step with a smooth activation, because the step has zero gradient almost everywhere and gradient descent cannot learn through it. That replacement is what makes backpropagation possible.

## Exercises

1. **Easy.** Find by hand a weight vector and bias that implement NOR (output +1 only for `[0, 0]`). Check it with the `step` function.
2. **Medium.** Implement the averaged perceptron: keep a running sum of `w` and `b` after every sample and predict with the average. Compare test accuracy with the plain perceptron on two overlapping blobs (scale 1.5), where the plain version never converges.
3. **Hard.** Verify the convergence bound empirically: compute `R = max ||x||` and estimate `gamma` with a hard-margin linear SVM (`sklearn.svm.SVC(kernel="linear", C=1e6)`) on data with the bias folded into `x`. Count total mistakes and compare with `(R / gamma)^2` as you move the blobs closer together.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Neuron | "A brain cell in code" | A weighted sum of inputs plus a bias, passed through an activation function |
| Perceptron | "The first neural network" | A single neuron with a step activation, trained by the mistake-driven update `w += lr * y * x` |
| Bias | "The offset" | A learned constant that shifts the decision threshold away from the origin |
| Linearly separable | "Easy data" | A hyperplane exists that puts every point of each class on its own side |
| Decision boundary | "Where the model flips" | The set of inputs where `w^T x + b = 0` |
| XOR problem | "Why neural nets died in 1969" | The proof that one linear unit cannot represent XOR, which motivates hidden layers |

## Further Reading

- [Rosenblatt (1958), The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://psycnet.apa.org/doi/10.1037/h0042519) - the original paper
- [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) - neurons, weights and biases explained visually
- [Dive into Deep Learning, Chapter 5: Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html) - from linear models to hidden layers
- [Classic ML from Scratch: Perceptron](https://github.com/Fer14/classic-ml-from-scratch) - the classic-ML view with the averaged variant and margins
