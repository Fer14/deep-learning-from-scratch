import numpy as np
import torch
from sklearn.linear_model import Perceptron as SklearnPerceptron


def make_separable_data(n_per_class, seed):
    rng = np.random.default_rng(seed)
    class_neg = rng.normal(loc=[-2.0, -1.5], scale=0.8, size=(n_per_class, 2))
    class_pos = rng.normal(loc=[2.0, 1.5], scale=0.8, size=(n_per_class, 2))
    X = np.vstack([class_neg, class_pos])
    y = np.concatenate([-np.ones(n_per_class), np.ones(n_per_class)])
    order = rng.permutation(len(y))
    return X[order], y[order]


def step(z):
    return np.where(z >= 0, 1.0, -1.0)


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


def logic_gate_data(gate):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    table = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0], "XOR": [0, 1, 1, 0]}
    y = np.array(table[gate], dtype=float) * 2 - 1
    return X, y


def main():
    print("=== Step 1: a single neuron is a weighted sum plus a step ===")
    w, b = np.array([1.0, 1.0]), -1.5
    for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        z = np.dot(w, x) + b
        print(f"  x={x} z={z:+.1f} output={int(step(z))}")

    print("\n=== Step 2: train on linearly separable blobs ===")
    X, y = make_separable_data(n_per_class=100, seed=0)
    model = PerceptronScratch().fit(X, y)
    print(f"  mistakes per epoch: {model.mistakes_per_epoch}")
    print(f"  converged after {len(model.mistakes_per_epoch)} epochs")
    print(f"  weights = {np.round(model.weights, 3)}, bias = {model.bias:.3f}")
    print(f"  training accuracy = {np.mean(model.predict(X) == y):.3f}")

    print("\n=== Step 3: logic gates ===")
    for gate in ["AND", "OR", "NAND", "XOR"]:
        Xg, yg = logic_gate_data(gate)
        gate_model = PerceptronScratch(max_epochs=100).fit(Xg, yg)
        acc = np.mean(gate_model.predict(Xg) == yg)
        converged = gate_model.mistakes_per_epoch[-1] == 0
        print(f"  {gate:4s} accuracy={acc:.2f} converged={converged} epochs={len(gate_model.mistakes_per_epoch)}")

    print("\n=== Step 4: why XOR fails ===")
    Xg, yg = logic_gate_data("XOR")
    xor_model = PerceptronScratch(max_epochs=100).fit(Xg, yg)
    print(f"  last 10 epoch mistake counts: {xor_model.mistakes_per_epoch[-10:]}")
    print("  no line separates {(0,1),(1,0)} from {(0,0),(1,1)}, so updates cycle forever")

    print("\n=== Use It: scikit-learn and a torch neuron ===")
    sk = SklearnPerceptron(eta0=1.0, max_iter=50, tol=None, shuffle=False, random_state=0).fit(X, y)
    print(f"  sklearn training accuracy = {sk.score(X, y):.3f}")
    torch.manual_seed(0)
    neuron = torch.nn.Linear(2, 1)
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor(model.weights, dtype=torch.float32).unsqueeze(0))
        neuron.bias.fill_(model.bias)
        torch_pred = torch.where(neuron(torch.tensor(X, dtype=torch.float32)).squeeze(1) >= 0, 1.0, -1.0)
    agreement = np.mean(torch_pred.numpy() == model.predict(X))
    print(f"  nn.Linear with the learned weights + step agrees with scratch: {agreement:.3f}")


if __name__ == "__main__":
    main()
