import math
import time

import torch

FORWARD_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "log": lambda a: math.log(a),
    "sin": lambda a: math.sin(a),
    "square": lambda a: a * a,
}

LOCAL_DERIVATIVES = {
    "add": lambda a, b: (1.0, 1.0),
    "sub": lambda a, b: (1.0, -1.0),
    "mul": lambda a, b: (b, a),
    "log": lambda a: (1.0 / a,),
    "sin": lambda a: (math.cos(a),),
    "square": lambda a: (2.0 * a,),
}


class Node:
    def __init__(self, name, op, parents):
        self.name = name
        self.op = op
        self.parents = parents
        self.value = None


class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []

    def input(self, name):
        node = Node(name, "input", [])
        self.nodes.append(node)
        self.inputs.append(node)
        return node

    def apply(self, op, *parents):
        node = Node(f"v{len(self.nodes) - len(self.inputs) + 1}", op, list(parents))
        self.nodes.append(node)
        return node

    def forward(self, values):
        for node, value in zip(self.inputs, values):
            node.value = value
        for node in self.nodes:
            if node.op != "input":
                node.value = FORWARD_OPS[node.op](*[p.value for p in node.parents])
        return self.nodes[-1].value

    def forward_mode(self, seed_input):
        tangent = {id(node): 0.0 for node in self.nodes}
        tangent[id(seed_input)] = 1.0
        for node in self.nodes:
            if node.op == "input":
                continue
            partials = LOCAL_DERIVATIVES[node.op](*[p.value for p in node.parents])
            tangent[id(node)] = sum(d * tangent[id(p)] for d, p in zip(partials, node.parents))
        return tangent[id(self.nodes[-1])]

    def reverse_mode(self):
        adjoint = {id(node): 0.0 for node in self.nodes}
        adjoint[id(self.nodes[-1])] = 1.0
        for node in reversed(self.nodes):
            if node.op == "input":
                continue
            partials = LOCAL_DERIVATIVES[node.op](*[p.value for p in node.parents])
            for d, parent in zip(partials, node.parents):
                adjoint[id(parent)] += adjoint[id(node)] * d
        return [adjoint[id(node)] for node in self.inputs]

    def trace(self):
        for node in self.nodes:
            args = ", ".join(p.name for p in node.parents)
            expression = node.name if node.op == "input" else f"{node.op}({args})"
            print(f"  {node.name:<4} = {expression:<14} = {node.value:9.4f}")


def build_example():
    graph = Graph()
    x1 = graph.input("x1")
    x2 = graph.input("x2")
    v1 = graph.apply("log", x1)
    v2 = graph.apply("mul", x1, x2)
    v3 = graph.apply("sin", x2)
    v4 = graph.apply("add", v1, v2)
    graph.apply("sub", v4, v3)
    return graph


def example_function(x1, x2):
    return math.log(x1) + x1 * x2 - math.sin(x2)


def build_sum_of_squares(n_inputs):
    graph = Graph()
    xs = [graph.input(f"x{i}") for i in range(n_inputs)]
    total = graph.apply("square", xs[0])
    for x in xs[1:]:
        total = graph.apply("add", total, graph.apply("square", x))
    return graph


def numerical_gradient(f, point, eps=1e-6):
    grads = []
    for i in range(len(point)):
        plus = list(point)
        minus = list(point)
        plus[i] += eps
        minus[i] -= eps
        grads.append((f(*plus) - f(*minus)) / (2 * eps))
    return grads


def main():
    point = (2.0, 5.0)

    print("=== Step 1: build the graph and run the forward pass ===")
    graph = build_example()
    value = graph.forward(point)
    graph.trace()
    print(f"  f(2, 5) = {value:.4f}   direct formula = {example_function(*point):.4f}")

    print("\n=== Step 2: forward mode, one sweep per input ===")
    forward_grads = [graph.forward_mode(x) for x in graph.inputs]
    print(f"  df/dx1 = {forward_grads[0]:.4f}   df/dx2 = {forward_grads[1]:.4f}")

    print("\n=== Step 3: reverse mode, one sweep for all inputs ===")
    reverse_grads = graph.reverse_mode()
    print(f"  df/dx1 = {reverse_grads[0]:.4f}   df/dx2 = {reverse_grads[1]:.4f}")
    print(f"  by hand: df/dx1 = 1/x1 + x2 = {1 / 2 + 5:.4f}   df/dx2 = x1 - cos(x2) = {2 - math.cos(5):.4f}")

    print("\n=== Step 4: check against finite differences ===")
    numeric = numerical_gradient(example_function, point)
    print(f"  numeric: df/dx1 = {numeric[0]:.4f}   df/dx2 = {numeric[1]:.4f}")
    print(f"  max abs error of reverse mode = {max(abs(a - b) for a, b in zip(numeric, reverse_grads)):.2e}")

    print("\n=== Step 5: why deep learning uses reverse mode ===")
    for n_inputs in [10, 100, 400]:
        big = build_sum_of_squares(n_inputs)
        values = [0.01 * i for i in range(n_inputs)]
        big.forward(values)
        start = time.perf_counter()
        forward_all = [big.forward_mode(x) for x in big.inputs]
        forward_time = time.perf_counter() - start
        start = time.perf_counter()
        reverse_all = big.reverse_mode()
        reverse_time = time.perf_counter() - start
        agree = max(abs(a - b) for a, b in zip(forward_all, reverse_all))
        print(f"  {n_inputs:4d} inputs | forward mode {n_inputs:4d} sweeps {forward_time * 1000:8.2f} ms | reverse mode 1 sweep {reverse_time * 1000:6.2f} ms | max diff {agree:.1e}")

    print("\n=== Use It: PyTorch records the same graph ===")
    x1 = torch.tensor(2.0, requires_grad=True)
    x2 = torch.tensor(5.0, requires_grad=True)
    f = torch.log(x1) + x1 * x2 - torch.sin(x2)
    frontier = [(f.grad_fn, 0)]
    while frontier:
        fn, depth = frontier.pop()
        print(f"  {'  ' * depth}{type(fn).__name__}")
        for child, _ in reversed(fn.next_functions):
            if child is not None:
                frontier.append((child, depth + 1))
    f.backward()
    print(f"  torch: f = {f.item():.4f}, df/dx1 = {x1.grad.item():.4f}, df/dx2 = {x2.grad.item():.4f}")


if __name__ == "__main__":
    main()
