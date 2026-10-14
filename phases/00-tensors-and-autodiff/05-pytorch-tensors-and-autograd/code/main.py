import numpy as np
import torch


class ScaledSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return torch.log1p(torch.exp(beta * x)) / beta

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * torch.sigmoid(ctx.beta * x), None


def make_regression_data(n_samples, generator):
    X = torch.randn(n_samples, 3, generator=generator)
    true_w = torch.tensor([2.0, -1.0, 0.5])
    y = X @ true_w + 0.3 + 0.1 * torch.randn(n_samples, generator=generator)
    return X, y, true_w


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("=== Step 1: creating tensors ===")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    zeros = torch.zeros(2, 3)
    ints = torch.arange(6)
    rand = torch.rand(2, 2)
    for name, t in [("tensor", a), ("zeros", zeros), ("arange", ints), ("rand", rand)]:
        print(f"  {name:<7} shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    print(f"  int + float promotes to {(ints + 0.5).dtype}; float32 default, float64 from numpy: {torch.from_numpy(np.ones(2)).dtype}")

    print("\n=== Step 2: requires_grad and backward ===")
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x
    print(f"  y = {y.item()}, y.grad_fn = {type(y.grad_fn).__name__}, x is leaf: {x.is_leaf}, y is leaf: {y.is_leaf}")
    y.backward()
    print(f"  dy/dx at 3 = {x.grad.item()} (expected 2x + 2 = 8)")

    print("\n=== Step 3: gradients accumulate until you zero them ===")
    x.grad = None
    for call in range(3):
        y = x ** 2 + 2 * x
        y.backward()
        print(f"  after backward call {call + 1}: x.grad = {x.grad.item()}")
    x.grad = None
    (x ** 2 + 2 * x).backward()
    print(f"  after resetting x.grad = None: x.grad = {x.grad.item()}")

    print("\n=== Step 4: non-scalar outputs need a vector (vector-Jacobian product) ===")
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = v ** 2
    weights = torch.tensor([1.0, 0.1, 0.01])
    out.backward(gradient=weights)
    print(f"  out = v**2, backward(gradient={weights.numpy().round(3)}) -> v.grad = {v.grad.numpy().round(3)} (= 2v * weights)")
    jacobian = torch.autograd.functional.jacobian(lambda t: t ** 2, v.detach())
    print(f"  full Jacobian diagonal = {torch.diagonal(jacobian).numpy()}; weights @ J = {(weights @ jacobian).numpy().round(3)}")

    print("\n=== Step 5: torch.autograd.grad and higher-order derivatives ===")
    z = torch.tensor(2.0, requires_grad=True)
    f = z ** 3
    (first,) = torch.autograd.grad(f, z, create_graph=True)
    (second,) = torch.autograd.grad(first, z)
    print(f"  f = z^3 at z = 2: df/dz = {first.item()} (3z^2), d2f/dz2 = {second.item()} (6z)")

    print("\n=== Step 6: no_grad, detach and the graph lifetime ===")
    w = torch.randn(3, requires_grad=True)
    with torch.no_grad():
        prediction = w * 2
    print(f"  inside no_grad: requires_grad = {prediction.requires_grad}")
    detached = w.detach()
    print(f"  w.detach(): requires_grad = {detached.requires_grad}, same storage as w: {detached.data_ptr() == w.data_ptr()}")
    loss = (w ** 2).sum()
    loss.backward(retain_graph=True)
    loss.backward()
    print(f"  backward twice with retain_graph=True works, w.grad = 2 * 2w: {torch.allclose(w.grad, 4 * w.detach())}")
    loss = (w.exp()).sum()
    loss.backward()
    try:
        loss.backward()
    except RuntimeError as error:
        print(f"  second backward without retain_graph raises: {str(error).splitlines()[0][:70]}...")

    print("\n=== Step 7: a custom autograd.Function ===")
    inputs = torch.randn(5, dtype=torch.float64, requires_grad=True)
    passed = torch.autograd.gradcheck(lambda t: ScaledSoftplus.apply(t, 2.0), (inputs,))
    print(f"  gradcheck on ScaledSoftplus: {passed}")
    reference = torch.nn.functional.softplus(inputs, beta=2.0)
    print(f"  matches F.softplus(beta=2): {torch.allclose(ScaledSoftplus.apply(inputs, 2.0), reference)}")

    print("\n=== Step 8: linear regression with raw autograd ===")
    generator = torch.Generator().manual_seed(0)
    X, y, true_w = make_regression_data(200, generator)
    weight = torch.zeros(3, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    for step in range(200):
        loss = ((X @ weight + bias - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            weight -= 0.1 * weight.grad
            bias -= 0.1 * bias.grad
        weight.grad = None
        bias.grad = None
        if step % 50 == 0:
            print(f"  step {step:3d} | mse {loss.item():.4f}")
    print(f"  learned w = {np.round(weight.detach().numpy(), 3)}, b = {bias.item():.3f}; true w = {true_w.tolist()}, b = 0.3")

    print("\n=== Use It: the same loop with torch.optim ===")
    weight = torch.zeros(3, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([weight, bias], lr=0.1)
    for step in range(200):
        optimizer.zero_grad()
        loss = ((X @ weight + bias - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    print(f"  torch.optim.SGD: w = {np.round(weight.detach().numpy(), 3)}, b = {bias.item():.3f}, final mse {loss.item():.4f}")


if __name__ == "__main__":
    main()
