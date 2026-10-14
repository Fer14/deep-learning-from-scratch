import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.datasets import load_digits


def load_data():
    digits = load_digits()
    x = torch.tensor(digits.data, dtype=torch.float32) / 16.0
    y = torch.tensor(digits.target)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    x, y = x[order], y[order]
    return (x[:1400], y[:1400]), (x[1400:], y[1400:])


def make_mlp(hidden=256):
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def linears(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def train(model, data, epochs, masks=None, lr=1e-3):
    x, y = data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for i in range(0, len(x), 64):
            loss = F.cross_entropy(model(x[i:i + 64]), y[i:i + 64])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if masks is not None:
                apply_masks(model, masks)
    return model.eval()


def accuracy(model, data):
    x, y = data
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def layerwise_masks(model, sparsity):
    masks = []
    for layer in linears(model):
        magnitude = layer.weight.detach().abs()
        k = int(sparsity * magnitude.numel())
        threshold = magnitude.flatten().kthvalue(k).values if k > 0 else -1.0
        masks.append((magnitude > threshold).float())
    return masks


def global_masks(model, sparsity):
    all_magnitudes = torch.cat([layer.weight.detach().abs().flatten() for layer in linears(model)])
    k = int(sparsity * all_magnitudes.numel())
    threshold = all_magnitudes.kthvalue(k).values
    return [(layer.weight.detach().abs() > threshold).float() for layer in linears(model)]


def apply_masks(model, masks):
    with torch.no_grad():
        for layer, mask in zip(linears(model), masks):
            layer.weight.mul_(mask)


def pruned_copy(model, masks):
    pruned = copy.deepcopy(model)
    apply_masks(pruned, masks)
    return pruned


def sparsity_of(model):
    weights = [layer.weight for layer in linears(model)]
    zeros = sum((w == 0).sum().item() for w in weights)
    return zeros / sum(w.numel() for w in weights)


def iterative_pruning(model, data, final_sparsity, rounds, epochs_per_round):
    model = copy.deepcopy(model)
    for r in range(1, rounds + 1):
        sparsity = final_sparsity * (1 - (1 - r / rounds) ** 3)
        masks = global_masks(model, sparsity)
        apply_masks(model, masks)
        train(model, data, epochs_per_round, masks=masks)
    return model


def structured_prune(model, keep):
    first, second, last = linears(model)
    importance = first.weight.detach().norm(dim=1) * second.weight.detach().norm(dim=0)
    keep_1 = importance.topk(keep).indices.sort().values
    importance_2 = second.weight.detach()[:, keep_1].norm(dim=1) * last.weight.detach().norm(dim=0)
    keep_2 = importance_2.topk(keep).indices.sort().values
    small = make_mlp(hidden=keep)
    new_first, new_second, new_last = linears(small)
    with torch.no_grad():
        new_first.weight.copy_(first.weight[keep_1])
        new_first.bias.copy_(first.bias[keep_1])
        new_second.weight.copy_(second.weight[keep_2][:, keep_1])
        new_second.bias.copy_(second.bias[keep_2])
        new_last.weight.copy_(last.weight[:, keep_2])
        new_last.bias.copy_(last.bias)
    return small.eval()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def time_forward(model, x, repeats=300):
    with torch.no_grad():
        for _ in range(20):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
    return (time.perf_counter() - start) / repeats


def csr_bytes(model):
    total = 0
    for layer in linears(model):
        nnz = int((layer.weight != 0).sum())
        total += nnz * (4 + 4) + (layer.weight.shape[0] + 1) * 4
    return total


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    train_data, test_data = load_data()
    dense = train(make_mlp(), train_data, epochs=40)
    base = accuracy(dense, test_data)
    print(f"Dense MLP 64-256-256-10: {n_params(dense)} parameters, test accuracy {base:.4f}\n")

    print("1) One-shot magnitude pruning, no fine-tuning")
    print(f"  {'sparsity':>8s} {'layer-wise':>11s} {'global':>8s}   global sparsity per layer")
    for sparsity in [0.5, 0.8, 0.9, 0.95, 0.98]:
        layer_model = pruned_copy(dense, layerwise_masks(dense, sparsity))
        masks = global_masks(dense, sparsity)
        global_model = pruned_copy(dense, masks)
        per_layer = ", ".join(f"{1 - m.mean().item():.2f}" for m in masks)
        print(f"  {sparsity:8.2f} {accuracy(layer_model, test_data):11.4f} {accuracy(global_model, test_data):8.4f}   [{per_layer}]")

    print("\n2) Iterative pruning: prune a little, fine-tune with the mask fixed, repeat")
    for final in [0.95, 0.98]:
        start = time.perf_counter()
        sparse = iterative_pruning(dense, train_data, final, rounds=8, epochs_per_round=10)
        print(f"  target {final:.2f}: actual sparsity {sparsity_of(sparse):.3f}, accuracy {accuracy(sparse, test_data):.4f} "
              f"(one-shot was {accuracy(pruned_copy(dense, global_masks(dense, final)), test_data):.4f}), "
              f"{time.perf_counter() - start:.1f}s")
    print(f"  storage at 98%: dense fp32 {sum(l.weight.numel() for l in linears(sparse)) * 4} bytes, CSR {csr_bytes(sparse)} bytes")

    print("\n3) Unstructured zeros do not make a dense matmul faster; removing neurons does")
    x = test_data[0].repeat(8, 1)
    print(f"  {'model':34s} {'params':>8s} {'accuracy':>9s} {'ms/batch':>9s}")
    print(f"  {'dense':34s} {n_params(dense):8d} {base:9.4f} {time_forward(dense, x) * 1e3:9.3f}")
    print(f"  {'98% unstructured (zeros in place)':34s} {n_params(sparse):8d} {accuracy(sparse, test_data):9.4f} {time_forward(sparse, x) * 1e3:9.3f}")
    for keep in [64, 32]:
        small = structured_prune(dense, keep)
        before = accuracy(small, test_data)
        train(small, train_data, epochs=10)
        print(f"  {f'structured, {keep} neurons/layer':34s} {n_params(small):8d} {accuracy(small, test_data):9.4f} "
              f"{time_forward(small, x) * 1e3:9.3f}   (before fine-tune {before:.4f})")

    print("\n4) Use It: torch.nn.utils.prune")
    library = copy.deepcopy(dense)
    parameters = [(layer, "weight") for layer in linears(library)]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=0.9)
    library_masks = [layer.weight_mask for layer in linears(library)]
    ours = global_masks(dense, 0.9)
    same = all(torch.equal(a, b) for a, b in zip(library_masks, ours))
    print(f"  same masks as our global pruning at 90%: {same}")
    print(f"  pruned layer keeps 'weight_orig' and 'weight_mask': {sorted(n for n, _ in linears(library)[0].named_buffers())}, "
          f"{sorted(n for n, _ in linears(library)[0].named_parameters())}")
    for layer, name in parameters:
        prune.remove(layer, name)
    print(f"  after prune.remove: sparsity {sparsity_of(library):.3f}, accuracy {accuracy(library, test_data):.4f}")


if __name__ == "__main__":
    main()
