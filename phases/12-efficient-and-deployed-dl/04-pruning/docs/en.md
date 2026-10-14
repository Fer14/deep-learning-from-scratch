# Pruning

> Most weights in a trained network barely matter: remove them gradually, let the rest compensate, and keep the speedup only if you remove whole neurons.

**Type:** Build
**Languages:** Python
**Prerequisites:** Profiling, Quantization (this phase), Phase 02 (Training Deep Nets: Dropout and Weight Decay)
**Time:** ~60 minutes

## Learning Objectives

- Implement magnitude pruning with layer-wise and global thresholds and explain why global works better
- Use binary masks to fine-tune a pruned network without letting pruned weights come back
- Show why iterative pruning with a gradual (cubic) schedule beats one-shot pruning at high sparsity
- Distinguish unstructured from structured pruning, and measure which one actually makes inference faster
- Reproduce your masks with `torch.nn.utils.prune`

## The Problem

A trained network is heavily over-parameterized. That helps optimization, but at deployment time you pay for every weight in memory and, often, in compute. Pruning removes weights that contribute little. The classic results are dramatic: Han et al. (2015) removed 90% of AlexNet's weights with no accuracy loss, and the lottery ticket hypothesis (Frankle and Carbin, 2018) suggests that dense networks contain small subnetworks that could have done the job on their own.

The fine print is just as important. Removing weights one at a time leaves a matrix full of scattered zeros, and a normal dense matrix multiply does exactly the same work whether a weight is zero or not. To get real speed you either need sparse kernels and hardware, or you remove whole rows and columns so the matrices themselves get smaller. This lesson builds both and measures the difference.

## The Concept

### Magnitude as importance

The simplest importance score is `|w|`: a weight close to zero changes the output little when you remove it. Pick a sparsity `s`, find the threshold below which a fraction `s` of the weights fall, and zero them:

```
mask = |W| > threshold        W_pruned = W * mask
```

**Layer-wise** pruning uses one threshold per layer, so every layer loses the same fraction. **Global** pruning uses one threshold for all layers, so layers with many redundant weights lose more and small, critical layers lose less. In this lesson's MLP the middle 256x256 layer is the most redundant; global pruning at 90% removes 94% of it but only 75% of the input layer.

### Masks during fine-tuning

After pruning, a few epochs of training let the remaining weights compensate. The optimizer would happily make pruned weights nonzero again, so you reapply the mask after every step:

```
for each batch:
    loss.backward(); optimizer.step()
    W <- W * mask
```

### Iterative pruning and the cubic schedule

Pruning 98% in one shot destroys the network faster than fine-tuning can repair it. Iterative pruning raises sparsity over several rounds, fine-tuning in between. The schedule matters. Zhu and Gupta (2017) proposed pruning fast at first, when there are many redundant weights, and slowly at the end:

```
s_r = s_final * (1 - (1 - r / R)^3)       r = 1..R

round       1     2     3     4     5     6     7     8
linear    .12   .25   .37   .49   .61   .74   .86   .98
cubic     .32   .57   .74   .86   .93   .96   .98   .98
```

The cubic schedule spends the last rounds making tiny changes, which gives the network time to adapt when every remaining weight matters.

### Unstructured versus structured

```
unstructured (scattered zeros)        structured (whole neurons removed)
[ w 0 w 0 w ]                         [ w w w ]
[ 0 0 w w 0 ]          -->            [ w w w ]       smaller dense matrix
[ w 0 0 0 w ]
same shape, same dense FLOPs          fewer rows and columns, fewer FLOPs
```

| | Unstructured | Structured |
|---|---|---|
| Granularity | single weights | neurons, channels, heads, layers |
| Accuracy at a given parameter count | higher | lower |
| Speedup with dense kernels | none | proportional to the size reduction |
| Needs special support | sparse kernels or formats | nothing |

A middle ground is N:M sparsity (for example 2 zeros in every block of 4), which NVIDIA GPUs since Ampere accelerate in hardware.

## Build It

### Step 1: Baseline

The same MLP as the quantization lesson: 64-256-256-10, 85,002 parameters, 97.5% test accuracy.

### Step 2: Layer-wise and global masks

```python
def global_masks(model, sparsity):
    all_magnitudes = torch.cat([layer.weight.detach().abs().flatten() for layer in linears(model)])
    k = int(sparsity * all_magnitudes.numel())
    threshold = all_magnitudes.kthvalue(k).values
    return [(layer.weight.detach().abs() > threshold).float() for layer in linears(model)]
```

`kthvalue` finds the k-th smallest magnitude without sorting everything. Without any fine-tuning, both methods keep accuracy at 50% sparsity, but global is far better beyond that: 82% versus 62% accuracy at 90% sparsity. The printed per-layer sparsities show global pruning hitting the middle layer hardest.

### Step 3: Iterative pruning

```python
def iterative_pruning(model, data, final_sparsity, rounds, epochs_per_round):
    model = copy.deepcopy(model)
    for r in range(1, rounds + 1):
        sparsity = final_sparsity * (1 - (1 - r / rounds) ** 3)
        masks = global_masks(model, sparsity)
        apply_masks(model, masks)
        train(model, data, epochs_per_round, masks=masks)
    return model
```

`train` calls `apply_masks` after every optimizer step. Eight rounds of 10 epochs each give 96% accuracy at 95% sparsity and 94% at 98% sparsity, where one-shot pruning had collapsed to 49% and 13%. (With a linear schedule, the 98% model only reaches about 80%; try it.) At 98% sparsity the model has about 1,700 nonzero weights. Stored in CSR format (values, column indices, row pointers) the weights take 15.6 KB instead of 338 KB.

### Step 4: Does it run faster?

The 98% sparse model still stores dense matrices full of zeros, and its forward pass takes about the same time as the dense one. `structured_prune` instead ranks hidden neurons by the product of their incoming and outgoing weight norms, keeps the top 64 (or 32) per layer, and copies the surviving rows and columns into a genuinely smaller `make_mlp(hidden=keep)`. Before fine-tuning, accuracy drops to 65% (64 neurons); after 10 epochs it is back to 96%, with about 10x fewer parameters and a forward pass more than 10x faster.

## Use It

```python
parameters = [(layer, "weight") for layer in linears(library)]
prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=0.9)
...
prune.remove(layer, "weight")
```

`torch.nn.utils.prune` produces exactly the same masks as `global_masks`. It works by reparameterizing: the module gets a `weight_orig` parameter and a `weight_mask` buffer, and a forward pre-hook recomputes `weight = weight_orig * weight_mask` before every call, so gradients automatically skip pruned entries. `prune.remove` makes the pruning permanent by baking the mask into a plain `weight`. The module also offers `ln_structured` for removing whole channels by norm, but note that it only zeros them; to get the speedup you still have to rebuild smaller layers as `structured_prune` does. For sparse execution, PyTorch has `torch.sparse` tensors and semi-structured 2:4 sparsity on supported GPUs.

## Exercises

1. **Easy.** Change the schedule in `iterative_pruning` back to linear (`final_sparsity * r / rounds`) and compare accuracy at 98%. Then try 3 rounds and 16 rounds with the cubic schedule.
2. **Medium.** Lottery ticket: after finding the 95% mask, reset the surviving weights to their values at initialization (save them before training) and retrain from scratch with the mask fixed. Compare with a random mask of the same sparsity.
3. **Hard.** Implement 2:4 semi-structured pruning: in every group of 4 consecutive weights in a row, keep the 2 with the largest magnitude. Fine-tune and compare accuracy with unstructured pruning at 50% sparsity.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pruning | "Cutting weights" | Removing weights or structures from a trained network, usually followed by fine-tuning |
| Sparsity | "Percent zeros" | The fraction of weights that are exactly zero |
| Magnitude pruning | "Drop small weights" | Using `\|w\|` as the importance score and removing the smallest |
| Global pruning | "One threshold" | A single magnitude threshold across all layers, so sparsity varies by layer |
| Mask | "Which weights survive" | A binary tensor multiplied into the weights to keep pruned entries at zero |
| Iterative pruning | "Prune, retrain, repeat" | Gradually increasing sparsity with fine-tuning between steps |
| Unstructured pruning | "Scattered zeros" | Removing individual weights; saves storage with sparse formats but not dense compute |
| Structured pruning | "Remove neurons or channels" | Removing whole rows, columns, channels or heads, producing smaller dense layers |

## Further Reading

- [Han et al., Learning both Weights and Connections for Efficient Neural Networks (2015)](https://arxiv.org/abs/1506.02626) - magnitude pruning with retraining at scale
- [Frankle and Carbin, The Lottery Ticket Hypothesis (2018)](https://arxiv.org/abs/1803.03635) - sparse subnetworks that train well from their original initialization
- [Zhu and Gupta, To prune, or not to prune (2017)](https://arxiv.org/abs/1710.01878) - the gradual cubic sparsity schedule
- [PyTorch docs: Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) - `torch.nn.utils.prune`, masks, hooks and custom pruning methods
