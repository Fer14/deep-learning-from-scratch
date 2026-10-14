# Learning Rate Schedules

> Take big steps while you are far away, small steps when you are close. The schedule is how you encode that.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Neural Networks: An MLP from Scratch in NumPy), Phase 02 lesson SGD, Momentum and Adam
**Time:** ~60 minutes

## Learning Objectives

- Explain why a constant learning rate leaves SGD bouncing at a noise floor and how decaying it lets the model settle
- Implement step decay, exponential decay, cosine annealing and linear warmup as plain functions of the step number
- Run a learning rate range test to pick a base learning rate in a few hundred steps
- Reproduce your schedules exactly with `torch.optim.lr_scheduler`

## The Problem

You found in the last lesson that the learning rate is capped by the steepest direction of the loss. In practice there is a second force: minibatch noise. Every step uses a noisy gradient, so even at the bottom of the valley SGD keeps getting kicked around. The size of those kicks is proportional to the learning rate. A large learning rate makes fast progress early and then rattles around the minimum forever. A small learning rate settles nicely but takes ages to get there.

You do not have to choose. The learning rate can change during training: large at the start to cover ground, small at the end to settle. Almost every published training recipe, from ResNets to GPT, specifies a schedule, and getting it wrong can cost you more than picking the wrong architecture. The good news is that a schedule is just a function from step number to a number, and a handful of shapes cover nearly everything.

## The Concept

### Noise floor

Near a minimum, an SGD step is roughly `w <- w - lr * (true gradient + noise)`. The true gradient goes to zero, the noise does not. The parameters end up jittering in a region whose size grows with `lr`. Decaying `lr` shrinks that region:

```
loss
 |\
 | \       constant lr: plateaus at a noisy floor
 |  \____/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
 |       \
 |        \___ decayed lr: keeps going down
 |            \______________________
 +--------------------------------------- step
```

### The common shapes

With `T` total steps and base rate `lr0`:

| Schedule | Formula | Used by |
|----------|---------|---------|
| Constant | `lr0` | quick experiments |
| Step decay | `lr0 * gamma^(floor(t / k))` | classic ResNet recipes (divide by 10 at epochs 30, 60, 90) |
| Exponential | `lr0 * gamma^t` | older RNN and RL code |
| Cosine | `lr_min + 0.5 (lr0 - lr_min)(1 + cos(pi t / T))` | most modern vision and language training |
| Warmup + cosine | linear ramp for `W` steps, then cosine | transformers, large-batch training |

```
lr
 |      step           cosine            warmup + cosine
 |----+               --.                  .--.
 |    |                  `.              /     `.
 |    +----+               `.           /        `.
 |         +---           ___`--      _/            `.___
 +-----------------  ----------------  --------------------- t
```

Cosine is popular because it has no hyperparameters beyond `T` and spends a long time at moderately high learning rates before a smooth landing.

### Warmup

At step 0 the network is random, gradients can be very large, and adaptive optimizers like Adam have only seen one gradient, so their variance estimate is unreliable. A large learning rate in those first steps can throw the weights into a bad region they never recover from. Linear warmup ramps the learning rate from near zero to `lr0` over the first few hundred or thousand steps:

```
lr_t = lr0 * (t + 1) / W     for t < W
```

Warmup matters most for transformers, Adam and very large batches. On the small MLP in this lesson it barely changes anything, which is a useful reminder that recipes are tuned for the models they come from.

### Picking the base learning rate: the range test

Before choosing a schedule you need `lr0`. The learning rate range test (Leslie Smith) trains for a couple of hundred steps while increasing the learning rate exponentially from tiny to huge, and records a smoothed loss:

```
smoothed loss
 |----.
 |     `.
 |       `.          /
 |         `.___    /   <- blows up
 |              `--'
 +------------------------------- lr (log scale)
 1e-4        1e-2   ^best   1
```

The loss is flat while `lr` is too small, falls once it becomes useful, and explodes past the stability limit. A good `lr0` is a bit below the minimum, often a third to a tenth of it.

## Build It

### Step 1: Schedules are functions

Every schedule has the same signature, so the training loop does not care which one you pass.

```python
def step_decay(step, total_steps, base_lr, drop_every=300, gamma=0.1):
    return base_lr * gamma ** (step // drop_every)


def cosine(step, total_steps, base_lr, min_lr=0.0):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine(step, total_steps, base_lr, warmup_steps=50):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
```

### Step 2: The range test

```python
lrs = min_lr * (max_lr / min_lr) ** (np.arange(steps) / (steps - 1))
for lr in lrs:
    batch = rng.choice(len(y), size=batch_size, replace=False)
    loss, grads = loss_and_grads(params, X[batch], y[batch])
    if not np.isfinite(loss) or loss > 10:
        break
    smoothed = loss if smoothed is None else 0.9 * smoothed + 0.1 * loss
    ...
```

On the 8x8 digits dataset with a 64-128-128-10 MLP and SGD with momentum 0.9, the smoothed loss is lowest around `lr = 0.28` and the run blows up near 0.66. That suggests a base learning rate somewhere around 0.05 to 0.15.

### Step 3: Plug a schedule into the training loop

The only change to the loop from the previous lesson is that `lr` is computed every step:

```python
for start in range(0, len(y), batch_size):
    batch = order[start:start + batch_size]
    loss, grads = loss_and_grads(params, X[batch], y[batch])
    lr = schedule(step, total_steps, base_lr)
    for name in params:
        velocity[name] = beta * velocity[name] + grads[name]
        params[name] -= lr * velocity[name]
    step += 1
```

### Step 4: Compare

Twenty epochs at a deliberately aggressive base rate of 0.15:

```
constant         | train loss 0.0360 | test loss 0.1032 | test acc 0.980
step decay       | train loss 0.0038 | test loss 0.0824 | test acc 0.980
exponential      | train loss 0.0020 | test loss 0.0599 | test acc 0.977
cosine           | train loss 0.0013 | test loss 0.0417 | test acc 0.987
warmup + cosine  | train loss 0.0016 | test loss 0.0496 | test acc 0.982
```

Every decaying schedule ends with a training loss 10-30x lower than the constant one, and cosine has the best test loss. The script also sweeps the base rate for 10 epochs:

```
base lr 0.05 | constant train loss 0.0085 | cosine train loss 0.0166
base lr 0.15 | constant train loss 0.0254 | cosine train loss 0.0045
base lr 0.30 | constant train loss 1.5791 | cosine train loss 0.0414
```

At a small rate, decay just slows you down. Near the edge of stability, decay is the difference between a model that trains and one that does not. Open `schedules.png` to see the learning rate curves next to the smoothed losses.

## Use It

PyTorch schedulers wrap an optimizer and update `optimizer.param_groups[i]["lr"]` each time you call `scheduler.step()`:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
for batch in loader:
    loss = criterion(model(batch.x), batch.y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

The script records the learning rate PyTorch uses at each of 900 steps and compares it with your functions:

```
StepLR                | max |ours - torch| over 900 steps = 2.2e-19
ExponentialLR         | max |ours - torch| over 900 steps = 1.1e-16
CosineAnnealingLR     | max |ours - torch| over 900 steps = 2.2e-16
LambdaLR(warmup+cos)  | max |ours - torch| over 900 steps = 2.8e-17
```

Things to remember:

- Call `scheduler.step()` after `optimizer.step()`. Decide whether you step it per batch or per epoch and size `T_max` or `step_size` accordingly.
- `LambdaLR` multiplies the base rate by any function you write. That is how most libraries implement warmup + cosine.
- `OneCycleLR` implements Smith's one-cycle policy (warm up to a high rate, then anneal) and also cycles momentum.
- A scheduler has state; save `scheduler.state_dict()` in checkpoints along with the optimizer.

## Exercises

1. **Easy.** Add a `min_lr=0.01 * base_lr` floor to the cosine schedule and check whether test loss changes.
2. **Medium.** Implement a "linear warmup, constant, linear decay to zero" (trapezoidal or WSD) schedule and compare it with cosine at base rate 0.15. Then reproduce it with `LambdaLR`.
3. **Hard.** Swap the optimizer for the `Adam` class from the previous lesson with `lr=3e-3` and a deeper 6-layer MLP. Measure how often training diverges over 5 seeds with and without 100 warmup steps.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Learning rate schedule | "LR decay" | A function `lr(t)` that sets the step size at every iteration |
| Noise floor | "The loss stops going down" | The region SGD keeps bouncing in because of minibatch noise; its size scales with the learning rate |
| Step decay | "Divide by 10 at milestones" | Multiply the rate by `gamma` every `k` steps or at fixed epochs |
| Cosine annealing | "Cosine schedule" | Half a cosine from `lr0` down to `lr_min` over the run |
| Warmup | "Start slow" | Ramp the rate up linearly over the first `W` steps to avoid early instability |
| LR range test | "LR finder" | Short run with exponentially increasing rate to locate the stability limit |

## Further Reading

- [Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts (2016)](https://arxiv.org/abs/1608.03983) - introduces cosine annealing
- [Smith, Cyclical Learning Rates for Training Neural Networks (2015)](https://arxiv.org/abs/1506.01186) - the learning rate range test and cyclical schedules
- [Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour (2017)](https://arxiv.org/abs/1706.02677) - the linear scaling rule and why warmup is needed
- [PyTorch docs: How to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) - every built-in scheduler and how to chain them
