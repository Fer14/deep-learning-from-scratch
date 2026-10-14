# Phase 02: Training Deep Nets

> Knowing how to compute a gradient is the easy part. Making a deep network actually learn is a craft.

This phase is about everything between "I have a model and a loss" and "the model trains reliably". You start with the optimizers that replaced plain gradient descent (momentum and Adam) and the learning rate schedules that go with them. Then you look at why deep networks fail before the first step and how initialization and normalization layers keep signals well scaled. Dropout, weight decay and data augmentation fight overfitting, a hand-built data loader shows what `DataLoader` does for you, and a debugging checklist turns silent training bugs into loud ones. The phase ends with mixed precision training and checkpointing, which make long runs cheaper and restartable. Every lesson builds the technique in NumPy first and then checks it against PyTorch.

**Prerequisites:** Phase 00 (Tensors and Autodiff), Phase 01 (Neural Networks)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [SGD, Momentum and Adam](01-sgd-momentum-and-adam/) | Three optimizers with a shared `step` interface, raced on an ill-conditioned bowl and matched against `torch.optim` |
| 02 | [Learning Rate Schedules](02-learning-rate-schedules/) | Step, exponential, cosine and warmup schedules plus an LR range test, matched against `lr_scheduler` |
| 03 | [Weight Initialization](03-weight-initialization/) | Activation and gradient statistics through deep stacks, Xavier and He init, and the symmetry problem |
| 04 | [Batch and Layer Normalization](04-batch-and-layer-normalization/) | Batch norm and layer norm forward and backward with running statistics, gradient checked |
| 05 | [Dropout and Weight Decay](05-dropout-and-weight-decay/) | Inverted dropout, L2 vs decoupled weight decay (AdamW) on an overfitting MLP |
| 06 | [Data Loading and Augmentation](06-data-loading-and-augmentation/) | A Dataset / DataLoader pair and composable image augmentations that buy shift invariance |
| 07 | [Debugging Training](07-debugging-training/) | A gradient checker and a checklist of sanity checks that catch silent training bugs |
| 08 | [Mixed Precision and Checkpointing](08-mixed-precision-and-checkpointing/) | fp16 training with master weights and dynamic loss scaling, and bit-identical checkpoint resume |

Run the first lesson:

```bash
uv run python phases/02-training-deep-nets/01-sgd-momentum-and-adam/code/main.py
```
