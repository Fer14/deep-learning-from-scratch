# Visualizing Features and Receptive Fields

> A trained CNN is not a black box if you know where to look: its filters, its feature maps, and the pixels its gradients point at.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 02 (Training Deep Nets), Deeper CNNs: VGG, Residual Networks
**Time:** ~50 minutes

## Learning Objectives

- Compute the theoretical receptive field and jump of every layer, and measure the effective receptive field with gradients
- Plot first-layer filters and the feature maps they produce for a real input
- Build saliency maps (input gradients) and occlusion maps, and explain what each one does and does not tell you
- Synthesize an input that maximizes a class score by gradient ascent on the pixels

## The Problem

Your CNN reaches 97% on digits. When it is wrong, why? When it is right, is it right for the right reason, or has it latched onto something irrelevant like the thickness of the border? In medical imaging, a famous failure mode is a model that detects the hospital's watermark instead of the disease. Accuracy alone cannot tell you.

Interpretability tools give you partial answers. Some describe the network itself (what does each filter respond to, how much of the image can a unit see?), others explain one prediction (which pixels mattered for this output?). None of them is a full explanation, but together they are the standard debugging kit for vision models, and each one takes only a few lines once you can backpropagate to the input.

## The Concept

### Theoretical receptive field

From the VGG lesson: walk through the layers keeping the field `r` and jump `j`:

```
r <- r + (k - 1) * j
j <- j * s
```

For the small CNN in this lesson (conv3, pool2, conv3, pool2, conv3):

| layer | r | j |
|-------|---|---|
| conv1 3x3 | 3 | 1 |
| pool1 2x2 | 4 | 2 |
| conv2 3x3 | 8 | 2 |
| pool2 2x2 | 10 | 4 |
| conv3 3x3 | 18 | 4 |

Each conv3 unit can be influenced by an 18x18 patch of the input.

### Effective receptive field

"Can be influenced" is not "is influenced equally". A center pixel reaches the output through many more paths than a border pixel of the receptive field. Luo et al. (2016) showed that the gradient of a unit with respect to its input looks roughly Gaussian and occupies only a fraction of the theoretical field. You measure it by backpropagating from one central unit to a random input and averaging `|grad|` over several inputs.

```
theoretical RF: every pixel in the square can matter
effective RF:   most of the gradient mass sits near the center

  . . . . . . .         . . . . . . .
  . . . . . . .         . . . o . . .
  . . . . . . .         . . o O o . .
  . . . . . . .   vs    . o O @ O o .
  . . . . . . .         . . o O o . .
  . . . . . . .         . . . o . . .
  . . . . . . .         . . . . . . .
```

### Filters and feature maps

First-layer kernels live in pixel space, so you can plot them directly: in trained CNNs they look like oriented edge and blob detectors. Deeper kernels have many input channels and are not interpretable by eye; for those you plot the **feature maps** (activations) instead, which show where in the image each channel fires.

### Saliency maps

For an input `x` and class score `s_c(x)`, the saliency map is `|ds_c/dx|`: how much a tiny change of each pixel would change the score. It is one backward pass. It is also noisy and only describes the local linear behavior of the network around `x`.

### Occlusion maps

Slide a gray (or black) square over the image and record how much the probability of the correct class drops at each position. It is slow (one forward pass per position) but easy to trust: it measures what happens when information is actually removed.

### Class visualization

Turn the question around: which input maximizes `s_c`? Start from a blank image and run gradient ascent on the pixels, with a penalty on the pixel norm so the image does not explode:

```
x <- x + lr * d/dx [ s_c(x) - lambda * ||x||^2 ]
```

The result shows what patterns the network associates with a class. Without stronger priors (blurring, jitter) the images are noisy and often unnatural: a reminder that CNNs do not see digits the way you do.

## Build It

### Step 1: The model and its receptive fields

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.head = nn.Linear(32, 10)

    def features(self, images):
        x1 = torch.relu(self.conv1(images))
        x2 = torch.relu(self.conv2(F.max_pool2d(x1, 2)))
        x3 = torch.relu(self.conv3(F.max_pool2d(x2, 2)))
        return x1, x2, x3

    def forward(self, images):
        return self.head(self.features(images)[2].mean(dim=(2, 3)))
```

Returning all three feature maps from `features` is what makes the rest of the lesson possible: you can backpropagate from, or plot, any layer.

### Step 2: Effective receptive field

```python
def effective_receptive_field(model, layer_index, size, trials=16):
    total = torch.zeros(size, size)
    device = next(model.parameters()).device
    for _ in range(trials):
        image = torch.randn(1, 1, size, size, device=device, requires_grad=True)
        activation = model.features(image)[layer_index]
        center = activation.shape[-1] // 2
        activation[0, :, center, center].sum().backward()
        total += image.grad[0, 0].abs().cpu()
    return total / total.max()
```

On a 32x32 input (bigger than the field, so borders do not interfere):

```
Effective RF of conv3 (untrained): nonzero width 18, 30% of gradient mass in its central 8x8 (uniform would be 20%)
Effective RF of conv3 (trained): nonzero width 18, 28% of gradient mass in its central 8x8 (uniform would be 20%)
```

The nonzero region matches the formula exactly (18), and the gradient mass is concentrated in the middle. In this small network training changes the shape only a little; in deep networks, Luo et al. found training widens the effective field.

### Step 3: Saliency and occlusion

```python
def saliency_map(model, image, label):
    image = image.clone().unsqueeze(0).requires_grad_(True)
    model(image)[0, label].backward()
    return image.grad[0, 0].abs()


def occlusion_map(model, image, label, patch=4, stride=2):
    size = image.shape[-1]
    baseline = torch.softmax(model(image.unsqueeze(0)), dim=1)[0, label].item()
    positions = range(0, size - patch + 1, stride)
    heat = torch.zeros(len(positions), len(positions))
    with torch.no_grad():
        for i, row in enumerate(positions):
            for j, col in enumerate(positions):
                occluded = image.clone()
                occluded[:, row:row + patch, col:col + patch] = image.min()
                prob = torch.softmax(model(occluded.unsqueeze(0)), dim=1)[0, label].item()
                heat[i, j] = baseline - prob
    return heat
```

For the first test digit (a 2), covering one 4x4 patch near the bottom stroke is enough to drop the probability of "2" to almost zero. That bottom stroke is what separates a 2 from a 7 or a 1.

### Step 4: Maximize a class

```python
def maximize_class(model, label, size, steps=300, lr=0.05):
    device = next(model.parameters()).device
    image = torch.zeros(1, 1, size, size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=lr)
    for _ in range(steps):
        score = model(image)[0, label] - 0.05 * image.pow(2).sum()
        optimizer.zero_grad()
        (-score).backward()
        optimizer.step()
    return image.detach()[0, 0]
```

The optimizer only touches the image; the model's weights stay fixed. Every synthesized image is classified as its target class (`[0, 1, 2, ..., 9]`), yet most look like structured noise.

### Step 5: Look at the figure

`features.png` has four rows: the eight first-layer filters, their feature maps on the test digit, the input with its saliency, occlusion and effective receptive fields, and the ten class visualizations. Several filters respond to the left or right edges of strokes, others to their tops; the feature maps show the same digit seen through each of those detectors.

## Use It

You rarely write these from scratch in production. Libraries such as Captum (for PyTorch) implement saliency, Integrated Gradients, Grad-CAM, occlusion and more behind one interface. Two things are worth knowing about PyTorch itself:

- **Forward hooks** let you grab intermediate activations from any model without changing its code: `layer.register_forward_hook(lambda module, inputs, output: store.append(output))`. That is how you would get feature maps from a torchvision ResNet.
- **`torch.autograd.grad(outputs, inputs)`** returns input gradients directly without touching `.grad` attributes, which is cleaner for saliency.

Treat every saliency method with skepticism. Adebayo et al. (2018) showed that some popular methods produce almost the same maps for a trained network and a network with randomized weights. Occlusion and simple "sanity checks" like that one are your defense.

## Exercises

1. **Easy.** Plot the conv2 feature maps (16 channels at 8x8) for a 0 and a 1. Which channels respond very differently between the two digits?
2. **Medium.** Implement Grad-CAM: take the gradient of the class score with respect to the conv3 feature maps, average it over space to get one weight per channel, form the weighted sum of the maps, apply ReLU and upsample to 16x16. Compare it with the saliency and occlusion maps.
3. **Hard.** Add jitter (random shifts of 1 pixel each step) and occasional Gaussian blur of the image to `maximize_class`. Do the class visualizations start to look like digits? Then run the Adebayo sanity check: re-initialize the model's weights and recompute the saliency map. How similar is it to the trained one?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Theoretical receptive field | "What a neuron sees" | All input pixels that can affect a unit, computed from kernel sizes and strides |
| Effective receptive field | "What a neuron actually uses" | The region where the input gradient is significant; roughly Gaussian and smaller than the theoretical field |
| Feature map | "Activation map" | The output of one channel of a layer, showing where its pattern occurs in the input |
| Saliency map | "Heatmap of importance" | The absolute gradient of a class score with respect to each input pixel; a local, linear sensitivity measure |
| Occlusion sensitivity | "Blocking out parts" | The drop in class probability when a patch is masked, measured at every position |
| Activation maximization | "What the network dreams" | Gradient ascent on the input to maximize a chosen unit or class score, with the weights frozen |
| Forward hook | "Tapping into a layer" | A callback PyTorch runs on a module's output during the forward pass |

## Further Reading

- [Zeiler and Fergus, Visualizing and Understanding Convolutional Networks (2013)](https://arxiv.org/abs/1311.2901) -- feature visualization and occlusion experiments on AlexNet
- [Simonyan, Vedaldi and Zisserman, Deep Inside Convolutional Networks (2013)](https://arxiv.org/abs/1312.6034) -- saliency maps and class visualization by gradient ascent
- [Luo et al., Understanding the Effective Receptive Field in Deep Convolutional Neural Networks (2016)](https://arxiv.org/abs/1701.04128) -- the Gaussian shape of the effective receptive field
- [Olah, Mordvintsev and Schubert, Feature Visualization (Distill, 2017)](https://distill.pub/2017/feature-visualization/) -- how to make activation maximization produce meaningful images
