# Phase 03: CNNs

> Look locally, share the weights, shrink the image, deepen the features.

This phase builds convolutional networks from the operation up. You write convolution as nested loops and then as one matrix multiply, add padding, stride and pooling, and train LeNet-5 with a convolution layer you wrote yourself. From there you follow the history of deeper networks: VGG's stacks of 3x3 kernels and why depth alone stops working, then the residual connection that fixed it. You open up a trained CNN with receptive fields, saliency, occlusion and class visualizations, reuse a pretrained backbone on a new task with few labels, and finish by turning a classifier into a grid-based object detector and a U-Net segmenter. Every lesson uses the 8x8 scikit-learn digits (upsampled or composed into scenes when needed), builds the idea with plain tensor ops first, and then compares it with `torch.nn` and `torchvision`.

**Prerequisites:** Phase 00 (Tensors and Autodiff), Phase 01 (Neural Networks), Phase 02 (Training Deep Nets)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Convolution from Scratch](01-convolution-from-scratch/) | Loop, multi-channel and im2col convolutions that match `F.conv2d`, with edge filters on digits |
| 02 | [Padding, Stride and Pooling](02-padding-stride-and-pooling/) | The output-size formula, padded and strided convolution, max/avg/global pooling, and gradient routing |
| 03 | [LeNet](03-lenet/) | LeNet-5 with a hand-written convolution layer, trained on digits and compared with an MLP on shifted inputs |
| 04 | [Deeper CNNs: VGG](04-deeper-cnns-vgg/) | A configuration-driven VGG builder, receptive field arithmetic, and an exact rebuild of VGG-11 |
| 05 | [Residual Networks](05-residual-networks/) | Plain vs residual networks up to 30 layers, gradient norms per layer, and the ResNet basic block |
| 06 | [Visualizing Features and Receptive Fields](06-visualizing-features-and-receptive-fields/) | Filters, feature maps, effective receptive fields, saliency, occlusion and class visualization |
| 07 | [Transfer Learning](07-transfer-learning/) | Pretraining on digits 0-4 and transferring to 5-9 with 1 to 30 labels per class: probe vs fine-tune vs scratch |
| 08 | [Detection and Segmentation Intuition](08-detection-and-segmentation-intuition/) | A YOLO-style grid detector with hand-written IoU and NMS, and a tiny U-Net on synthetic digit scenes |

Run the first lesson:

```bash
uv run python phases/03-cnns/01-convolution-from-scratch/code/main.py
```
