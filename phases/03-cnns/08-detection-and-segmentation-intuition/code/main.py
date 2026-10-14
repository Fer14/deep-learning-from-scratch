from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.ops import box_iou, nms

CANVAS = 32
GRID = 4
CELL = CANVAS // GRID
NUM_CLASSES = 10


def make_scenes(digit_images, digit_labels, count, generator, max_objects=3):
    canvases = torch.zeros(count, 1, CANVAS, CANVAS)
    masks = torch.zeros(count, CANVAS, CANVAS, dtype=torch.long)
    scenes = []
    for index in range(count):
        objects = []
        wanted = torch.randint(1, max_objects + 1, (1,), generator=generator).item()
        attempts = 0
        while len(objects) < wanted and attempts < 50:
            attempts += 1
            size = torch.randint(10, 15, (1,), generator=generator).item()
            x1 = torch.randint(0, CANVAS - size + 1, (1,), generator=generator).item()
            y1 = torch.randint(0, CANVAS - size + 1, (1,), generator=generator).item()
            box = torch.tensor([x1, y1, x1 + size, y1 + size], dtype=torch.float32)
            if objects and box_iou(box[None], torch.stack([o[0] for o in objects])).max() > 0:
                continue
            pick = torch.randint(0, len(digit_labels), (1,), generator=generator).item()
            digit = F.interpolate(digit_images[pick][None, None], size=(size, size), mode="bilinear",
                                  align_corners=False)[0, 0]
            canvases[index, 0, y1:y1 + size, x1:x1 + size] = digit
            masks[index, y1:y1 + size, x1:x1 + size][digit > 0.3] = digit_labels[pick] + 1
            objects.append((box, digit_labels[pick].item()))
        scenes.append(objects)
    return canvases, masks, scenes


def encode_grid_targets(scenes):
    targets = torch.zeros(len(scenes), GRID, GRID, 6)
    for index, objects in enumerate(scenes):
        for box, label in objects:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            col, row = int(center_x // CELL), int(center_y // CELL)
            targets[index, row, col] = torch.tensor([
                1.0, center_x / CELL - col, center_y / CELL - row,
                (box[2] - box[0]) / CANVAS, (box[3] - box[1]) / CANVAS, float(label)])
    return targets


class GridDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 5 + NUM_CLASSES, 1)

    def forward(self, images):
        return self.head(self.backbone(images)).permute(0, 2, 3, 1)


def detection_loss(predictions, targets):
    objectness = targets[..., 0]
    has_object = objectness > 0
    object_loss = F.binary_cross_entropy_with_logits(predictions[..., 0], objectness)
    box_loss = F.mse_loss(torch.sigmoid(predictions[..., 1:5][has_object]), targets[..., 1:5][has_object])
    class_loss = F.cross_entropy(predictions[..., 5:][has_object], targets[..., 5][has_object].long())
    return object_loss + 5.0 * box_loss + class_loss


def decode_predictions(predictions, score_threshold=0.5):
    rows = torch.arange(GRID, device=predictions.device).view(GRID, 1).expand(GRID, GRID)
    cols = torch.arange(GRID, device=predictions.device).view(1, GRID).expand(GRID, GRID)
    results = []
    for prediction in predictions:
        box_params = torch.sigmoid(prediction[..., 1:5])
        center_x = (cols + box_params[..., 0]) * CELL
        center_y = (rows + box_params[..., 1]) * CELL
        width = box_params[..., 2] * CANVAS
        height = box_params[..., 3] * CANVAS
        boxes = torch.stack([center_x - width / 2, center_y - height / 2,
                             center_x + width / 2, center_y + height / 2], dim=-1).reshape(-1, 4)
        class_probs = torch.softmax(prediction[..., 5:], dim=-1).reshape(-1, NUM_CLASSES)
        class_prob, label = class_probs.max(dim=-1)
        score = torch.sigmoid(prediction[..., 0]).reshape(-1) * class_prob
        keep = score > score_threshold
        results.append((boxes[keep], score[keep], label[keep]))
    return results


def iou_matrix(boxes_a, boxes_b):
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).prod(dim=-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).prod(dim=-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def non_max_suppression(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    keep = []
    while len(order) > 0:
        best = order[0]
        keep.append(best.item())
        if len(order) == 1:
            break
        overlaps = iou_matrix(boxes[best][None], boxes[order[1:]])[0]
        order = order[1:][overlaps <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def precision_recall(detections, scenes, iou_threshold=0.5):
    true_positives, predicted, actual, matched_ious = 0, 0, 0, []
    for (boxes, scores, labels), objects in zip(detections, scenes):
        actual += len(objects)
        predicted += len(boxes)
        if len(boxes) == 0 or len(objects) == 0:
            continue
        truth_boxes = torch.stack([o[0] for o in objects]).to(boxes.device)
        truth_labels = torch.tensor([o[1] for o in objects], device=boxes.device)
        ious = iou_matrix(boxes, truth_boxes)
        used = set()
        for pred_index in scores.argsort(descending=True).tolist():
            best = ious[pred_index].argmax().item()
            if ious[pred_index, best] >= iou_threshold and best not in used and labels[pred_index] == truth_labels[best]:
                used.add(best)
                true_positives += 1
                matched_ious.append(ious[pred_index, best].item())
    mean_iou = sum(matched_ious) / max(len(matched_ious), 1)
    return true_positives / max(predicted, 1), true_positives / max(actual, 1), mean_iou


class TinyUNet(nn.Module):
    def __init__(self, num_outputs=NUM_CLASSES + 1):
        super().__init__()
        self.encode1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.encode2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decode2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.decode1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.classify = nn.Conv2d(16, num_outputs, 1)

    def forward(self, images):
        skip1 = self.encode1(images)
        skip2 = self.encode2(F.max_pool2d(skip1, 2))
        x = self.bottleneck(F.max_pool2d(skip2, 2))
        x = self.decode2(torch.cat([self.up2(x), skip2], dim=1))
        x = self.decode1(torch.cat([self.up1(x), skip1], dim=1))
        return self.classify(x)


def segmentation_scores(logits, masks):
    predictions = logits.argmax(dim=1)
    pixel_accuracy = (predictions == masks).float().mean().item()
    ious = []
    for label in range(NUM_CLASSES + 1):
        intersection = ((predictions == label) & (masks == label)).sum().item()
        union = ((predictions == label) | (masks == label)).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return pixel_accuracy, sum(ious) / len(ious)


def fit(model, images, targets, loss_fn, epochs, batch_size=64, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(1)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(images), generator=generator).to(images.device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = loss_fn(model(images[batch]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if epoch % 5 == 4:
            print(f"    epoch {epoch + 1:>2}  loss {total / len(order):.4f}")


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    digits = load_digits()
    digit_images = torch.tensor(digits.images / 16.0, dtype=torch.float32)
    digit_labels = torch.tensor(digits.target)
    generator = torch.Generator().manual_seed(0)
    train_images, train_masks, train_scenes = make_scenes(digit_images[:1400], digit_labels[:1400], 2500, generator)
    test_images, test_masks, test_scenes = make_scenes(digit_images[1400:], digit_labels[1400:], 500, generator)
    objects = sum(len(s) for s in train_scenes)
    print(f"Synthetic scenes: {len(train_scenes)} train / {len(test_scenes)} test, {CANVAS}x{CANVAS}, "
          f"{objects} training objects, digits 10-14 px")

    train_images, test_images = train_images.to(device), test_images.to(device)
    train_targets = encode_grid_targets(train_scenes).to(device)

    print(f"\nTraining a {GRID}x{GRID} grid detector (one prediction per {CELL}x{CELL} cell):")
    torch.manual_seed(0)
    detector = GridDetector().to(device)
    fit(detector, train_images, train_targets, detection_loss, epochs=20)
    detector.eval()
    with torch.no_grad():
        raw_predictions = detector(test_images)
    detections = decode_predictions(raw_predictions)
    precision, recall, mean_iou = precision_recall(detections, test_scenes)
    print(f"Test precision {precision:.3f}, recall {recall:.3f} at IoU >= 0.5, mean IoU of hits {mean_iou:.3f}")

    low_threshold = decode_predictions(raw_predictions, score_threshold=0.05)
    before = sum(len(b) for b, _, _ in low_threshold)
    suppressed = []
    for boxes, scores, labels in low_threshold:
        keep = non_max_suppression(boxes, scores, 0.3)
        library_keep = nms(boxes, scores, 0.3)
        assert torch.equal(keep.to(library_keep.device), library_keep)
        suppressed.append((boxes[keep], scores[keep], labels[keep]))
    after = sum(len(b) for b, _, _ in suppressed)
    p_low, r_low, _ = precision_recall(low_threshold, test_scenes)
    p_nms, r_nms, _ = precision_recall(suppressed, test_scenes)
    print(f"Score threshold 0.05: {before} boxes, precision {p_low:.3f}, recall {r_low:.3f}")
    print(f"  after your NMS (IoU 0.3): {after} boxes, precision {p_nms:.3f}, recall {r_nms:.3f} "
          f"(matches torchvision.ops.nms on all {len(test_scenes)} images)")

    sample_boxes = torch.stack([o[0] for o in test_scenes[0]])
    print(f"Your IoU vs torchvision.ops.box_iou max diff: "
          f"{(iou_matrix(sample_boxes, sample_boxes + 2) - box_iou(sample_boxes, sample_boxes + 2)).abs().max():.1e}")

    print("\nTraining a tiny U-Net for semantic segmentation (11 classes: background + 10 digits):")
    torch.manual_seed(0)
    segmenter = TinyUNet().to(device)
    fit(segmenter, train_images, train_masks.to(device), F.cross_entropy, epochs=10)
    segmenter.eval()
    with torch.no_grad():
        segmentation_logits = segmenter(test_images)
    pixel_accuracy, mean_iou = segmentation_scores(segmentation_logits, test_masks.to(device))
    background = (test_masks == 0).float().mean().item()
    print(f"Test pixel accuracy {pixel_accuracy:.3f} (all-background baseline {background:.3f}), mIoU {mean_iou:.3f}")

    library_detector = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=NUM_CLASSES + 1)
    library_params = sum(p.numel() for p in library_detector.parameters())
    print(f"\ntorchvision fasterrcnn_mobilenet_v3_large_fpn(weights=None): {library_params:,} params "
          f"vs your grid detector {sum(p.numel() for p in detector.parameters()):,}")
    library_detector.eval()
    with torch.no_grad():
        outputs = library_detector([test_images[0].cpu().repeat(3, 1, 1)])
    print(f"  untrained forward pass returns keys {sorted(outputs[0].keys())}")

    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for column in range(4):
        axes[0, column].imshow(test_images[column, 0].cpu(), cmap="gray")
        boxes, scores, labels = detections[column]
        for box, score, label in zip(boxes.cpu(), scores.cpu(), labels.cpu()):
            axes[0, column].add_patch(patches.Rectangle((box[0] - 0.5, box[1] - 0.5), box[2] - box[0], box[3] - box[1],
                                                        fill=False, color="lime", linewidth=1.5))
            axes[0, column].text(box[0], box[1] - 1, f"{label.item()} {score:.2f}", color="lime", fontsize=8)
        axes[1, column].imshow(segmentation_logits[column].argmax(dim=0).cpu(), cmap="tab20", vmin=0, vmax=19)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_title("detections")
    axes[1, 0].set_title("segmentation")
    figure.tight_layout()
    output_path = Path(__file__).parent / "detection_segmentation.png"
    figure.savefig(output_path, dpi=90)
    print(f"\nSaved {output_path.name}")


if __name__ == "__main__":
    main()
