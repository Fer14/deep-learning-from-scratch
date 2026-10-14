import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def load_data():
    digits = load_digits()
    x = torch.tensor(digits.data, dtype=torch.float32) / 16.0
    y = torch.tensor(digits.target)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    x, y = x[order], y[order]
    return (x[:1200], y[:1200]), (x[1200:], y[1200:])


def teacher_net():
    return nn.Sequential(nn.Linear(64, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 10))


def student_net(hidden=32):
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def augment(x, generator):
    return (x + 0.1 * torch.randn(x.shape, generator=generator)).clamp(0, 1)


def softmax_with_temperature(logits, temperature):
    scaled = logits / temperature
    scaled = scaled - scaled.max(dim=1, keepdim=True).values
    exp = scaled.exp()
    return exp / exp.sum(dim=1, keepdim=True)


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    p_teacher = softmax_with_temperature(teacher_logits, temperature)
    log_p_student = torch.log_softmax(student_logits / temperature, dim=1)
    soft = (p_teacher * (torch.log(p_teacher + 1e-12) - log_p_student)).sum(1).mean() * temperature**2
    if labels is None:
        return soft
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard


def train(model, x, y, epochs, generator, teacher=None, temperature=4.0, alpha=0.9, lr=3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        model.train()
        for i in range(0, len(x), 64):
            idx = order[i:i + 64]
            batch = augment(x[idx], generator)
            logits = model(batch)
            if teacher is None:
                loss = F.cross_entropy(logits, y[idx])
            else:
                with torch.no_grad():
                    teacher_logits = teacher(batch)
                labels = None if y is None else y[idx]
                loss = distillation_loss(logits, teacher_logits, labels, temperature, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def accuracy(model, data, digit=None):
    x, y = data
    if digit is not None:
        x, y = x[y == digit], y[y == digit]
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(0)
    train_data, test_data = load_data()
    x_train, y_train = train_data

    teacher = train(teacher_net(), x_train, y_train, 60, generator, lr=1e-3)
    print(f"Teacher: {n_params(teacher):6d} parameters, test accuracy {accuracy(teacher, test_data):.4f}")
    print(f"Student: {n_params(student_net()):6d} parameters ({n_params(teacher) / n_params(student_net()):.0f}x smaller)\n")

    print("1) Dark knowledge: the teacher's probabilities for one test image of a 3")
    image = test_data[0][test_data[1] == 3][:1]
    with torch.no_grad():
        logits = teacher(image)
    for temperature in [1, 4, 10]:
        probs = softmax_with_temperature(logits, temperature)[0]
        ranked = probs.argsort(descending=True)[:4]
        print(f"  T={temperature:2d}: " + "  ".join(f"'{d}': {probs[d]:.3f}" for d in ranked.tolist()))

    print("\n2) Labels are scarce: 100 labeled images, but 1200 unlabeled ones the teacher can annotate")
    n_labeled = 100
    with torch.no_grad():
        pseudo_labels = teacher(x_train).argmax(1)
    variants = [
        ("hard labels, 100 images", lambda g: train(student_net(), x_train[:n_labeled], y_train[:n_labeled], 240, g)),
        ("teacher argmax, 1200 images", lambda g: train(student_net(), x_train, pseudo_labels, 60, g)),
        ("soft targets T=4, 1200 images", lambda g: train(student_net(), x_train, None, 60, g, teacher=teacher, temperature=4.0)),
    ]
    for name, make in variants:
        values = []
        for seed in range(3):
            torch.manual_seed(100 + seed)
            values.append(accuracy(make(torch.Generator().manual_seed(seed)), test_data))
        print(f"  {name:30s}: test accuracy {sum(values) / 3:.4f}  (3 seeds: {', '.join(f'{v:.3f}' for v in values)})")

    print("\n3) Hinton's experiment: the transfer set contains no 3s at all")
    keep = y_train != 3
    torch.manual_seed(7)
    hard_student = train(student_net(), x_train[keep], y_train[keep], 100, torch.Generator().manual_seed(7))
    torch.manual_seed(7)
    soft_student = train(student_net(), x_train[keep], None, 100, torch.Generator().manual_seed(7), teacher=teacher, temperature=8.0)
    print(f"  {'student':34s} {'overall':>8s} {'on 3s':>7s}")
    for name, model in [("hard labels, no 3s", hard_student), ("soft targets T=8, no 3s", soft_student)]:
        print(f"  {name:34s} {accuracy(model, test_data):8.4f} {accuracy(model, test_data, digit=3):7.4f}")
    with torch.no_grad():
        soft_student[-1].bias[3] += 3.0
    print(f"  {'soft targets, bias of 3 raised by 3':34s} {accuracy(soft_student, test_data):8.4f} {accuracy(soft_student, test_data, digit=3):7.4f}")
    with torch.no_grad():
        soft_student[-1].bias[3] -= 3.0

    print("\n4) Use It: F.kl_div gives the same soft loss")
    x_batch = x_train[:64]
    with torch.no_grad():
        s_logits, t_logits = soft_student(x_batch), teacher(x_batch)
    temperature = 4.0
    hand = distillation_loss(s_logits, t_logits, None, temperature, alpha=1.0)
    library = F.kl_div(F.log_softmax(s_logits / temperature, 1), F.log_softmax(t_logits / temperature, 1),
                       reduction="batchmean", log_target=True) * temperature**2
    print(f"  hand-written {hand.item():.6f} | F.kl_div {library.item():.6f}")


if __name__ == "__main__":
    main()
