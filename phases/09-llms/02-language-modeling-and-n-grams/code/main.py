import math
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

CORPUS = """the old harbor town woke up before the sun. the baker lit the oven and the smell of warm bread drifted down to the water. the fishermen pulled their boats into the cold sea and the gulls followed them, crying for scraps.
on the hill above the town stood a lighthouse. the keeper was a quiet woman who wrote everything in a small blue notebook. she wrote the wind, the waves, the color of the sky and the names of the boats that came home.
every morning the keeper climbed the stairs of the lighthouse. there were one hundred and twelve stairs, and she counted them every time. at the top she cleaned the great lamp and looked out at the sea.
the baker had a son who wanted to be a sailor. he watched the boats leave the harbor and he dreamed of islands far away. his father wanted him to bake bread, but the boy wanted to read the stars.
one evening a storm came from the north. the wind pushed the waves over the wall of the harbor and the rain fell like stones. the keeper lit the great lamp and the light turned slowly over the dark water.
a small boat was still out at sea. the fishermen on the dock could not see it, but the keeper saw a tiny light between the waves. she wrote in her notebook: a boat in the storm, north of the rocks, one light.
the baker's son ran up the hill to the lighthouse. he climbed the one hundred and twelve stairs and stood next to the keeper. together they watched the tiny light move closer to the rocks.
the keeper turned the great lamp toward the boat. the light showed the safe path between the rocks, and the boat followed it slowly into the harbor. the fishermen cheered and the gulls flew in circles above the dock.
the next morning the sea was calm and the sky was clear. the baker made a large loaf of bread for the keeper and his son carried it up the hill. the keeper smiled and wrote in her notebook: calm sea, clear sky, warm bread.
after that day the boy came to the lighthouse every evening. the keeper taught him the names of the stars and the shapes of the clouds. she taught him how to read the wind and how to clean the great lamp.
the boy still baked bread in the morning, but in the evening he climbed the stairs and watched the sea. he wrote his own notes in a small green notebook: the wind from the west, the waves are low, three boats came home.
years later the keeper grew old and her hands began to shake. the boy, now a young man, became the new keeper of the lighthouse. he kept the blue notebook next to his green one and read it on quiet nights.
the town changed slowly. new houses were built on the hill and new boats came to the harbor. but the lighthouse stayed the same, and every night the light turned over the dark water.
the new keeper had a daughter who loved numbers. she counted the stairs, the boats, the gulls and the waves. she wrote the numbers in a small red notebook and drew lines between them.
one winter the daughter noticed that the storms came more often when the wind turned from the north. she showed her father the lines in her notebook, and he showed her the old blue notebook. the same pattern was there, written many years before.
the keeper and his daughter wrote a letter to the fishermen. when the wind turns north, they wrote, stay in the harbor and wait for the light. the fishermen read the letter and pinned it on the wall of the bakery.
the bakery still smells of warm bread in the morning. the gulls still cry above the dock. and on the hill the lighthouse still stands, with three small notebooks on the table at the top of the stairs.
"""


def split_text(text, train_fraction):
    cut = int(len(text) * train_fraction)
    return text[:cut], text[cut:]


class NGramModel:
    def __init__(self, order, vocab, smoothing):
        self.order = order
        self.vocab = vocab
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)

    def fit(self, text):
        padded = "\n" * (self.order - 1) + text
        for i in range(self.order - 1, len(padded)):
            context = padded[i - self.order + 1 : i]
            self.counts[context][padded[i]] += 1
        return self

    def prob(self, context, char):
        context = context[-(self.order - 1) :] if self.order > 1 else ""
        table = self.counts.get(context, Counter())
        total = sum(table.values())
        return (table[char] + self.smoothing) / (total + self.smoothing * len(self.vocab))

    def distribution(self, context):
        return np.array([self.prob(context, c) for c in self.vocab])


def perplexity(model, text, history):
    padded = history + text
    offset = len(history)
    total_log_prob = 0.0
    for i in range(offset, len(padded)):
        total_log_prob += math.log(model.prob(padded[:i], padded[i]))
    return math.exp(-total_log_prob / len(text))


class InterpolatedModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def prob(self, context, char):
        return sum(w * m.prob(context, char) for m, w in zip(self.models, self.weights))

    def distribution(self, context):
        return sum(w * m.distribution(context) for m, w in zip(self.models, self.weights))


def sample(model, vocab, prompt, length, rng):
    text = prompt
    for _ in range(length):
        probs = model.distribution(text)
        probs = probs / probs.sum()
        text += vocab[rng.choice(len(vocab), p=probs)]
    return text


def count_unseen_contexts(model, text):
    history = "\n" * (model.order - 1)
    padded = history + text
    unseen = 0
    for i in range(len(history), len(padded)):
        context = padded[i - model.order + 1 : i] if model.order > 1 else ""
        if context not in model.counts:
            unseen += 1
    return unseen / len(text)


def train_neural_bigram(train_ids, vocab_size, steps, learning_rate):
    torch.manual_seed(0)
    logits_table = torch.zeros(vocab_size, vocab_size, requires_grad=True)
    inputs = torch.tensor(train_ids[:-1])
    targets = torch.tensor(train_ids[1:])
    optimizer = torch.optim.Adam([logits_table], lr=learning_rate)
    for _ in range(steps):
        loss = F.cross_entropy(logits_table[inputs], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return logits_table.detach(), loss.item()


def main():
    rng = np.random.default_rng(0)
    vocab = sorted(set(CORPUS) | {"\n"})
    train_text, val_text = split_text(CORPUS, 0.9)
    print(f"vocab {len(vocab)} characters | train {len(train_text)} | val {len(val_text)}")
    print(f"uniform model perplexity = vocab size = {len(vocab)}")

    print("\nstep 1: count-based n-gram models (add-k smoothing, k=0.1)")
    models = {}
    for order in range(1, 7):
        model = NGramModel(order, vocab, smoothing=0.1).fit(train_text)
        models[order] = model
        train_ppl = perplexity(model, train_text, "\n" * (order - 1))
        val_ppl = perplexity(model, val_text, train_text[-(order - 1) :] if order > 1 else "")
        unseen = count_unseen_contexts(model, val_text)
        print(f"  n={order} | contexts {len(model.counts):5d} | train ppl {train_ppl:6.2f} | val ppl {val_ppl:6.2f} | val contexts never seen {unseen:5.1%}")

    print("\nstep 2: what the bigram model learned after 't'")
    after_t = models[2].distribution("t")
    top = np.argsort(after_t)[::-1][:5]
    print("  " + "  ".join(f"{vocab[i]!r}:{after_t[i]:.2f}" for i in top))

    print("\nstep 3: smoothing strength for the 4-gram")
    for k in [1.0, 0.1, 0.01, 0.001]:
        model = NGramModel(4, vocab, smoothing=k).fit(train_text)
        val_ppl = perplexity(model, val_text, train_text[-3:])
        print(f"  k={k:<6} val ppl {val_ppl:6.2f}")

    print("\nstep 4: interpolate orders 1-5")
    mixed = InterpolatedModel([models[o] for o in range(1, 6)], [0.05, 0.1, 0.2, 0.3, 0.35])
    print(f"  interpolated val ppl {perplexity(mixed, val_text, train_text[-4:]):6.2f}")

    print("\nstep 5: sampling")
    for order in [2, 4, 6]:
        text = sample(models[order], vocab, "the keeper", 100, rng)
        print(f"  n={order} k=0.1:   {text!r}")
    sharp_models = [NGramModel(order, vocab, smoothing=0.001).fit(train_text) for order in range(1, 6)]
    print(f"  n=5 k=0.001: {sample(sharp_models[4], vocab, 'the keeper', 100, rng)!r}")
    sharp_mixed = InterpolatedModel(sharp_models, [0.01, 0.04, 0.15, 0.3, 0.5])
    print(f"  interpolated: {sample(sharp_mixed, vocab, 'the keeper', 100, rng)!r}")

    print("\nuse it: a neural bigram in PyTorch")
    char_to_id = {c: i for i, c in enumerate(vocab)}
    train_ids = [char_to_id[c] for c in train_text]
    val_ids = [char_to_id[c] for c in val_text]
    logits_table, train_loss = train_neural_bigram(train_ids, len(vocab), steps=500, learning_rate=0.1)
    val_loss = F.cross_entropy(logits_table[torch.tensor(val_ids[:-1])], torch.tensor(val_ids[1:])).item()
    count_ppl = perplexity(models[2], val_text[1:], val_text[0])
    print(f"  neural bigram: train loss {train_loss:.3f} | val ppl {math.exp(val_loss):.2f}")
    print(f"  count bigram:  val ppl {count_ppl:.2f}")
    t_row = F.softmax(logits_table[char_to_id["t"]], dim=0).numpy()
    print(f"  P(h|t): neural {t_row[char_to_id['h']]:.3f} vs counts {models[2].prob('t', 'h'):.3f}")


if __name__ == "__main__":
    main()
