import random

import torch
import torch.nn as nn
import torch.nn.functional as F

ROYALS = {"king": "male", "queen": "female", "prince": "male", "princess": "female"}
COMMONERS = {"man": "male", "woman": "female", "boy": "male", "girl": "female"}
ANIMALS = ["cat", "dog", "horse", "cow", "sheep"]
FOODS = ["bread", "apple", "rice", "cheese", "soup"]
PLACES_ROYAL = ["palace", "castle", "throne"]
PLACES_COMMON = ["village", "farm", "market"]
POSSESSIVE = {"male": "his", "female": "her"}
PRONOUN = {"male": "he", "female": "she"}


def make_corpus(num_sentences, seed):
    rng = random.Random(seed)
    people = {**ROYALS, **COMMONERS}
    sentences = []
    for _ in range(num_sentences):
        kind = rng.randrange(5)
        person = rng.choice(list(people))
        gender = people[person]
        places = PLACES_ROYAL if person in ROYALS else PLACES_COMMON
        if kind == 0:
            sentence = f"the {person} lives in the {rng.choice(places)}"
        elif kind == 1:
            sentence = f"{PRONOUN[gender]} is the {person} and {PRONOUN[gender]} rides {POSSESSIVE[gender]} {rng.choice(ANIMALS)}"
        elif kind == 2:
            sentence = f"the {person} eats {rng.choice(FOODS)} at the {rng.choice(places)}"
        elif kind == 3:
            sentence = f"the {rng.choice(ANIMALS)} sleeps near the {rng.choice(PLACES_ROYAL + PLACES_COMMON)}"
        else:
            sentence = f"the {rng.choice(ANIMALS)} eats {rng.choice(FOODS)} with {POSSESSIVE[gender]} {person}"
        sentences.append(sentence.split())
    return sentences


def build_vocabulary(sentences):
    counts = {}
    for sentence in sentences:
        for word in sentence:
            counts[word] = counts.get(word, 0) + 1
    words = sorted(counts, key=lambda w: (-counts[w], w))
    return words, {word: index for index, word in enumerate(words)}, torch.tensor([counts[w] for w in words], dtype=torch.float32)


def skipgram_pairs(sentences, word_to_index, window):
    centers, contexts = [], []
    for sentence in sentences:
        ids = [word_to_index[w] for w in sentence]
        for position, center in enumerate(ids):
            for offset in range(-window, window + 1):
                neighbor = position + offset
                if offset != 0 and 0 <= neighbor < len(ids):
                    centers.append(center)
                    contexts.append(ids[neighbor])
    return torch.tensor(centers), torch.tensor(contexts)


class SkipGramNegativeSampling:
    def __init__(self, vocab_size, dim, generator):
        self.center_vectors = (torch.randn(vocab_size, dim, generator=generator) * 0.1).requires_grad_(True)
        self.context_vectors = torch.zeros(vocab_size, dim, requires_grad=True)

    def parameters(self):
        return [self.center_vectors, self.context_vectors]

    def loss(self, centers, contexts, negatives):
        center = self.center_vectors[centers]
        positive = self.context_vectors[contexts]
        negative = self.context_vectors[negatives]
        positive_score = (center * positive).sum(dim=-1)
        negative_score = torch.einsum("bd,bkd->bk", center, negative)
        return -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_score).sum(dim=-1)).mean()


def train_skipgram(model, centers, contexts, noise_distribution, epochs, batch_size, negatives_per_pair, generator):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        order = torch.randperm(len(centers), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            negatives = torch.multinomial(noise_distribution, len(batch) * negatives_per_pair, replacement=True,
                                          generator=generator).view(len(batch), negatives_per_pair)
            loss = model.loss(centers[batch], contexts[batch], negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if epoch % 5 == 4:
            print(f"  epoch {epoch + 1:>2}  loss {total / len(order):.4f}")


def nearest(embeddings, words, word_to_index, query_vector, exclude, k=3):
    normalized = F.normalize(embeddings, dim=1)
    similarity = normalized @ F.normalize(query_vector, dim=0)
    for word in exclude:
        similarity[word_to_index[word]] = -1.0
    best = similarity.topk(k)
    return [(words[i], s) for s, i in zip(best.values.tolist(), best.indices.tolist())]


def main():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    sentences = make_corpus(4000, seed=0)
    words, word_to_index, counts = build_vocabulary(sentences)
    print(f"Corpus: {len(sentences)} sentences, {int(counts.sum())} tokens, vocabulary of {len(words)} words")
    print("Examples:")
    for sentence in sentences[:4]:
        print("  " + " ".join(sentence))

    one_hot = F.one_hot(torch.tensor([word_to_index["king"], word_to_index["queen"]]), len(words)).float()
    print(f"\nOne-hot vectors: dimension {one_hot.shape[1]}, cosine(king, queen) = "
          f"{F.cosine_similarity(one_hot[0], one_hot[1], dim=0):.1f} (every pair is orthogonal)")

    centers, contexts = skipgram_pairs(sentences, word_to_index, window=4)
    noise_distribution = counts ** 0.75
    noise_distribution = noise_distribution / noise_distribution.sum()
    print(f"Skip-gram pairs with window 4: {len(centers)}")

    model = SkipGramNegativeSampling(len(words), 16, generator)
    print("\nTraining skip-gram with negative sampling (16-dim, 5 negatives per pair):")
    train_skipgram(model, centers, contexts, noise_distribution, epochs=15, batch_size=512, negatives_per_pair=5,
                   generator=generator)
    embeddings = model.center_vectors.detach()

    print("\nNearest neighbors by cosine similarity:")
    for word in ["king", "girl", "cat", "bread", "palace", "his"]:
        neighbors = nearest(embeddings, words, word_to_index, embeddings[word_to_index[word]], [word])
        print(f"  {word:<8} -> " + ", ".join(f"{w} ({s:.2f})" for w, s in neighbors))

    groups = {"royal people": list(ROYALS), "common people": list(COMMONERS), "animals": ANIMALS, "foods": FOODS}
    normalized = F.normalize(embeddings, dim=1)
    print("\nAverage cosine similarity within and across groups:")
    names = list(groups)
    for first in names:
        row = []
        for second in names:
            a = normalized[[word_to_index[w] for w in groups[first]]]
            b = normalized[[word_to_index[w] for w in groups[second]]]
            row.append((a @ b.T).mean().item())
        print(f"  {first:<14} " + " ".join(f"{value:>6.2f}" for value in row))

    print("\nAnalogies (a - b + c):")
    for a, b, c in [("king", "man", "woman"), ("princess", "girl", "boy"), ("queen", "king", "prince")]:
        query = embeddings[word_to_index[a]] - embeddings[word_to_index[b]] + embeddings[word_to_index[c]]
        answer = nearest(embeddings, words, word_to_index, query, [a, b, c], k=1)[0]
        print(f"  {a} - {b} + {c} = {answer[0]} ({answer[1]:.2f})")

    layer = nn.Embedding(len(words), 16)
    with torch.no_grad():
        layer.weight.copy_(embeddings)
    ids = torch.tensor([word_to_index["cat"], word_to_index["queen"]])
    via_lookup = layer(ids)
    via_matmul = F.one_hot(ids, len(words)).float() @ layer.weight
    print(f"\nnn.Embedding lookup equals one_hot @ W: {torch.allclose(via_lookup, via_matmul)}")
    print(f"nn.Embedding parameters: {layer.weight.numel()} ({len(words)} words x 16 dims)")


if __name__ == "__main__":
    main()
