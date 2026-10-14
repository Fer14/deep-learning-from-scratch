# Representation Learning and Embeddings

> An embedding is a list of numbers where "close" means "similar". The whole game is deciding what similar should mean.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 06 Lessons 01-04 (Autoencoders through Contrastive Learning), Phase 04 (Sequence Models: Word Embeddings)
**Time:** ~50 minutes

## Learning Objectives

- Show that an embedding table is a matrix lookup, equal to a one-hot vector times a weight matrix, with sparse gradients
- Build five representations of the same data (raw, random, PCA, autoencoder, supervised) and compare them with kNN accuracy, precision@k and a linear probe
- Explain why cosine similarity on normalized embeddings is the default for retrieval
- Measure how a representation transfers to classes it never saw, and why supervised features can transfer worse than self-supervised ones

## The Problem

Every lesson in this phase produced a vector per input: an autoencoder code, a VAE mean, a contrastive embedding. Language models produce them for tokens, recommenders for users and items, CLIP for images and captions. These vectors feed search engines, clustering, deduplication, recommendation, and retrieval-augmented generation. When someone asks "is this embedding any good?", you need a way to answer that does not depend on one specific downstream model.

You also need to know how to use them. Should you compare embeddings with Euclidean distance or cosine similarity? Does an embedding trained to classify digits 0-4 tell you anything about 5-9? A representation is only as good as the question you ask it, and this lesson puts the common questions side by side on one dataset.

## The Concept

### Two kinds of embeddings

| Kind | Input | How you get the vector | Examples |
|---|---|---|---|
| Lookup embedding | a discrete id (word, user, item) | row `id` of a learned table `E` | word2vec, token embeddings, recommender factors |
| Encoder embedding | a continuous input (image, audio, text) | the output of a network `f(x)` | autoencoder codes, SimCLR `h`, CLIP, sentence encoders |

A lookup embedding is a linear layer applied to a one-hot vector:

```
one_hot(id) @ E  ==  E[id]
```

Indexing is the fast way to compute it. In the backward pass only the rows that were used get a gradient, which is why embedding tables with millions of rows are trainable.

### Three ways to score an embedding

```mermaid
flowchart LR
    F[frozen embeddings] --> K[kNN: label of the nearest neighbors]
    F --> P[precision@k: fraction of the top-k results that share the query's label]
    F --> L[linear probe: logistic regression on top]
```

- **kNN accuracy** asks whether the local neighborhood is pure. No training, sensitive to the distance you pick.
- **Precision@k** is the retrieval view: you search with a query, how many of the top `k` results are relevant?
- **Linear probe** asks whether the information is there and linearly readable, even if the geometry is not tidy.

They often disagree, and each matches a different use case.

### Cosine similarity

```
cos(a, b) = a . b / (||a|| ||b||)
```

Cosine ignores vector length and compares only direction. Many encoders put "how confident" or "how much ink" into the norm, which has nothing to do with "what it is". Normalizing to unit length and taking dot products is the standard in retrieval and is what contrastive losses optimize. On unit vectors, Euclidean distance and cosine are equivalent: `||a - b||^2 = 2 - 2 cos(a, b)`.

### What the training objective puts in the vector

| Objective | Keeps | Throws away |
|---|---|---|
| Reconstruction (autoencoder) | whatever explains the most pixel variance | nothing on purpose |
| Contrastive | what survives the augmentations | what augmentations change |
| Supervised classification | what separates the training classes | everything else, including what would separate new classes |

The last row is the important one. Supervised features can be excellent for the labels they were trained on and surprisingly poor for anything else.

## Build It

### Step 1: Embedding tables

```python
table = torch.randn(10, 4, generator=generator, requires_grad=True)
ids = torch.tensor([3, 3, 7])
by_lookup = table[ids]
by_one_hot = F.one_hot(ids, num_classes=10).float() @ table
```

```
lookup == one-hot @ table: True
rows with nonzero gradient: [3, 7] (row 3 gradient 2, used twice)
```

Row 3 was used twice, so its gradient accumulates both contributions. The other eight rows get exactly zero.

### Step 2: Five representations

All are built from the same 1497 training digits:

- **raw pixels**: the 64 values themselves, the baseline you must always beat
- **random projection**: `relu(x @ W)` with a random `64 x 32` matrix, no training
- **PCA**: the top 32 principal components
- **autoencoder**: the 32D code of a `64 -> 128 -> 32 -> 128 -> 64` autoencoder (no labels)
- **supervised**: the 32D hidden layer of an MLP trained with cross-entropy on the labels

### Step 3: Score them

```python
def knn_accuracy(train_features, y_train, test_features, y_test, k=5):
    similarity = normalize(test_features) @ normalize(train_features).T
    neighbor_labels = y_train[similarity.topk(k, dim=1).indices]
    votes = F.one_hot(neighbor_labels, num_classes=10).sum(dim=1)
    return float((votes.argmax(dim=1) == y_test).float().mean())


def precision_at_k(features, labels, k=5):
    unit = normalize(features)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(float("-inf"))
    neighbors = similarity.topk(k, dim=1).indices
    return float((labels[neighbors] == labels[:, None]).float().mean())
```

```
representation     | kNN acc (cosine) | precision@5 | linear probe
raw pixels (64)    |            0.990 |       0.913 | 0.973
random projection  |            0.940 |       0.785 | 0.933
PCA                |            0.990 |       0.903 | 0.960
autoencoder        |            0.987 |       0.897 | 0.957
supervised         |            0.987 |       0.964 | 0.977
```

Two honest lessons. First, on easy data, raw pixels are a very strong baseline: 8x8 centered digits are almost linearly separable already, and kNN on pixels gets 99%. Always report the raw baseline, or you might celebrate an embedding that adds nothing. Second, the metrics disagree. The supervised embedding barely changes kNN accuracy but clearly wins precision@5 (0.964): its neighborhoods are tighter, which is what a search engine cares about. The random projection loses information on every metric.

### Step 4: Cosine vs Euclidean

To simulate encoders that put irrelevant information in the norm, every test embedding is multiplied by a random factor between 0.2 and 3.2:

```
Euclidean kNN 0.870 | cosine kNN 0.987
```

Euclidean neighbors are now dominated by vector length. Cosine does not change at all, because rescaling does not change direction.

### Step 5: Transfer to unseen classes

Train a supervised encoder only on digits 0-4. Then use kNN to classify digits 5-9 with it, compared with raw pixels and the autoencoder trained on all digits without labels:

```
kNN on digits 5-9 with raw pixels (64)                     : 1.000
kNN on digits 5-9 with autoencoder (all digits, no labels) : 0.994
kNN on digits 5-9 with supervised on 0-4 only              : 0.876
```

The supervised encoder is clearly worse on new classes. It learned exactly what separates 0 from 1 from 2 from 3 from 4, and compressed away the details that separate a 5 from an 8. The label-free autoencoder kept those details. This is the main argument for self-supervised pretraining: labels define a narrow notion of similarity, while reconstruction and contrastive objectives keep a broader one that later tasks can use.

`retrieval.png` shows six test queries with their top 5 neighbors under raw pixels and under the supervised embedding.

## Use It

PyTorch has a layer for each piece:

```python
embedding = nn.Embedding(10, 4)
scores = F.cosine_similarity(supervised_test[:1], supervised_test, dim=1)
top = scores.topk(5)
```

```
nn.Embedding matches the table lookup: True
query label 6 -> top-5 labels [6, 6, 6, 6, 6] similarities [0.996, 0.994, 0.993, 0.992, 0.991]
```

`nn.Embedding` has options worth knowing: `padding_idx` (a row that always stays zero and gets no gradient), `max_norm` (renormalize rows on lookup) and `sparse=True` (sparse gradients for huge tables). `torch.cdist` gives all pairwise Euclidean distances. Beyond a few hundred thousand vectors you move to an approximate nearest neighbor index (FAISS, ScaNN, HNSW) instead of a full similarity matrix.

## Exercises

1. **Easy.** Recompute the table in Step 3 with Euclidean distance instead of cosine. Which representations change most, and why?
2. **Medium.** Add the contrastive encoder from lesson 04 as a sixth representation and rerun Step 5 (train it on all digits). Where does it land between the autoencoder and the supervised encoder?
3. **Hard.** Implement the alignment and uniformity metrics of Wang and Isola: alignment is the mean squared distance between normalized embeddings of two augmentations of the same image; uniformity is `log mean exp(-2 ||u - v||^2)` over random pairs. Compute both for every representation and relate them to precision@5.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Embedding | "A vector for a thing" | A learned mapping from an item or input to a vector where geometry encodes similarity |
| Embedding table | "The lookup layer" | A `num_items x dim` matrix indexed by id; equal to one-hot times a weight matrix |
| Representation | "Features" | The internal vector a model computes, judged by how useful it is for other tasks |
| Cosine similarity | "Angle between vectors" | Dot product of unit-normalized vectors; ignores norms |
| kNN accuracy | "Neighbor vote" | Classify a test point by majority label among its nearest training embeddings |
| Precision@k | "Retrieval quality" | Fraction of the top `k` retrieved items that are relevant to the query |
| Linear probe | "Frozen features + logistic regression" | A linear classifier trained on frozen embeddings to test what information is linearly available |
| Transfer | "Reuse for a new task" | Using a representation on data or classes it was not trained on |

## Further Reading

- [Bengio, Courville and Vincent, Representation Learning: A Review and New Perspectives (arXiv:1206.5538)](https://arxiv.org/abs/1206.5538) - what makes a representation good, and the families of methods to learn one
- [Mikolov et al., Efficient Estimation of Word Representations in Vector Space (arXiv:1301.3781)](https://arxiv.org/abs/1301.3781) - word2vec, the lookup embedding that started the trend
- [Wang and Isola, Understanding Contrastive Representation Learning through Alignment and Uniformity (arXiv:2005.10242)](https://arxiv.org/abs/2005.10242) - two simple metrics for embedding geometry
- [PyTorch docs: nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) - `padding_idx`, `max_norm` and sparse gradients
