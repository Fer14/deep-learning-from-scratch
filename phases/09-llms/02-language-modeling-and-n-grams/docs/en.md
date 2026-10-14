# Language Modeling and N-grams

> A language model is a machine that answers one question over and over: what comes next?

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 09 (LLMs: Tokenization and BPE), Phase 04 (Sequence Models: RNN from Scratch)
**Time:** ~60 minutes

## Learning Objectives

- Write the chain rule factorization that every autoregressive language model, from bigrams to GPT-4, is built on
- Build count-based character n-gram models with add-k smoothing and evaluate them with perplexity
- Explain the sparsity wall: why longer contexts overfit and how smoothing and interpolation help
- Train a neural bigram with gradient descent and show it learns the same table as counting

## The Problem

You want a number for "how likely is this text?" and a way to generate more text that looks like it. Both needs reduce to the same object: a probability distribution over the next token given the previous ones.

The cheapest way to get that distribution is to count. If after `"th"` you saw `"e"` 300 times and `"a"` 20 times, predict `"e"` with high probability. This is the n-gram model, and it powered speech recognition and machine translation for decades. It also hits a wall you need to understand, because the entire neural language model story (RNNs, then transformers) is about getting past it.

## The Concept

### The chain rule

Any probability over a sequence factorizes exactly:

```
P(x_1, ..., x_T) = P(x_1) * P(x_2 | x_1) * P(x_3 | x_1, x_2) * ... * P(x_T | x_1..x_{T-1})
```

No approximation yet. A GPT models every factor with one network. An n-gram model makes the Markov assumption: only the last `n - 1` tokens matter.

```
P(x_t | x_1..x_{t-1})  ~=  P(x_t | x_{t-n+1}..x_{t-1})
```

| n | Name | Context |
|---|------|---------|
| 1 | unigram | nothing |
| 2 | bigram | previous character |
| 3 | trigram | previous two |
| 6 | 6-gram | previous five |

### Estimating by counting

The maximum likelihood estimate is a ratio of counts:

```
P(c | context) = count(context + c) / count(context)
```

Any combination never seen in training gets probability zero, and one zero makes the whole sequence probability zero. Add-k smoothing pretends every character was seen `k` extra times:

```
P(c | context) = (count(context + c) + k) / (count(context) + k * V)
```

`V` is the vocabulary size. With a large `k` you flatten everything toward uniform; with a tiny `k` unseen events are almost impossible.

### Perplexity

The training objective of every language model is the average negative log-likelihood (cross-entropy) per token. Perplexity is its exponential:

```
PPL = exp( -(1/T) * sum_t log P(x_t | context_t) )
```

Read it as "the model is as confused as if it were choosing uniformly among PPL options". A uniform model over 30 characters has perplexity 30. A perfect model has perplexity 1.

### The sparsity wall

The number of possible contexts grows as `V^(n-1)`. With 30 characters, a 6-gram has 24 million possible contexts, and your 3 kB corpus contains about 1,700 of them. On held-out text a third of the contexts were never seen, and the model falls back to uniform guesses there.

```
order n:     1     2     3     4     5     6
train PPL:  16.9   7.6   3.8   2.9   2.8   3.0
val PPL:    17.0   7.8   5.0   5.3   6.5   8.1
```

Training perplexity drops while validation perplexity turns back up after `n = 3`: classic overfitting, caused by counting each context independently. The model has no idea that `"the keep"` and `"he keep"` should predict similar things. Neural models fix exactly this by mapping contexts to vectors, so similar contexts share statistics.

### Interpolation

A cheap fix: mix orders.

```
P(c | ctx) = l1 * P1(c) + l2 * P2(c | 1 char) + ... + l5 * P5(c | 4 chars),   sum l = 1
```

When the long context was never seen, the short ones still give a sensible answer. Kneser-Ney smoothing is the refined version of this idea that dominated before neural LMs.

## Build It

```bash
uv run python phases/09-llms/02-language-modeling-and-n-grams/code/main.py
```

### Step 1: Count

```python
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
```

The whole model is a dictionary from context string to a counter of next characters. Padding with newlines gives the first characters a context.

### Step 2: Perplexity

```python
def perplexity(model, text, history):
    padded = history + text
    offset = len(history)
    total_log_prob = 0.0
    for i in range(offset, len(padded)):
        total_log_prob += math.log(model.prob(padded[:i], padded[i]))
    return math.exp(-total_log_prob / len(text))
```

Sum log-probabilities, never multiply raw probabilities: a product of 300 numbers below 1 underflows to zero.

### Step 3: Inspect and tune

The bigram distribution after `'t'` is `'h': 0.48, ' ': 0.14, 'e': 0.12`: the model learned that "th" is the most common pair in English. Sweeping the smoothing of the 4-gram shows a sweet spot: `k = 1` gives val PPL 9.7, `k = 0.01` gives 5.3, `k = 0.001` gives 6.8.

### Step 4: Interpolate

```python
class InterpolatedModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def prob(self, context, char):
        return sum(w * m.prob(context, char) for m, w in zip(self.models, self.weights))
```

Mixing orders 1 to 5 reaches val PPL 4.9, better than any single order.

### Step 5: Generate

Sampling repeatedly from `P(next | context)` and appending the result is how every autoregressive model generates text. The samples show both failure modes: the bigram produces word-like noise (`"the keeperil atornesm t tathe"`), and the high-order models collapse into uniform garbage the moment they wander into an unseen context. The interpolated model produces fragments like `"the counted of the stairs and to read for scraps"`: local phrases copied from training, glued without global sense.

## Use It

A neural network can learn the same distribution. The neural bigram is a `V x V` table of logits trained with cross-entropy:

```python
logits_table = torch.zeros(vocab_size, vocab_size, requires_grad=True)
optimizer = torch.optim.Adam([logits_table], lr=learning_rate)
for _ in range(steps):
    loss = F.cross_entropy(logits_table[inputs], targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```
neural bigram: val ppl 7.91
count bigram:  val ppl 7.78
P(h|t): neural 0.489 vs counts 0.485
```

Minimizing cross-entropy over a free table recovers the count ratios: maximum likelihood gives the same answer whether you count or descend. The neural version is not better here, but it is a different kind of object. Replace the table lookup with an embedding plus an MLP over several previous characters (Bengio et al., 2003) and similar contexts start sharing parameters. Replace the MLP with attention over the whole context and you get the GPT you build in the next lesson. The objective, `F.cross_entropy` on next-token targets, never changes.

## Exercises

1. **Easy.** Compute the perplexity of the unigram model on the validation text and check that it is close to `exp` of the entropy of the character frequencies.
2. **Medium.** Implement stupid backoff: use the highest order whose context was seen, otherwise drop to a shorter one. Compare its samples with the interpolated model.
3. **Hard.** Choose the five interpolation weights by minimizing validation perplexity with the EM algorithm or a simple grid search, and report how much perplexity you gain over the hand-picked weights.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Language model | "A chatbot" | A probability distribution over token sequences, usually factored as next-token predictions |
| Autoregressive | "Generates one word at a time" | Each factor conditions on all previously generated tokens, via the chain rule |
| N-gram | "Word statistics" | A Markov model that conditions only on the previous `n - 1` tokens |
| Add-k smoothing | "Laplace smoothing" (k = 1) | Adds `k` pseudo-counts to every outcome so nothing has zero probability |
| Perplexity | "How surprised the model is" | `exp` of the average negative log-likelihood per token |
| Cross-entropy | "The loss" | Average `-log P(true next token)`; the training objective of GPT models |
| Interpolation | "Mixing models" | A weighted sum of n-gram distributions of different orders |

## Further Reading

- [Jurafsky and Martin: Speech and Language Processing, Chapter 3 (N-gram Language Models)](https://web.stanford.edu/~jurafsky/slp3/) - the standard treatment, including Kneser-Ney smoothing
- [Bengio et al. (2003): A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) - the first neural LM, built to beat n-gram sparsity
- [Andrej Karpathy: The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) - counting bigrams and training the neural bigram, step by step
- [Shannon (1951): Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) - the original experiments on predicting the next letter
