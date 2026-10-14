import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

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

PRETOKENIZE_PATTERN = re.compile(r" ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+")


def pretokenize(text):
    return PRETOKENIZE_PATTERN.findall(text)


def merge_pair(ids, pair, new_id):
    merged = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def count_pairs(chunks):
    counts = Counter()
    for chunk in chunks:
        counts.update(zip(chunk, chunk[1:]))
    return counts


def train_bpe(text, num_merges):
    chunks = [list(piece.encode("utf-8")) for piece in pretokenize(text)]
    merges = {}
    for step in range(num_merges):
        counts = count_pairs(chunks)
        if not counts:
            break
        best_pair = max(counts, key=lambda pair: (counts[pair], -pair[0], -pair[1]))
        if counts[best_pair] < 2:
            break
        new_id = 256 + step
        chunks = [merge_pair(chunk, best_pair, new_id) for chunk in chunks]
        merges[best_pair] = new_id
    return merges


def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (left, right), new_id in merges.items():
        vocab[new_id] = vocab[left] + vocab[right]
    return vocab


def encode_chunk(chunk_bytes, merges):
    ids = list(chunk_bytes)
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge_pair(ids, pair, merges[pair])
    return ids


def encode(text, merges):
    ids = []
    for piece in pretokenize(text):
        ids.extend(encode_chunk(piece.encode("utf-8"), merges))
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def show_tokens(text, merges, vocab):
    ids = encode(text, merges)
    pieces = [vocab[i].decode("utf-8", errors="backslashreplace") for i in ids]
    return ids, "|".join(pieces)


def main():
    train_text = CORPUS
    print(f"corpus: {len(train_text)} characters, {len(train_text.encode('utf-8'))} bytes")

    print("\nstep 1: three ways to cut text")
    words = train_text.split()
    chars = sorted(set(train_text))
    print(f"  characters: {len(train_text):5d} tokens, vocab {len(chars)}")
    print(f"  words:      {len(words):5d} tokens, vocab {len(set(words))}")
    print(f"  bytes:      {len(train_text.encode('utf-8')):5d} tokens, vocab 256")

    print("\nstep 2: pre-tokenization")
    print(f"  {pretokenize('the keeper climbed 112 stairs, slowly.')}")

    print("\nstep 3: train BPE")
    merges = train_bpe(train_text, num_merges=400)
    vocab = build_vocab(merges)
    print(f"  learned {len(merges)} merges, vocab size {len(vocab)}")
    first = [vocab[new_id].decode("utf-8", errors="replace") for new_id in list(merges.values())[:15]]
    print(f"  first merges: {first}")
    last = [vocab[new_id].decode("utf-8", errors="replace") for new_id in list(merges.values())[-8:]]
    print(f"  last merges:  {last}")

    print("\nstep 4: encode and decode")
    for sample in ["the keeper lit the lamp.", "lighthouses and harbors", "a café in Zürich"]:
        ids, pieces = show_tokens(sample, merges, vocab)
        round_trip = decode(ids, vocab) == sample
        print(f"  {sample!r}: {len(sample.encode('utf-8'))} bytes -> {len(ids)} tokens  {pieces}  round-trip={round_trip}")

    print("\nstep 5: compression vs number of merges")
    held_out = "the keeper and the baker watched the boats come home before the storm."
    merge_counts = [0, 25, 50, 100, 200, len(merges)]
    ratios = []
    for n in merge_counts:
        partial = dict(list(merges.items())[:n])
        train_ratio = len(train_text.encode("utf-8")) / len(encode(train_text, partial))
        held_ratio = len(held_out.encode("utf-8")) / len(encode(held_out, partial))
        ratios.append((train_ratio, held_ratio))
        print(f"  merges {n:3d} | vocab {256 + n:3d} | bytes/token train {train_ratio:.2f} held-out {held_ratio:.2f}")

    print("\nuse it: token ids become vectors through an embedding table")
    torch.manual_seed(0)
    embedding = torch.nn.Embedding(len(vocab), 16)
    ids = torch.tensor(encode("the lighthouse", merges))
    vectors = embedding(ids)
    manual = embedding.weight[ids]
    print(f"  ids {ids.tolist()} -> vectors {tuple(vectors.shape)}, same as indexing the weight matrix: {torch.equal(vectors, manual)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([256 + n for n in merge_counts], [r[0] for r in ratios], marker="o", label="training text")
    ax.plot([256 + n for n in merge_counts], [r[1] for r in ratios], marker="s", label="held-out sentence")
    ax.set_xlabel("vocabulary size")
    ax.set_ylabel("bytes per token")
    ax.set_title("BPE compression")
    ax.legend()
    fig.tight_layout()
    out_path = Path(__file__).parent / "bpe_compression.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved {out_path.name}")


if __name__ == "__main__":
    main()
