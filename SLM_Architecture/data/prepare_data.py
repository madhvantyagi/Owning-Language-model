"""
Data loading, mixing, and packing for pretraining.

DATA MIX (by tokens):
  70% FineWeb-edu     — educational web text, highest-quality general corpus
  15% OpenWebMath     — math-heavy web pages (biases toward our GRPO target)
  15% StarCoderData   — code (improves structured reasoning)

WHY these three?
  - FineWeb-edu: filtered by an educational-quality classifier. Best signal-per-token
    for small models. Proven by SmolLM2. At our 15B token budget, we can't afford
    to waste tokens on random web data — quality is everything.
  - OpenWebMath: gives the model early exposure to math notation, equations,
    quantitative text. This makes SFT + GRPO on math easier later.
  - StarCoderData: code teaches the model structure, indentation, logic.
    Code and math reasoning are correlated; small models benefit from both.

WHY stream (not download)?
  - Combined corpus is hundreds of GB. Can't fit on disk.
  - Streaming: load one example, tokenize, yield, discard. Constant memory.
  - Tradeoff: network-bound, slower than local. Acceptable.

PACKING:
  Variable-length documents waste GPU memory (you'd pad to max length).
  Packing = concatenate all tokens into one long stream, then chop into
  fixed seq_len+1 blocks. Zero padding. 100% useful tokens per batch.

  The +1 is for next-token prediction: input = tokens[0:seq_len], target = tokens[1:seq_len+1].
"""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# Mix ratios (must sum to 1.0)
MIX = {
    "fineweb_edu": 0.70,
    "openwebmath": 0.15,
    "starcoder": 0.15,
}

DATASET_CONFIG = {
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_key": "text",
    },
    "openwebmath": {
        "path": "open-web-math/openweb-math",
        "name": None,
        "split": "train",
        "text_key": "text",
    },
    "starcoder": {
        "path": "bigcode/starcoderdata",
        "name": None,
        "split": "train",
        "text_key": "content",
    },
}


def get_tokenizer(path: str = "tokenizer/"):
    """Load our trained tokenizer from a directory."""
    return AutoTokenizer.from_pretrained(path)


def load_stream(name: str):
    """Open a streaming connection to one of our three datasets."""
    cfg = DATASET_CONFIG[name]
    if cfg["name"]:
        return load_dataset(cfg["path"], cfg["name"], split=cfg["split"], streaming=True)
    return load_dataset(cfg["path"], split=cfg["split"], streaming=True)


def stream_tokens(name: str, tokenizer, seq_len: int = 2048, infinite: bool = True):
    """
    Tokenize one dataset, pack into seq_len+1 chunks, yield one chunk at a time.

    Packing logic:
      - Tokenize each document into a list of token IDs.
      - Append <eos> at end (marks document boundary — model learns where docs end).
      - Accumulate into a buffer.
      - When buffer has >= seq_len+1 tokens, yield the first seq_len+1, keep the rest.
    """
    cfg = DATASET_CONFIG[name]
    ds = load_stream(name)
    text_key = cfg["text_key"]
    eos_id = tokenizer.eos_token_id

    buffer = []

    for example in ds:
        text = example.get(text_key, "")
        if not text or len(text.strip()) == 0:
            continue

        # Tokenize WITHOUT special tokens — we manage <eos> ourselves.
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)  # document boundary marker

        buffer.extend(tokens)

        # Yield seq_len+1 chunks (the +1 lets us shift by 1 for next-token target)
        while len(buffer) >= seq_len + 1:
            yield buffer[:seq_len + 1]
            buffer = buffer[seq_len + 1:]

    # Tail of finite stream
    if not infinite and buffer:
        # Pad short tail with eos to fill a full block
        buffer = buffer + [eos_id] * (seq_len + 1 - len(buffer))
        yield buffer[:seq_len + 1]


def mix_streams(tokenizer, seq_len: int = 2048, weights: dict = None, seed: int = 42):
    """
    Interleave all three token streams according to MIX weights.

    For each output sequence, randomly pick which dataset to draw from,
    using the weights as probabilities. This gives a 70/15/15 mix on average.

    Random sampling > round-robin because:
      - Round-robin gives exactly 70/15/15 in order, but you get "blocks" of
        the same source. Random sampling gives a smoother mix per batch.
    """
    if weights is None:
        weights = MIX

    rng = random.Random(seed)
    streams = {name: stream_tokens(name, tokenizer, seq_len) for name in weights}
    names = list(weights.keys())
    probs = [weights[n] for n in names]

    while True:
        name = rng.choices(names, weights=probs, k=1)[0]
        try:
            yield next(streams[name])
        except StopIteration:
            # Refill exhausted stream (for infinite training)
            streams[name] = stream_tokens(name, tokenizer, seq_len)


def collate_batch(sequences: list, seq_len: int = 2048):
    """
    Turn a list of token sequences into model-ready tensors.

    INPUT:  list of N sequences, each a list of seq_len+1 token IDs.
    OUTPUT: dict with input_ids and labels, each shape (N, seq_len).

    The shift:
      sequence = [t0, t1, t2, ..., t2048]   (length 2049)
      input    = [t0, t1, t2, ..., t2047]   (length 2048)
      label    = [t1, t2, t3, ..., t2048]   (length 2048)

    So at position i, the model sees input[i] and must predict label[i] = input[i+1].
    This is next-token prediction.
    """
    input_ids = torch.tensor([s[:seq_len] for s in sequences], dtype=torch.long)
    labels = torch.tensor([s[1:seq_len + 1] for s in sequences], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    # Quick sanity check — show one mixed batch
    print("Loading tokenizer...")
    tok = get_tokenizer()

    print("\nStreaming 3 example sequences from the mix:")
    stream = mix_streams(tok, seq_len=2048)
    for i in range(3):
        seq = next(stream)
        print(f"\n  Sequence {i}: {len(seq)} tokens")
        print(f"  First 20 token IDs: {seq[:20]}")
        print(f"  First 20 decoded:   '{tok.decode(seq[:20])}'")
