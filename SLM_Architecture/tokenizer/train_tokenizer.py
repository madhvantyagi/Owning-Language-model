"""
Train a byte-level BPE tokenizer on FineWeb-edu.

WHAT IS BPE (Byte Pair Encoding), in simple words:

  Start: every byte (0-255) is a token. Text is a sequence of bytes.
  Repeat until vocab is full:
    1. Count how often each adjacent pair of tokens occurs in the corpus.
    2. Find the most frequent pair.
    3. Merge them into one new token. Add to vocab.
    4. Apply this merge everywhere in the corpus.

  Example on "ab ab bc ab ab":
    Pairs: (a,b)=4, (b,c)=1, (b, )=1, etc.  → merge (a,b) into "ab"
    Now: "ab ab bc ab ab"  (4 tokens instead of 10)
    Repeat: (ab, ' ')=3 → merge into "ab "

WHY byte-level (not character-level)?
  - Bytes cover every possible input: English, code, emoji, typos, anything.
  - No <unk> token ever needed. Every text is fully tokenizable.
  - Cost: slightly longer sequences than char-level for non-English.

WHY BPE (not word-level or Unigram)?
  - Word-level: huge vocab, can't handle new words or typos.
  - BPE: learns subwords ("playing" → "play" + "ing"), generalizes.
  - Unigram (SentencePiece alternative): similar quality, different algorithm.
  - BPE is the GPT/Llama/Mistral standard — most tooling, most docs.

CONCRETE EXAMPLE:
  Text: "The cat sat"
  After training (with a real corpus), tokenizes to something like:
    ["The", "Ġcat", "Ġsat"]  → IDs [464, 3797, 6471]
  Note the Ġ — it represents a leading space (byte-level artifact).
  " cat" becomes one token because spaces-before-common-words are frequent.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


VOCAB_SIZE = 32768


def text_iterator(n_examples: int = 2_000_000):
    """
    Stream FineWeb-edu sample. 2M examples ≈ several GB of text.
    That's plenty to train a 32K vocab BPE.
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )
    for i, ex in enumerate(ds):
        if i >= n_examples:
            break
        yield ex["text"]


def train_tokenizer(save_dir: str = "tokenizer/"):
    # Set up byte-level BPE
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        # ByteLevel alphabet = all 256 bytes. BPE merges build on top of these.
        initial_alphabet=ByteLevel.alphabet(),
    )

    print("Training tokenizer on FineWeb-edu sample (2M examples)...")
    print("This streams ~several GB of text. Expect ~10-20 min on good connection.\n")

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer,
        length=2_000_000,
    )

    # Wrap in HuggingFace's PreTrainedTokenizerFast so AutoTokenizer can load it
    # with proper special token configuration (bos/eos/pad/unk).
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    os.makedirs(save_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(save_dir)
    print(f"\nSaved to directory: {save_dir}")
    print(f"Final vocab size: {fast_tokenizer.vocab_size}")

    # Sanity check with example sentences
    print("\n=== Tokenization examples ===")
    examples = [
        "The cat sat on the mat.",
        "def add(a, b): return a + b",
        "If x + 3 = 7, then x = 4.",
        "supercalifragilisticexpialidocious",
    ]
    for text in examples:
        enc = fast_tokenizer.encode(text)
        print(f"\n  Text:    '{text}'")
        print(f"  Tokens:  {enc.tokens}")
        print(f"  IDs:     {enc.ids}")
        print(f"  Decode:  '{fast_tokenizer.decode(enc.ids)}'")


if __name__ == "__main__":
    train_tokenizer()
