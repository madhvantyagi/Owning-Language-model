"""
Validation loss + perplexity evaluation.

WHY val loss is THE metric during pretraining:
  - It's the direct measure of "how well does the model predict text it hasn't seen?"
  - Goes down = model learning general patterns (good).
  - Goes down then up = overfitting (bad — but unlikely with 15B tokens of web data).
  - Stays flat = something is broken (LR wrong, data corrupted, bug).

WHY perplexity:
  Perplexity = exp(loss). It's the "effective vocabulary size" the model is choosing among.
  - Perplexity = 1.0 = perfect prediction (impossible on natural text)
  - Perplexity = 32768 = uniform random guessing (vocab size)
  - Perplexity = 20 = "as if uniform among 20 equally-likely next tokens"
  - Good small-model perplexity on web text: ~10-25 after serious pretraining.

We sample val sequences from FineWeb-edu (the dominant 70% of our mix).
For a more thorough eval, also run lm-evaluation-harness on standard benchmarks:
  hellaswag, arc_easy, piqa, openbookqa
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from data.prepare_data import get_tokenizer, stream_tokens
from configs.train_config import TrainConfig
from model.slm_model import SLM

cfg = TrainConfig()


@torch.no_grad()
def evaluate(model_path: str, n_sequences: int = 200):
    """
    Compute average cross-entropy loss and perplexity on held-out sequences.
    Loads OUR model (model.pt + config.json format).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Loading model from {model_path}...")
    model = SLM.load(model_path, device=device).to(dtype)
    model.eval()

    tokenizer = get_tokenizer()

    print(f"Sampling {n_sequences} sequences from FineWeb-edu for eval...")
    stream = stream_tokens("fineweb_edu", tokenizer, cfg.seq_len)

    total_loss_sum = 0.0
    total_tokens = 0

    for _ in tqdm(range(n_sequences), desc="Evaluating"):
        try:
            seq = next(stream)
        except StopIteration:
            break

        input_ids = torch.tensor(
            [seq[:cfg.seq_len]], dtype=torch.long, device=device
        )
        labels = torch.tensor(
            [seq[1:cfg.seq_len + 1]], dtype=torch.long, device=device
        )

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        # outputs.loss is already averaged over seq_len positions
        total_loss_sum += outputs.loss.item() * cfg.seq_len
        total_tokens += cfg.seq_len

    avg_loss = total_loss_sum / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    print("\n" + "=" * 40)
    print(f"Val loss:     {avg_loss:.4f}")
    print(f"Perplexity:   {perplexity:.2f}")
    print("=" * 40)
    print("\nReference points:")
    print(f"  Random (vocab={tokenizer.vocab_size}): loss={math.log(tokenizer.vocab_size):.4f}")
    print(f"  Good small model on web text:          loss~3.0-3.5, ppl~20-35")

    return avg_loss, perplexity


# Generate sample text from the model — qualitative sanity check
@torch.no_grad()
def generate_sample(model_path: str, prompt: str = "The cat sat", max_new_tokens: int = 50):
    """Generate text to eyeball model quality. Useful early in training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    model = SLM.load(model_path, device=device).to(dtype)
    tokenizer = get_tokenizer()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated:\n{text}")
    return text


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final"

    if "--generate" in sys.argv:
        prompt = sys.argv[sys.argv.index("--generate") + 1] if len(sys.argv) > sys.argv.index("--generate") + 1 else "The cat sat"
        generate_sample(model_path, prompt)
    else:
        evaluate(model_path)
