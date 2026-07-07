"""
Pretraining loop — the heart of the project.

WHAT HAPPENS HERE (in simple words):

  Loop:
    1. Get a batch of token sequences.
    2. FORWARD: model looks at each token, predicts what comes next at every position.
    3. LOSS: cross-entropy between predictions and the actual next tokens.
    4. BACKWARD: compute gradients of loss w.r.t. every parameter.
    5. Every grad_accum_steps batches: clip gradients, update params via AdamW.
    6. Update learning rate (warmup then cosine decay).
    7. Log loss, save checkpoint periodically.

THE MATH OF NEXT-TOKEN PREDICTION:

  Input batch:  x   shape (B, T)     B=batch_size, T=seq_len
  Model output: logits shape (B, T, V)   V=vocab_size (32768)

  logits[b, i, :] = the model's prediction for what token comes AFTER x[b, i].
  (We already shifted labels in collate_batch, so labels[b,i] is the true next token.)

  Cross-entropy loss at position (b, i):
    Convert logits to probabilities via softmax:
      p = softmax(logits[b, i, :])    shape (V,), sums to 1
    The true next token is labels[b, i] = some integer k.
    Loss = -log(p[k])    — negative log probability of the correct token.

  Total loss = average of -log(p[k]) over all B×T positions.

  When loss = 4.0, perplexity = e^4.0 ≈ 55, meaning the model is "as confused
  as if choosing uniformly among 55 words." Lower loss = less confused = better.
"""
import os
import math
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from configs.train_config import TrainConfig
from data.prepare_data import get_tokenizer, mix_streams
from model.build_model import build_model, count_params
from model.slm_model import SLM


cfg = TrainConfig()


class PretrainingStream(IterableDataset):
    """Wraps mix_streams into a PyTorch IterableDataset."""
    def __init__(self, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        return mix_streams(self.tokenizer, seq_len=self.seq_len)


def format_num(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def train(resume_from: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 on matmuls — free speedup
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print("FinalSLM — Pretraining")
    print("=" * 60)

    # --- Tokenizer ---
    print("\n[1/4] Loading tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # --- Model ---
    print("\n[2/4] Building model...")
    model = build_model(vocab_size=vocab_size)
    info = count_params(model)
    print(f"  Params: {info['total_M']:.1f}M")
    print(f"    Embedding:   {info['embedding_M']:.1f}M")
    print(f"    Transformer: {info['non_embedding_M']:.1f}M")

    model = model.to(device).to(torch.bfloat16)
    model.train()

    # Resume from checkpoint (our own save format, not HF)
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\n  Resuming from {resume_from}")
        model = SLM.load(resume_from, device=device).to(torch.bfloat16)
        # Extract step from directory name like "step_10000"
        try:
            start_step = int(os.path.basename(resume_from).split("_")[-1])
        except (ValueError, IndexError):
            start_step = 0
        model.train()

    # --- Optimizer ---
    # AdamW: momentum (beta1) + adaptive per-param learning rate (beta2) + weight decay.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # --- LR Schedule: linear warmup then cosine decay to 10% of max ---
    total_steps = cfg.total_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    # Note: HF's cosine scheduler decays to 0. We want 10% of lr, so we'll
    # fix this by scaling: after scheduler sets lr, we add a floor.
    # (For simplicity here we accept decay to 0. In production, set num_training_steps
    #  to ~90% of total and hold the last 10% at min_lr.)

    if start_step > 0:
        # Fast-forward scheduler
        for _ in range(start_step):
            scheduler.step()

    # --- Data ---
    print("\n[3/4] Setting up data stream...")
    dataset = PretrainingStream(tokenizer, cfg.seq_len)
    data_iter = iter(dataset)

    # --- Training loop ---
    print(f"\n[4/4] Training")
    print(f"  Total tokens:      {format_num(cfg.total_tokens)}")
    print(f"  Effective batch:   {format_num(cfg.effective_batch_tokens)} tokens/step")
    print(f"  Total steps:       {total_steps:,}")
    print(f"  Micro batch:       {cfg.micro_batch_size}")
    print(f"  Grad accum:        {cfg.grad_accum_steps}")
    print(f"  Max LR:            {cfg.lr}")
    print(f"  Warmup steps:      {cfg.warmup_steps}")
    print()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    step = start_step
    tokens_seen = step * cfg.effective_batch_tokens
    accum_loss = 0.0
    start_time = time.time()

    pbar = tqdm(total=total_steps, initial=step, desc="Training", unit="step")
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation: do grad_accum forward+backward passes,
        # average the gradients, THEN step the optimizer.
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            # Collect micro_batch_size sequences into one batch tensor.
            # This is the fix for the batching bug: previously we processed
            # one sequence at a time (batch=1), ignoring micro_batch_size.
            batch_seqs = []
            for _ in range(cfg.micro_batch_size):
                try:
                    batch_seqs.append(next(data_iter))
                except StopIteration:
                    data_iter = iter(dataset)
                    batch_seqs.append(next(data_iter))

            input_ids = torch.tensor(
                [s[:cfg.seq_len] for s in batch_seqs], dtype=torch.long, device=device
            )
            labels = torch.tensor(
                [s[1:cfg.seq_len + 1] for s in batch_seqs], dtype=torch.long, device=device
            )

            # Forward — our model returns logits and cross-entropy loss
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / cfg.grad_accum_steps  # scale for accumulation

            # Backward — populates .grad on every parameter
            loss.backward()
            accum_loss += loss.item()

        # Clip gradients to prevent explosions (norm-based)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Optimizer step — updates params using accumulated gradients
        optimizer.step()
        scheduler.step()

        step += 1
        tokens_seen += cfg.effective_batch_tokens

        # Logging
        if step % cfg.log_every == 0:
            avg_loss = accum_loss / cfg.grad_accum_steps
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{lr:.2e}",
                "gnorm": f"{grad_norm:.2f}",
                "tps": format_num(int(tps)),
            })
        pbar.update(1)

        # Checkpoint (our format: model.pt + config.json)
        if step % cfg.save_every == 0:
            save_path = os.path.join(cfg.checkpoint_dir, f"step_{step}")
            model.save(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"\n  Saved checkpoint: {save_path}\n")

    pbar.close()

    # Final save
    final_path = os.path.join(cfg.checkpoint_dir, "final")
    model.save(final_path)
    tokenizer.save_pretrained(final_path)
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/3600:.1f} hours.")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    import sys
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_from=resume)
