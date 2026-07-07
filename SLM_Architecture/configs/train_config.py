"""
Training hyperparameters for pretraining.

Each value has a reason. Don't change one without understanding why.

COMPUTE BUDGET (honest math):
  - 600 Colab units ≈ 200 A100-hours total
  - Reserve ~80 hours for SFT + GRPO later
  - Pretraining budget: ~120 A100-hours
  - 360M model on A100 40GB: realistic ~30-40% MFU
  - Achievable: 120 hr × 3600 s × 150 TFLOPs × 0.35 MFU / (6 × 360M FLOPs/token)
              ≈ 10-15 BILLION tokens

  So total_tokens = 15B. This is UNDERTRAINED vs SmolLM2-360M (4T tokens).
  The model will be coherent but weak on knowledge. That's the honest tradeoff
  for the size you chose. The RL stage is where the interesting work happens.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # === Data ===
    total_tokens: int = 15_000_000_000  # 15B — see budget math above
    seq_len: int = 2048

    # === Batch ===
    # Effective batch = micro_batch × grad_accum × seq_len
    #               = 8 × 32 × 2048 = 524,288 tokens per optimizer step
    # NOTE: our attention is plain PyTorch (not FlashAttention), so it
    # materializes the full (B, H, T, T) attention matrix. micro_batch=8
    # with gradient checkpointing fits comfortably on A100 40GB.
    # Tune upward after checking GPU memory during the debug run.
    micro_batch_size: int = 8    # sequences per forward pass
    grad_accum_steps: int = 32   # accumulate grads over this many micro-batches
    # ^ Together: ~524K tokens/step. Large batch = stable training, good throughput.

    # === Optimizer (AdamW) ===
    lr: float = 4e-4              # max learning rate. nanogpt uses 6e-4 for 350M;
                                  # we use 4e-4 to be safe with bf16 stability.
    weight_decay: float = 0.1     # standard. Prevents overfitting on weights.
    beta1: float = 0.9            # Adam momentum
    beta2: float = 0.95           # 0.95 NOT 0.999 — critical for LLMs.
                                  # 0.999 adapts too slowly; 0.95 is the Llama/nanogpt recipe.
    eps: float = 1e-8
    grad_clip: float = 1.0        # clip grad norm to 1.0 — prevents explosions.

    # === LR Schedule ===
    warmup_steps: int = 2000      # linearly ramp LR from 0 to lr over first 2k steps
                                  # ~2% of total steps. Prevents early instability.
    # cosine decay to 10% of max over the rest of training (handled in train.py)

    # === Precision ===
    dtype: str = "bfloat16"       # bf16 not fp16. Same speed, far better stability.
                                  # bf16 has same exponent range as fp32 (no overflow).

    # === Steps (computed) ===
    @property
    def effective_batch_tokens(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.effective_batch_tokens  # ~28,600

    # === Logging / checkpointing ===
    log_every: int = 10
    eval_every: int = 1000
    save_every: int = 2000
    checkpoint_dir: str = "./checkpoints"

    # === Reproducibility ===
    seed: int = 42
