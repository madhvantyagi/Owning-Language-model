"""
SLMConfig — architecture configuration for our from-scratch SLM.

This is OUR config, not HuggingFace's LlamaConfig. We own it.
Every value is deliberate (see comments).
"""
from dataclasses import dataclass


@dataclass
class SLMConfig:
    # === Vocab ===
    vocab_size: int = 32768       # our BPE tokenizer size

    # === Model dimensions ===
    d_model: int = 960            # hidden width. 960 = 15 heads × 64 head_dim.
    n_layers: int = 30           # depth. Deep+thin = better reasoning at small scale.
    n_heads: int = 15            # attention heads. 960/15 = 64 head_dim (FlashAttention-friendly).
    intermediate_size: int = 2560  # SwiGLU inner size = 8/3 × d_model exactly.

    # === Sequence ===
    max_seq_len: int = 2048

    # === Misc ===
    norm_eps: float = 1e-5       # RMSNorm epsilon (prevents div-by-zero)
    rope_theta: float = 10000.0  # RoPE base frequency
    tie_embeddings: bool = True  # share input embedding and output LM head

    # === Special tokens ===
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # === Training memory ===
    use_gradient_checkpointing: bool = True  # recompute during backward — saves ~60% memory

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads  # 64


def get_model_config(vocab_size: int = 32768) -> SLMConfig:
    """Factory: returns the config for our 360M SLM."""
    return SLMConfig(vocab_size=vocab_size)
