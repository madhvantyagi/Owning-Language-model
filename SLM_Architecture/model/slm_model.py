"""
slm_model.py — Our Small Language Model, written from scratch in PyTorch.

Every architectural line is ours. We use torch primitives (matmul, softmax,
cross_entropy) because reimplementing matrix multiply is not the lesson.

ARCHITECTURE (Llama-style):
  x = embed(token_ids)                              # (B, T) -> (B, T, d_model)
  for layer in range(n_layers):
      x = x + Attention(RMSNorm(x))                # pre-norm + residual
      x = x + MLP(RMSNorm(x))                      # pre-norm + residual
  x = RMSNorm(x)
  logits = x @ embed.weight.T   (tied LM head)     # (B, T, d_model) -> (B, T, V)

COMPONENTS:
  1. RMSNorm       — normalization (replaces LayerNorm)
  2. RoPE          — positional encoding (rotates Q and K by position)
  3. Attention     — causal multi-head self-attention
  4. MLP           — SwiGLU (gated)
  5. DecoderLayer  — one block (norm + attn + norm + mlp + residuals)
  6. SLM           — full model: embed + layers + head + loss + generate
"""
import os
import json
import math
import sys
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from configs.model_config import SLMConfig


# ============================================================
# 1. RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    LayerNorm:  y = (x - mean) / std * gamma + beta
    RMSNorm:    y = x / rms(x) * gamma        where rms(x) = sqrt(mean(x^2))

    Dropping the mean subtraction doesn't hurt quality and saves compute.
    Llama, Mistral, SmolLM all use RMSNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma — learned per-feature scale

    def forward(self, x):
        # x: (batch, seq, dim). Normalize over the last dim.
        # rsqrt = 1/sqrt. We add eps inside sqrt to avoid division by zero.
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ============================================================
# 2. RoPE — Rotary Position Embedding
# ============================================================
def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute cos/sin tables for RoPE.

    RoPE idea (in simple words):
      Instead of adding a position vector to each token, we ROTATE the
      Query and Key vectors. The rotation angle depends on the position.
      Token at position p has its (2i, 2i+1) pair of dims rotated by angle:
          angle_{p,i} = p / theta^(2i / head_dim)

      Consequence: the dot product between Q at position p and K at position q
      depends only on (p - q) — relative position. That's the magic.

      theta controls how fast the angle grows. theta=10000 is the original value.
      Bigger theta = slower rotation = better long-distance extrapolation.
    """
    # Frequencies for each dim pair: 1 / theta^(2i/head_dim) for i = 0,1,...,head_dim/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Positions: 0, 1, 2, ..., max_seq_len-1
    positions = torch.arange(max_seq_len).float()
    # angle[p, i] = p * inv_freq[i]
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()  # each shape: (max_seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    """
    Apply RoPE rotation to x (queries or keys).

    x shape:   (batch, n_heads, seq, head_dim)
    cos, sin:  (seq, head_dim/2)

    For each pair of dims (2i, 2i+1):
      x'[2i]   = x[2i] * cos - x[2i+1] * sin
      x'[2i+1] = x[2i] * sin + x[2i+1] * cos
    (A 2D rotation by angle theta.)
    """
    # Split into even-indexed and odd-indexed halves
    x_even = x[..., 0::2]  # (B, H, T, head_dim/2)
    x_odd = x[..., 1::2]   # (B, H, T, head_dim/2)

    # Broadcast cos/sin to match x: (1, 1, T, head_dim/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Rotate
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    # Interleave back: (B, H, T, head_dim/2, 2) -> (B, H, T, head_dim)
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)


# ============================================================
# 3. Multi-Head Causal Self-Attention
# ============================================================
class Attention(nn.Module):
    """
    The core of the transformer. Each token looks at all previous tokens
    (and itself) and gathers information.

    For each head h:
      Q = x @ W_Q   (B, T, head_dim)   "what am I looking for?"
      K = x @ W_K   (B, T, head_dim)   "what do I have?"
      V = x @ W_V   (B, T, head_dim)   "what info do I contribute?"

      scores = Q @ K^T / sqrt(head_dim)            (B, T, T)
      mask future positions (causal)
      weights = softmax(scores)                    (B, T, T) rows sum to 1
      out = weights @ V                            (B, T, head_dim)

    Multiple heads run in parallel, each learning different relationships.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Single fused projection for Q, K, V (3 * d_model output).
        # More efficient than three separate projections.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Output projection: concatenate heads back to d_model
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape  # batch, seq_len, d_model
        H, D = self.n_heads, self.head_dim

        # Project to Q, K, V then split
        qkv = self.qkv_proj(x)              # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)      # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Apply RoPE to Q and K (NOT V — V is content, position is encoded in Q/K)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Attention scores: (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        # Causal mask: position i cannot see positions j > i.
        # Upper triangle (excluding diagonal) -> set to -inf so softmax -> 0.
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over keys (last dim): each row sums to 1.
        weights = F.softmax(scores, dim=-1)

        # Weighted sum of values: (B, H, T, head_dim)
        out = weights @ v

        # Concatenate heads back: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(out)


# ============================================================
# 4. SwiGLU MLP
# ============================================================
class MLP(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit).

    Standard MLP:  y = W_down( GELU( W_up(x) ) )
    SwiGLU:        y = W_down( silu(W_gate(x)) * W_up(x) )

    The gate path lets the model dynamically zero out features per-token.
    silu(x) = x * sigmoid(x)  — a smooth ReLU.

    Three matrices (gate, up, down) vs two in standard MLP.
    intermediate_size = 8/3 * d_model keeps total MLP params balanced.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x):
        # gate and up both expand x to intermediate_size; multiply (gating); down shrinks back.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# 5. Decoder Layer (one transformer block)
# ============================================================
class DecoderLayer(nn.Module):
    """
    One block:
      x = x + Attention(RMSNorm(x))    # pre-norm + residual
      x = x + MLP(RMSNorm(x))          # pre-norm + residual

    Pre-norm (normalize BEFORE the sublayer) is critical for stable deep training.
    Original Transformer used post-norm and was hard to train past ~12 layers.

    Residual connections (the "+ x") let gradients flow directly to early layers,
    preventing the vanishing gradient problem in deep networks.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 6. Full SLM
# ============================================================
class SLM(nn.Module):
    """
    Full Small Language Model — embedding + N decoder layers + LM head.

    forward(input_ids, labels=None):
      Returns object with .logits (B, T, vocab) and .loss (scalar or None).
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.use_gradient_checkpointing

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])

        # Final norm (before LM head)
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head — tied with embedding if config.tie_embeddings
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None  # we'll use self.embed.weight directly

        # Precompute RoPE tables (not learnable — register as non-persistent buffers)
        head_dim = config.head_dim
        cos, sin = precompute_rope(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights. std=0.02 is the GPT-2 standard.
        Without proper init, training is unstable or fails to start.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        """
        input_ids: (B, T) token IDs
        labels:    (B, T) next-token targets (optional)

        Returns: SimpleNamespace with .logits (B, T, V) and .loss (scalar | None)
        """
        B, T = input_ids.shape

        # 1. Token embedding lookup
        x = self.embed(input_ids)  # (B, T, d_model)

        # 2. Slice RoPE tables to current sequence length
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        # 3. Pass through decoder layers
        #    Gradient checkpointing: recompute each layer's forward during backward
        #    instead of storing all activations. Trades ~30% more compute for ~60%
        #    less memory. Essential for fitting 30 layers on a single GPU.
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False
                )
            else:
                x = layer(x, cos, sin)

        # 4. Final norm
        x = self.norm_f(x)

        # 5. LM head -> logits over vocab
        if self.lm_head is None:
            # Tied: reuse embedding matrix transposed
            logits = x @ self.embed.weight.t()  # (B, T, vocab)
        else:
            logits = self.lm_head(x)

        # 6. Optional cross-entropy loss (next-token prediction)
        loss = None
        if labels is not None:
            # F.cross_entropy wants (N, C) and (N,). Flatten spatial dims.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return SimpleNamespace(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=50):
        """
        Autoregressive generation with top-k sampling.

        At each step:
          1. Forward pass, take logits at the last position.
          2. Scale by temperature (lower = sharper/more deterministic).
          3. Keep only top_k highest-probability tokens.
          4. Sample from the resulting distribution.
          5. Append sampled token, repeat.
        """
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            idx = input_ids[:, -self.config.max_seq_len:]
            logits = self.forward(idx).logits[:, -1, :]  # (B, vocab)

            if temperature > 0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids

    def save(self, path: str):
        """Save model weights + config to a directory."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load model from a directory saved by .save()."""
        with open(os.path.join(path, "config.json")) as f:
            config = SLMConfig(**json.load(f))
        model = cls(config)
        state = torch.load(os.path.join(path, "model.pt"), map_location=device)
        model.load_state_dict(state)
        return model.to(device)


def count_params(model: nn.Module) -> dict:
    """Count parameters, broken down."""
    total = sum(p.numel() for p in model.parameters())
    embedding = model.embed.weight.numel()
    return {
        "total": total,
        "total_M": total / 1e6,
        "embedding_M": embedding / 1e6,
        "non_embedding_M": (total - embedding) / 1e6,
    }


if __name__ == "__main__":
    config = SLMConfig()
    model = SLM(config)
    info = count_params(model)

    print(f"Total params:     {info['total_M']:.1f}M")
    print(f"  Embedding:      {info['embedding_M']:.1f}M")
    print(f"  Transformer:    {info['non_embedding_M']:.1f}M")
    print(f"  Per layer avg:  {info['non_embedding_M'] / config.n_layers:.1f}M")

    # Forward pass test
    print("\nForward pass test:")
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    out = model(input_ids)
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Logits shape: {out.logits.shape}")
    print(f"  Expected:     torch.Size([2, 10, {config.vocab_size}])")

    # Loss test
    labels = torch.randint(0, config.vocab_size, (2, 10))
    out = model(input_ids, labels=labels)
    print(f"  Loss:         {out.loss.item():.4f}  (random init ~ log({config.vocab_size}) = {math.log(config.vocab_size):.4f})")
