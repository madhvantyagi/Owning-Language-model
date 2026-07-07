"""
Build our from-scratch SLM and verify parameter count.

This now uses OUR model (model/slm_model.py), not HuggingFace's LlamaForCausalLM.
Every line of the architecture is ours.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from configs.model_config import SLMConfig, get_model_config
from model.slm_model import SLM, count_params


def build_model(vocab_size: int = 32768) -> SLM:
    """Construct our SLM with random init. Ready for training."""
    config = get_model_config(vocab_size=vocab_size)
    return SLM(config)


if __name__ == "__main__":
    model = build_model()
    info = count_params(model)
    print(f"Total params:     {info['total_M']:.1f}M")
    print(f"  Embedding:      {info['embedding_M']:.1f}M")
    print(f"  Transformer:    {info['non_embedding_M']:.1f}M")

    print("\nForward + loss test:")
    input_ids = torch.randint(0, 32768, (2, 10))
    labels = torch.randint(0, 32768, (2, 10))
    out = model(input_ids, labels=labels)
    print(f"  Logits: {out.logits.shape}")
    print(f"  Loss:   {out.loss.item():.4f}")
