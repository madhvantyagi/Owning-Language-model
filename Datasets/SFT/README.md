<!--
    Overview of the SFT dataset blend for Qwen3-8B-Reasoning-SFT.
    Tracks all data sources, row allocations, and file mappings.
-->
# SFT Datasets

This folder tracks the supervised fine-tuning datasets used by `Qwen3-8B-Reasoning-SFT`.
The current SFT blend targets 10,000 rows and normalizes every source into chat messages where assistant turns follow:

```text
<think>...</think>
final answer
```

## Current Blend

| Dataset | Link | Default rows | Share | Main use |
| --- | --- | ---: | ---: | --- |
| NovaSky-AI/Sky-T1_data_17k | https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k | 1,960 | 19.6% | Hard math/code long-reasoning examples |
| Jackrong/Competitive-Programming-python-blend | https://huggingface.co/datasets/Jackrong/Competitive-Programming-python-blend | 4,900 | 49% | Python competitive-programming SFT |
| angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k | https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k | 2,940 | 29.4% | Opus-style synthetic reasoning conversations |
| identity.jsonl | identity.jsonl | 200 | 2% | Zero identity/persona examples |

## Files

- [sky-t1-data-17k.md](sky-t1-data-17k.md)
- [competitive-programming-python-blend.md](competitive-programming-python-blend.md)
- [claude-opus-4-6-4-7-reasoning-8-7k.md](claude-opus-4-6-4-7-reasoning-8-7k.md)
- [identity.md](identity.md)

## Script Mapping

The SFT script reads these defaults unless overridden:

```bash
export SKY_DATASET="NovaSky-AI/Sky-T1_data_17k"
export COMPETITIVE_DATASET="Jackrong/Competitive-Programming-python-blend"
export OPUS_REASONING_DATASET="angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k"
export IDENTITY_JSONL="Datasets/SFT/identity.jsonl"
export IDENTITY_ROWS=200
export TOTAL_TRAIN_ROWS=10000
```

Run `dry-run-data` before training to confirm each source loads, normalizes, and ends with an assistant message.
