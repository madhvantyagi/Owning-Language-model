<!--
    Reference card for NovaSky-AI/Sky-T1_data_17k dataset.
    Hard math/code long-reasoning examples for SFT blend.
    Allocated 1,960 rows (19.6% of blend).
-->
# NovaSky-AI/Sky-T1_data_17k

- Link: https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k
- Loader name: `NovaSky-AI/Sky-T1_data_17k`
- Script env var: `SKY_DATASET`
- Default allocation: `1,960` rows out of `10,000` total SFT rows
- Share: `19.6%`
- Format on Hugging Face: JSON, `train` split, text conversations
- License on Hugging Face: Apache-2.0

## Why It Is In SFT

Sky-T1 is the hard reasoning anchor for the blend. It contains math, coding, science, and puzzle-style conversations with explicit long reasoning markers. In this repo, it is used to push Qwen3-8B toward step-by-step problem solving instead of only short answer imitation.

## Schema Notes

The loader expects:

- `system`
- `conversations`

The script converts `from: user` and `from: assistant` into normal chat roles. Assistant text using Sky-style thought and solution markers is rewritten into the repo format:

```text
<think>
...
</think>

final answer
```

By default, the separate Sky system prompt is not included unless `INCLUDE_SKY_SYSTEM_PROMPT=true`.

## Use Carefully

This is useful for reasoning SFT, not for reward training by itself. It gives demonstrated solutions, but it does not provide rejected answers or executable verifier signals.
