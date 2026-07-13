<!--
    Reference card for Jackrong/Competitive-Programming-python-blend.
    Python competitive-programming SFT data — largest source in the blend.
    Allocated 4,900 rows (49% of blend).
-->
# Jackrong/Competitive-Programming-python-blend

- Link: https://huggingface.co/datasets/Jackrong/Competitive-Programming-python-blend
- Loader name: `Jackrong/Competitive-Programming-python-blend`
- Script env var: `COMPETITIVE_DATASET`
- Local-file override: `COMPETITIVE_JSONL`
- Default allocation: `4,900` rows out of `10,000` total SFT rows
- Share: `49%`
- Format on Hugging Face: JSON, `train` split, ShareGPT-style `messages`

## Why It Is In SFT

This is the main coding-heavy part of the blend. It gives the model many prompt-to-solution examples for Python competitive programming, so the SFT run does not become only general math reasoning.

## Schema Notes

The script expects a `messages` field with chat turns:

- `role: user`
- `role: assistant`
- optional `role: system`

Assistant turns are normalized into the same training style used by the rest of the SFT mix. If a response already has a `<think>` block, the script keeps it. If not, it prefixes the answer with an empty thinking block:

```text
<think></think>
answer text
```

## Local JSONL Caveat

`COMPETITIVE_JSONL` can point to one or more local JSONL files, separated by commas. Do not point it at Git LFS pointer files. The script checks for pointer files and stops instead of silently training on invalid data.

## Use Carefully

This is SFT data for code-writing behavior. If the goal is RL or GRPO, you still need executable tests, answer checkers, or another reward signal.
