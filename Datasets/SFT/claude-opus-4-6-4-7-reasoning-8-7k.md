<!--
    Reference card for angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k.
    Opus-style synthetic reasoning conversations for SFT.
    Allocated 2,940 rows (29.4% of blend).
-->
# angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k

- Link: https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k
- Loader name: `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`
- Script env var: `OPUS_REASONING_DATASET`
- Default allocation: `2,940` rows out of `10,000` total SFT rows
- Share: `29.4%`
- Format on Hugging Face: JSON, `train` split, `messages` conversations
- License on Hugging Face: Apache-2.0

## Why It Is In SFT

This source gives the blend synthetic Opus-style reasoning traces across coding, math, science, and other instruction-following categories. It is used to add polished reasoning structure and broader task coverage around the harder Sky-T1 and coding-heavy competitive-programming data.

## Schema Notes

The script expects:

- `category`
- `messages`
- `model`

Only `messages` is needed for training. The normalizer keeps system prompts when `INCLUDE_SYSTEM_PROMPTS=true`, cleans roles, and requires each example to contain at least one user turn and one assistant turn. Examples are trimmed so the final turn is an assistant response, which keeps response-only loss masking sane.

## Use Carefully

Treat this as style and reasoning demonstration data. It is not a verifier-backed RL dataset, and the synthetic source should not dominate if the target model needs stronger contest-code correctness.
