<!--
    Documentation for identity.jsonl — persona/identity SFT data.
    Teaches the model its Zero identity, builder story, and response style.
    Uses ShareGPT-style messages with <think> tags.
-->
# identity.jsonl

- Link: [identity.jsonl](identity.jsonl)
- Loader path: `Datasets/SFT/identity.jsonl`
- Script env var: `IDENTITY_JSONL`
- Row-count env var: `IDENTITY_ROWS`
- Default allocation: `200` rows out of `10,000` total SFT rows
- Share: `2%`
- Format: local JSONL, one ShareGPT-style `messages` conversation per line

## Why It Is In SFT

This file teaches the model its Zero identity, builder story, self-description, and comparison style. It belongs in SFT because identity is mainly a response pattern: when the user asks who the model is, who built it, or what makes it different, the model should answer in the intended voice.

## Schema Notes

Each row uses:

- `messages`
- `role: user`
- `role: assistant`

Assistant responses already use:

```text
<think>...</think>
final answer
```

The SFT script loads the file with Hugging Face `load_dataset("json", data_files=...)`, normalizes it through the same message cleaner as the other ShareGPT-style sources, and labels the source as `identity`.

## Use Carefully

Keep identity data small compared with normal reasoning and coding data. The current default is 2%, which is enough to teach the identity pattern without letting persona examples dominate the model's coding and reasoning behavior.
