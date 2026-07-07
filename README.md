# Owning Your Language Model

**Build, train, align, merge, and reason — your own SLM from scratch.**

Architecture → Pretraining → SFT → LoRA → Frankenmerger → GRPO

Everything implemented in PyTorch. Every weight update understood. Every gradient accounted for.

---

## The Motive

Most people treat language models as black boxes. Call API → get text. The internals are a mystery.

This project tears it open. We write the tokenizer, the attention mechanism, the training loop, the loss functions, the generation code. Then we take it further — parameter-efficient fine-tuning, model merging, online reinforcement learning.

The goal isn't a leaderboard score. The goal is **ownership**: when something works or fails, you know exactly why.

---

## Architecture — Decoder-Only Transformer

The full forward pass:

$$
\begin{aligned}
x_0 &= \text{Embed}(\text{input\_ids}) && (B, T) \to (B, T, 960)\\[4pt]
x_{\ell+1} &= x_\ell + \text{DecoderLayer}(x_\ell) && \ell = 0, \dots, 29\\[4pt]
h &= \text{RMSNorm}(x_{30})\\[4pt]
\text{logits} &= h \cdot E^\top && (B, T, 960) \to (B, T, 32768)
\end{aligned}
$$

**RMSNorm** — normalization without centering:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma
$$

**RoPE** — encode position by rotating Q/K pairs:

$$
\begin{pmatrix} Q'_{2i} \\ Q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix} \begin{pmatrix} Q_{2i} \\ Q_{2i+1} \end{pmatrix}
$$

**Multi-Head Attention** — 15 heads, head_dim=64, causal mask:

$$
\text{out} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{64}} + \text{mask}\right) V
$$

**SwiGLU MLP** — gated activation, per-token processing:

$$
\text{SwiGLU}(x) = W_{\text{down}}(\text{silu}(W_{\text{gate}} x) \odot W_{\text{up}} x)
$$

---

## Pretraining

Next-token prediction on raw text, code, and math. Data streamed from FineWeb-edu, OpenWebMath, StarCoderData, packed into 2049-token blocks.

$$
\mathcal{L}_{\text{pretrain}} = -\frac{1}{BT}\sum_{b,t} \log \pi_\theta(y_{b,t} \mid x_{b,<t})
$$

AdamW ($\beta_1=0.9$, $\beta_2=0.95$, wd=0.1), linear warmup → cosine decay. 15B tokens at 363M params.

---

## SFT — Supervised Fine-Tuning

Same cross-entropy, now on instruction-response pairs. The gradient pushes mass toward a fixed demonstrated token at every position:

$$
\mathcal{L}_{\text{SFT}} = -\sum_t \log \pi_\theta(y^*_t \mid x, y^*_{<t})
$$

SFT can only reinforce "matches this string." No mechanism to discover strategies that correlate with success.

---

## LoRA — Low-Rank Adaptation

Freeze base weights, insert trainable rank-$r$ decompositions into attention projections:

$$
h = W_0 x + BA x \qquad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}
$$

The gradient through LoRA:

$$
\frac{\partial \mathcal{L}}{\partial A} = B^\top \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^\top
$$

Fine-tune 363M models on consumer GPUs (RTX 4090) at < 1% of full parameter cost.

---

## Frankenmerger — Model Merging

Combine multiple fine-tuned models by merging in weight space. No additional training.

**Task arithmetic** — treat each fine-tuned model as a task vector $\tau_i = \theta_i - \theta_{\text{base}}$:

$$
\theta_{\text{merged}} = \theta_{\text{base}} + \sum_i \alpha_i \tau_i
$$

**TIES-Merging** resolves sign conflicts: trim low-magnitude values, elect majority sign per parameter, average only agreeing parameters.

Fine-tuned models live in the same loss basin — linear interpolation stays in the basin while combining capabilities.

---

## GRPO — Group Relative Policy Optimization

Sample $G$ completions per prompt, score with a verifier, compute group-relative advantage:

$$
A_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
$$

Clipped policy gradient with importance weighting and KL regularization:

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i,t} \min\!\left(r_t(\theta) A_i,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_i\right) - \beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}})
$$

The fixed point is a Boltzmann tilting of the reference policy by reward:

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \cdot \exp(R(y) / \beta)
$$

Trajectories with above-average reward get exponentially amplified — regardless of whether they existed in any SFT dataset. This is how emergent chain-of-thought and self-correction arise.

---

## Getting Started

```bash
git clone https://github.com/madhvantyagi/Owning-Language-model.git
cd Owning-Language-model/SLM_Architecture
pip install -r requirements.txt
python tokenizer/train_tokenizer.py
python model/slm_model.py          # verify forward pass
python train/pretrain.py           # start training
```

---

<p align="center">
  <strong>Every weight we initialize, we train.</strong><br>
  <strong>Every gradient we backpropagate, we understand.</strong>
</p>

<p align="center">
  <sub><a href="https://github.com/madhvantyagi">@madhvantyagi</a> · · 2026</sub>
</p>
