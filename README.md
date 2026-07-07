
# Owning Your Language Model

### Build, train, align, merge, and reason — your own SLM from scratch.

> **Architecture → Pretraining → SFT → LoRA → Frankenmerger → GRPO**
>
> Every line of code is ours. Every weight update is understood. Every gradient has a reason.

---

<p align="center">
  <strong>d_model</strong> = 960 &nbsp;·&nbsp;
  <strong>Layers</strong> = 30 &nbsp;·&nbsp;
  <strong>Heads</strong> = 15 &nbsp;·&nbsp;
  <strong>Head dim</strong> = 64 &nbsp;·&nbsp;
  <strong>Vocab</strong> = 32,768 &nbsp;·&nbsp;
  <strong>Params</strong> = 363M
</p>

---

## The Motive

Most people use language models like appliances. Call an API, get text back. The internal machinery is a mystery — fine for an appliance, useless if you want to *invent* the next generation.

This project tears open the black box.

We implement the entire stack in ~1500 lines of PyTorch: tokenizer, data pipeline, attention mechanism, training loop, loss functions, generation code. Then we go further — parameter-efficient fine-tuning, model merging, and online reinforcement learning.

The goal is not SOTA on a leaderboard. The goal is **ownership**: when something works or fails, you know exactly why. And when you have an idea for something novel, you can *implement* it instead of waiting for someone else to.

---

## What We Build

### 1. The Architecture — Decoder-Only Transformer (Llama-style)

The full forward pass, from token IDs to logits:

$$
\begin{aligned}
x_0 &= \text{Embed}(\text{input\_ids}) && (B, T) \to (B, T, 960)\\[4pt]
x_{\ell+1} &= x_\ell + \text{DecoderLayer}(x_\ell) && \ell = 0, 1, \dots, 29\\[4pt]
h &= \text{RMSNorm}(x_{30}) && (B, T, 960)\\[4pt]
\text{logits} &= h \cdot E^\top && (B, T, 960) \to (B, T, 32768)
\end{aligned}
$$

The embedding matrix $E \in \mathbb{R}^{32768 \times 960}$ is **tied** — it converts tokens to vectors at the input and vectors to scores at the output, cutting 31.5M parameters.

#### RMSNorm

Normalization without centering — used in Llama, Mistral, DeepSeek:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

#### RoPE — Rotary Position Embedding

Attention doesn't know token order. RoPE encodes position by *rotating* the $(2i, 2i+1)$ pairs in Q and K:

$$
\begin{pmatrix} Q'_{2i} \\ Q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix} \begin{pmatrix} Q_{2i} \\ Q_{2i+1} \end{pmatrix}
$$

where $\theta_i = 1 / 10000^{2i/64}$. Now $Q_p \cdot K_q$ encodes relative distance $(p-q)$ naturally.

#### Multi-Head Causal Self-Attention

15 heads, head_dim = 64, complete dimension trace:

$$
\begin{aligned}
&(B, T, 960) \xrightarrow{W_{qkv}} (B, T, 2880) \xrightarrow{\text{split}} 3 \times (B, T, 960)\\
&Q, K, V \xrightarrow{\text{reshape}} (B, 15, T, 64)\\
&\text{scores} = \frac{Q K^\top}{\sqrt{64}} \quad \text{shape: } (B, 15, T, T)\\
&\text{weights} = \text{CausalSoftmax}(\text{scores})\\
&\text{out} = \text{weights} \cdot V \quad \text{shape: } (B, 15, T, 64) \to (B, T, 960)
\end{aligned}
$$

#### SwiGLU MLP

Gated activation — the gate learns to *suppress* features per token:

$$
\text{SwiGLU}(x) = W_{\text{down}} \big( \underbrace{\text{silu}(W_{\text{gate}} \cdot x)}_{\text{gate}} \odot \underbrace{W_{\text{up}} \cdot x}_{\text{candidate}} \big)
$$

---

### 2. Pretraining — Learning Language from Raw Text

The model learns vocabulary, syntax, grammar, and factual knowledge entirely from next-token prediction on web text, code, and math.

Given input tokens $x = (x_1, \dots, x_T)$ and labels $y = (x_2, \dots, x_{T+1})$ (shifted by one):

$$
p_t(k) = \frac{e^{z_t(k)}}{\sum_{j=1}^{32768} e^{z_t(j)}} \qquad
\mathcal{L}_t = -\log p_t(y_t)
$$

$$
\mathcal{L}_{\text{pretrain}} = \frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} -\log \pi_\theta(y_{b,t} \mid x_{b,<t})
$$

When $p_t(y_t) = 0.28$, loss $= 1.27$. When $p_t(y_t) = 0.001$, loss $= 6.91$ — bad predictions are penalized exponentially harder.

**Optimizer:** AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\lambda = 0.1$ weight decay. Linear warmup over 2000 steps, cosine decay to 10% of max LR $= 4 \times 10^{-4}$.

**Compute:** ~15B tokens × 363M params ≈ 120 A100-hours. Honest budget. Undertrained by Big Tech standards, but the RL stage is where the interesting behavior emerges anyway.

---

### 3. SFT — Supervised Fine-Tuning

The model learns to follow instructions by imitating demonstrations. Same cross-entropy as pretraining, but applied to instruction-response pairs:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t} \log \pi_\theta(y_t^* \mid x, y_{<t}^*)
$$

$y^*$ is a fixed human- or teacher-written response. The gradient pushes probability mass **toward exactly this token at exactly this position** — the model's own opinion about whether that token was likely or hard never enters the loss.

**The limitation:** SFT can only reinforce "matches this string." It has no mechanism to discover strategies that correlate with *success* — only strategies that correlate with the demonstration dataset. This is why SFT alone cannot produce emergent chain-of-thought or self-correction.

---

### 4. LoRA — Low-Rank Adaptation

Full fine-tuning updates every parameter. LoRA freezes the base weights and inserts trainable rank-$r$ decomposition matrices into the attention projections:

$$
W_0 + \Delta W = W_0 + BA \qquad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)
$$

The forward pass through a LoRA-modified layer:

$$
h = W_0 x + \underbrace{B A x}_{\text{LoRA adaptation}}
$$

**Why this works:** Fine-tuning shifts weights in a low-dimensional subspace. LoRA constrains the search to that subspace explicitly — you're parameterizing the *direction* of the update, not the full $d \times k$ space.

**Gradient dynamics:** Only $A$ and $B$ receive gradients. The base $W_0$ is frozen. During backprop, the error signal flows through $B A$ as:

$$
\frac{\partial \mathcal{L}}{\partial A} = B^\top \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^\top \qquad
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot (A x)^\top
$$

The rank $r$ controls expressivity: $r = 8$ typically captures > 90% of full fine-tuning's benefit at < 1% of the parameters.

**In our pipeline:** LoRA lets us fine-tune the 363M model on consumer GPUs (RTX 4090 with 24 GB), enabling rapid iteration on instruction-following without full pretraining costs.

---

### 5. Frankenmerger — Model Merging

Given multiple LoRA adapters or fully fine-tuned models, we can combine them by **merging weights in parameter space**. This lets us compose capabilities without additional training.

**Linear interpolation (model soup):**

$$
\theta_{\text{merged}} = \lambda \theta_A + (1 - \lambda) \theta_B
$$

**Task arithmetic** — treat each fine-tuned model as a "task vector" $\tau_i = \theta_i - \theta_{\text{base}}$:

$$
\theta_{\text{merged}} = \theta_{\text{base}} + \sum_i \alpha_i \tau_i
$$

**TIES-Merging** resolves sign conflicts between task vectors:

1. **Trim:** Zero out the bottom-$k\%$ of values in each $\tau_i$ by magnitude
2. **Elect sign:** For each parameter, take the majority sign across task vectors
3. **Disjoint merge:** Average only the parameters whose signs agree with the elected sign

**Why merging works:** Fine-tuned models live in the same basin of the loss landscape. Their weight configurations are connected by near-zero-loss paths (the linear mode connectivity phenomenon). Averaging them stays in the basin while combining capabilities.

**In our pipeline:** We fine-tune separate LoRA adapters for instruction-following, code generation, and math reasoning, then merge them into a single model. This replaces the need for multi-task training.

---

### 6. GRPO — Group Relative Policy Optimization

The model learns to *reason correctly* by maximizing a verifier's reward signal across its own sampled completions. This is where emergent behaviors appear.

#### The Objective

For each prompt $x$, sample $G$ completions $y_1, \dots, y_G \sim \pi_\theta$. Each gets reward $R_i$ from a verifier. The **group-relative advantage**:

$$
A_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
$$

where $\mu_{\text{group}}$ and $\sigma_{\text{group}}$ are the mean and std of rewards within the group of $G$. This baseline subtraction is **unbiased** (by the score function identity $\mathbb{E}[\nabla \log \pi_\theta] = 0$) and reduces gradient variance.

The clipped policy gradient:

$$
\mathcal{L}_{\text{policy}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \sum_{t} \min\left( r_{i,t}(\theta) A_i,\; \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right)
$$

Importance weight (ratio of new to old policy):

$$
r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}
$$

KL regularization against the reference policy:

$$
\mathcal{L}_{\text{KL}}(\theta) = -\beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}})
$$

**Total loss:**

$$
\mathcal{L}_{\text{GRPO}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{KL}}
$$

#### The Closed-Form Fixed Point

The optimal policy for the regularized reward-maximization objective has a known form, derived from the KL-constrained RL problem:

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \cdot \exp\big(R(y) / \beta\big)
$$

This is a **Boltzmann tilting** of the reference distribution. Trajectories with above-average reward get exponentially amplified — regardless of whether they ever appeared in any SFT dataset.

#### Why This Creates Emergent Behavior

SFT's gradient is $\text{softmax}(h) - \text{onehot}(y^*_{\text{fixed}})$ — always pushing toward a fixed external target.

GRPO's gradient is $A_i \cdot (\text{softmax}(h) - \text{onehot}(y_{\text{self}}))$ — the sign flips when $A_i < 0$, meaning the model *punishes itself* for tokens it just generated if that completion scored below average.

| Property | SFT | GRPO |
|----------|-----|------|
| Error signal | $\nabla = \text{softmax}(h) - \mathbf{1}_{y^*}$ | $\nabla = A_i \cdot (\text{softmax}(h) - \mathbf{1}_{y_i})$ |
| Coefficient | $1$ (always positive) | $A_i$ (signed — gradient flips when $A_i < 0$) |
| Target | External fixed demonstration | Self-generated completion |
| Can suppress tokens? | No | Yes |
| Can discover novel strategies? | No | Yes |

DeepSeek-R1-Zero's emergent long chain-of-thought and self-correction came from exactly this mechanism: the model discovered that longer reasoning chains correlated with higher verifier reward, and GRPO's gradient reinforced that correlation.

---

### 7. GSPO — Gradient-aware Supervised Policy Optimization

GSPO bridges SFT and GRPO by incorporating *gradient information* from a reward signal directly into the supervised loss.

The objective is a **reward-weighted** maximum likelihood:

$$
\mathcal{L}_{\text{GSPO}} = -\sum_{t} w(y^*_t) \cdot \log \pi_\theta(y^*_t \mid x, y^*_{<t})
$$

where the weight $w(y^*_t)$ depends on the reward of the full trajectory:

$$
w(y^*) = \sigma\left( \frac{R(y^*) - b}{\tau} \right)
$$

with $\sigma$ the sigmoid, $b$ a baseline (e.g., average reward of the batch), and $\tau$ a temperature controlling how sharply high-reward trajectories are up-weighted.

**The key difference from GRPO:** GSPO does not sample from the current policy — it still uses a fixed external demonstration $y^*$ — but it reweights the loss so the model focuses more on high-reward demonstrations and less on low-reward ones. It's a bridge between pure supervised learning and online RL.

| | SFT | GSPO | GRPO |
|---|---|---|---|
| Target distribution | Fixed demonstrations | Fixed demos, reward-weighted | Self-generated, reward-tilted |
| Sampling | None | None | From current policy $\pi_\theta$ |
| Gradient direction | Always toward $y^*$ | Toward $y^*$, scaled by $w(R)$ | Signed by advantage $A_i$ |
| Compute cost | Low | Low | High ($G$ samples per prompt) |

---

## The Full Gradient Flow

Every training method in this repo — from pretraining to GRPO — produces weight updates of the same structural form:

$$
\Delta W = -\eta \sum_t \big( \mathbf{error}_t \otimes \mathbf{h}_t \big)
$$

This is an **outer product** between the hidden state $\mathbf{h}_t$ at position $t$ and the backpropagated error signal $\mathbf{error}_t$. The shape of the update is identical for every method. The entire difference lives in what $\mathbf{error}_t$ equals:

| Method | $\mathbf{error}_t$ |
|--------|-------------------|
| Pretraining / SFT | $\text{softmax}(\mathbf{h}_t) - \mathbf{1}_{y^*_t}$ |
| GSPO | $w(y^*) \cdot (\text{softmax}(\mathbf{h}_t) - \mathbf{1}_{y^*_t})$ |
| GRPO | $A_i \cdot (\text{softmax}(\mathbf{h}_t) - \mathbf{1}_{y_{i,t}})$ |
| LoRA (any method) | Same as base, but projected through $B$ before updating $A$ |

Same machinery. Completely different targets.

---

## Getting Started

```bash
# Clone and enter
git clone https://github.com/madhvantyagi/Owning-Language-model.git
cd Owning-Language-model/SLM_Architecture

# Install dependencies
pip install -r requirements.txt

# Train the tokenizer
python tokenizer/train_tokenizer.py

# Verify the build
python model/slm_model.py

# Start pretraining (15B tokens)
python train/pretrain.py
```

### Compute Requirements

| Stage | Hardware | Time | Memory |
|-------|----------|------|--------|
| Tokenizer training | CPU | ~2 hours | 8 GB |
| Pretraining (15B tokens) | A100 40GB | ~120 hours | 35 GB |
| SFT | A100 40GB / RTX 4090 | ~40 hours | 24-35 GB |
| LoRA fine-tuning | RTX 4090 | ~8 hours | 16 GB |
| Frankenmerger | CPU / any | ~minutes | 8 GB |
| GRPO | A100 40GB | ~40 hours | 35 GB |

---

## Key Results

| Quantity | Value |
|----------|-------|
| Random init loss ($-\ln(1/32768)$) | $\approx 10.40$ |
| Target pretrain loss (perplexity $\approx 100$) | $\approx 4.61$ |
| Reasonable SFT loss | $\approx 1.5$ – $2.5$ |
| LoRA rank $r$ (default) | $8$ – $64$ |
| LoRA $\alpha$ scaling | $16$ – $32$ |
| Frankenmerge $\lambda$ (linear soup) | $0.3$ – $0.7$ |
| GRPO group size $G$ | $8$ – $64$ |
| GRPO clipping $\varepsilon$ | $0.2$ |
| GRPO KL coefficient $\beta$ | $0.01$ – $0.05$ |

---

## References

- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — architecture basis
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — parameter-efficient fine-tuning
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) — model merging via task vectors
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) — sign-aware merging
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — GRPO
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — RLHF foundations
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — original reference

---

<p align="center">
  <strong>Every weight we initialize, we train.</strong><br>
  <strong>Every gradient we backpropagate, we understand.</strong><br>
  <strong>Every logit we sample, we own.</strong>
</p>

<p align="center">
  <sub>Built from scratch by <a href="https://github.com/madhvantyagi">@madhvantyagi</a> · · 2026</sub>
</p>
