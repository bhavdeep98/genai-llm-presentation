# Section 4: Fine-Tuning — Making Models Your Own — Deep Dive Research Document

## Purpose
This document expands Section 4 (Fine-Tuning / LoRA) into research-paper-level depth. It covers the problem with full fine-tuning, the LoRA insight, QLoRA for democratization, the LoRA family tree, and practical decision-making. Your research on data quality, benchmark creation, and instruction-tuning efficiency connects to the critical question: what data do you fine-tune ON?

---

## Timeline: Fine-Tuning Milestones + Your Research

| Year | Milestone | Authors | Key Insight |
|------|-----------|---------|-------------|
| 2019 | Adapters | Houlsby et al. | Add small trainable layers (has inference latency) |
| 2021 | Intrinsic Dimensionality (ACL 2021) | Aghajanyan et al. | Pre-trained models live in low-dim subspace |
| 2021 | LoRA | Hu et al. (Microsoft) | Low-rank weight updates, zero inference latency |
| 2021 | Prefix Tuning | Li & Liang | Prepend learnable tokens |
| 2021 | **Natural Instructions** | **Swaroop Mishra** et al. | What to fine-tune ON matters more than how |
| 2022 | **Super-NaturalInstructions** | **Mishra**, Wang et al. | 1,616 tasks — scaling fine-tuning data diversity |
| 2022 | **"Is High Quality Data All We Need?"** | **Sachdeva, Mishra** et al. | Quality > quantity for fine-tuning |
| 2022 | **Generalized but not Robust?** | **Sachdeva**, Mishra et al. | Data modification effects on generalization |
| 2023 | QLoRA | Dettmers et al. (UW) | 65B model on single 48GB GPU |
| 2023 | AdaLoRA | Zhang et al. | Adaptive rank allocation per layer |
| 2023 | **VAIDA** | Arunkumar, **Mishra, Sachdeva** et al. | Better benchmark data → better fine-tuning |
| 2023 | **Self-Instruct** | Wang, **Mishra** et al. | LLMs generate their own fine-tuning data |
| 2023 | **TarGEN** | Gupta, **Mishra** et al. | Seedless synthetic fine-tuning data |
| 2024 | DoRA | Liu et al. | Magnitude + direction decomposition |
| 2024 | LoRA-XS | Multiple | SVD-initialized, 100x fewer params |
| 2024 | LoRA+ | Hayou et al. | Different learning rates for A and B |

---

## Slide 19: Why Fine-Tune?

### Core Concept
Pre-trained models know language; fine-tuning teaches them YOUR specific task, domain, format, or behavior. It bridges the gap between general capability and specific usefulness.

### The Fine-Tuning Spectrum

```
More parameters modified ←────────────────────────────────→ Fewer parameters modified
More control/capability  ←────────────────────────────────→ Less control, cheaper

┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Full          │ │ LoRA         │ │ QLoRA        │ │ Prefix       │ │ Prompt       │
│ Fine-Tuning   │ │              │ │              │ │ Tuning       │ │ Engineering  │
│               │ │              │ │              │ │              │ │              │
│ 100% params   │ │ 0.01-0.1%   │ │ 0.01-0.1%   │ │ 0.001%       │ │ 0% params    │
│ Full control  │ │ Near-full    │ │ Near-full    │ │ Limited      │ │ Limited      │
│ Max memory    │ │ Less memory  │ │ Min memory   │ │ Min memory   │ │ No training  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### What Fine-Tuning Teaches a Model

| What You Want | How Fine-Tuning Achieves It | Example |
|---------------|---------------------------|---------|
| New format | Learn output structure | JSON responses, markdown tables |
| Domain knowledge | Encode domain-specific facts | Medical terminology, legal language |
| Style/tone | Adjust generation distribution | Formal vs. casual, concise vs. detailed |
| New capabilities | Learn new task patterns | Code review, SQL generation |
| Safety behaviors | Learn to refuse/comply | Following content policies |


### Connection to Instruction-Tuning (Your Mentor's Work)

> **Research connection:** Swaroop Mishra's research demonstrates that WHAT you fine-tune on matters far more than HOW you fine-tune:
>
> - **Instruction-Example Equivalence (2023):** 1 well-written instruction = ~200 training examples. This means the quality of your fine-tuning DATA is the dominant factor, not the fine-tuning method.
> - **Super-NaturalInstructions (2022):** Tk-Instruct (11B, standard fine-tuning on diverse instructions) outperformed GPT-3 (175B, few-shot) on cross-task generalization. The DATA made the difference.
> - **TarGEN (2023):** When you don't have enough high-quality fine-tuning data, you can GENERATE it synthetically using LLMs — seedless, scalable, and effective (+1-2% over original data).
>
> **Practical implication:** Before optimizing your fine-tuning METHOD (LoRA rank, learning rate, etc.), first optimize your fine-tuning DATA (instructions, diversity, quality).
>
> [Paper: Mishra et al., 2023, "Instruction Tuned Models are Quick Learners," arXiv:2306.05539]
> [Paper: Gupta, Mishra et al., 2023, "TarGEN," arXiv:2310.17876]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Fine-tuning spectrum] | Create custom | Spectrum from full FT to prompt engineering |
| ![Data vs method importance] | Create custom | Pie chart: what matters most in fine-tuning |

- **[DIAGRAM TO CREATE]** The spectrum above as a horizontal bar with examples at each point
- **[DIAGRAM TO CREATE]** "What matters most?" — Data quality (60%) > Data diversity (25%) > Method choice (15%)

---

## Slide 20: The Problem with Full Fine-Tuning

### Core Concept
Full fine-tuning updates every parameter. For modern LLMs, this is prohibitively expensive in memory, storage, and deployment complexity.

### The Math of Full Fine-Tuning

**GPT-3 (175B parameters):**
```
Model weights:              175B × 2 bytes (FP16) = 350 GB
Gradients:                  175B × 2 bytes        = 350 GB
Optimizer states (Adam):    175B × 8 bytes        = 1,400 GB
                                                    ─────────
Total training memory:                              ~2,100 GB (2.1 TB)
```

That's ~26 A100 GPUs (80GB each) just to HOLD the training state.

**Storage per variant:**
- Each fine-tuned model = 350GB
- 10 tasks = 10 copies = 3.5 TB
- 100 customers = 100 copies = 35 TB
- Completely impractical for multi-tenant deployment

**Llama 3 (405B parameters):**
```
Model weights:              405B × 2 bytes = 810 GB
Training memory (estimated): ~5-6 TB total
Cost: 16K+ H100 GPUs for weeks
```

### Why This Matters Practically

| Scenario | Full FT Cost | Reality Check |
|----------|-------------|--------------|
| Startup with one A100 (80GB) | Can't fine-tune anything >7B | Most models are 7B+ |
| Enterprise with 8×A100 | Can maybe fine-tune 70B | GPT-4 class impossible |
| Multi-task deployment | N copies of full model | Storage/serving explodes |
| Rapid iteration | Days per experiment | Too slow for R&D |

### The Question That LoRA Answers

> "What if the update ΔW that fine-tuning learns doesn't NEED all d×d dimensions?
> What if it lives in a much smaller subspace?"

This is the **intrinsic dimensionality hypothesis** (Aghajanyan et al., 2021, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning," ACL 2021, arXiv:2012.13255): pre-trained models, when adapted to new tasks, move through a surprisingly low-dimensional subspace of parameter space.


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Memory breakdown] | Create custom | Stacked bar: weights + gradients + optimizer |
| ![Storage explosion] | Create custom | 10 tasks × 350GB visual |
| ![GPU requirement chart] | Create custom | Model size vs. minimum GPUs needed |

- **[DIAGRAM TO CREATE]** Memory breakdown pie chart for GPT-3 full fine-tuning (weights 17%, gradients 17%, optimizer 66%)
- **[DIAGRAM TO CREATE]** "The deployment problem" — 10 customer variants × 350GB = impractical

---

## Slide 21: LoRA — The Key Insight

### Core Concept
Pre-trained weight updates have LOW intrinsic rank. Instead of updating the full d×d weight matrix, decompose the update into two small matrices: ΔW = B × A, where the rank r is tiny (4-16) compared to d (4096-12288).

### The Mathematical Foundation

**Full fine-tuning:**
```
W' = W + ΔW        (ΔW ∈ R^(d×k), all d×k parameters learned)
```

**LoRA:**
```
W' = W + B·A        (B ∈ R^(d×r), A ∈ R^(r×k), only r×(d+k) parameters learned)
```

**The parameter savings:**
```
Full:  d × k = 4096 × 4096 = 16,777,216 parameters per layer
LoRA:  r × (d + k) = 8 × (4096 + 4096) = 65,536 parameters per layer
Reduction: 256x fewer parameters (for r=8)
```

For GPT-3 (175B params total):
```
Full fine-tuning:   175,000,000,000 trainable parameters
LoRA (r=4):              4,700,000 trainable parameters
Reduction:              ~37,000x fewer parameters
```

### Why Low Rank Works — Intuition

**Analogy:** Imagine you've learned to speak English (pre-training). Now you want to learn to write formal emails (fine-tuning). You don't need to relearn all of English — you need a small adjustment to your writing style. That adjustment lives in a low-dimensional space: "more formal vocabulary," "shorter sentences," "professional tone." A few directions of change, not millions.

**The evidence (Aghajanyan et al., 2021, ACL):**
- Measured intrinsic dimensionality of fine-tuning across tasks
- Found that most tasks can be solved by moving in a subspace of ~200-800 dimensions
- Even for a model with millions of parameters, the USEFUL update is low-rank

### Step-by-Step: How LoRA Works in Practice

```
Forward pass (during training):
  Input x
    ├── Through frozen W:     h₁ = Wx
    └── Through LoRA path:    h₂ = B(Ax)     ← only A,B have gradients
  Output = h₁ + α·h₂

Forward pass (during inference — after merging):
  W_merged = W + (α/r)·BA
  Output = W_merged · x                       ← zero extra latency!
```

**Key detail — scaling factor α:**
- α controls the magnitude of the LoRA update
- Typically α = rank (so α/r = 1)
- Higher α = larger updates = faster adaptation but risk instability


### Where to Apply LoRA (Layer Selection)

| Configuration | Layers | Performance | Common Usage |
|--------------|--------|-------------|-------------|
| Q, V only | Attention Q and V projections | Good | Original paper default |
| Q, K, V, O | All attention projections | Better | Current recommendation |
| All linear | Attention + FFN up/down/gate | Best | Modern best practice |
| Selective | Only layers 10-50 (middle) | Varies | Experimental |

**Current best practice (2025-2026):** Apply LoRA to ALL linear layers (attention Q, K, V, O + FFN layers) with rank 8-16. This gives the best performance-efficiency tradeoff.

### LoRA Results (from the paper)

| Task | Full FT (175B) | LoRA r=4 (4.7M) | LoRA r=8 | Gap |
|------|---------------|-----------------|----------|-----|
| WikiSQL | 73.8 | 73.4 | 73.8 | <0.5% |
| MNLI | 89.7 | 89.4 | 89.7 | <0.3% |
| SAMSum | 52.0 | 51.8 | 52.0 | <0.2% |
| Average | — | — | — | **<0.3% gap** |

**The remarkable finding:** 10,000x fewer parameters achieve within 0.3% of full fine-tuning performance.

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![LoRA mechanism] | Hu et al. 2021, Figure 1 | Frozen W with parallel low-rank BA path |
| ![Rank comparison] | Create custom | Performance vs. rank (r=1,2,4,8,16,64) |
| ![Parameter savings] | Create custom | Bar chart: Full FT vs. LoRA parameter counts |
| ![Merge at inference] | Create custom | Training (parallel path) vs. Inference (merged) |

- **[SOURCE: Hu et al., 2021, "LoRA," arXiv:2106.09685, Figure 1]** — The canonical LoRA diagram
- **[SOURCE: Aghajanyan et al., 2021, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning," ACL 2021, arXiv:2012.13255]** — Theoretical foundation
- **[DIAGRAM TO CREATE]** Two-panel: (1) Training: W frozen + BA path, (2) Inference: W+BA merged into one matrix

---

## Slide 22: LoRA — How It Works (Deeper)

### Initialization and Training Details

**Initialization:**
```
A: Random Gaussian (small values)    ← creates diversity in starting directions
B: Zeros                             ← ensures ΔW = BA = 0 at start
```

**Why B=0?** The model starts exactly where pre-training left off. The LoRA path contributes nothing initially, then gradually learns the task-specific adjustment. This is critical for stable training.

**Training hyperparameters (practical guide):**
| Hyperparameter | Typical Value | Effect |
|---------------|---------------|--------|
| Rank (r) | 4-16 | Higher = more expressive, more params |
| Alpha (α) | Same as rank | Scales the LoRA contribution |
| Learning rate | 1e-4 to 3e-4 | Higher than full FT (small params need bigger steps) |
| Dropout | 0.05-0.1 | Prevents overfitting the small adapter |
| Target modules | All linear | Modern default |
| Batch size | 4-16 (with grad accumulation) | Larger helps stability |

### Task Switching and Multi-Tenant Deployment

```
Base Model (7B, ~14GB):          Shared across all customers
  + Customer A LoRA (~20MB):     Legal domain, formal tone
  + Customer B LoRA (~20MB):     Medical Q&A, conservative answers
  + Customer C LoRA (~20MB):     Creative writing, playful tone

Total storage: 14GB + 60MB = 14.06GB
vs. Full FT: 14GB × 3 = 42GB
```

**Scaling this:**
- 1000 customers × 20MB LoRA = 20GB additional storage
- 1000 customers × 14GB full FT = 14TB additional storage
- **700x storage reduction** at scale


### Connection to Data Quality (Your Research)

> **Research connection:** LoRA makes fine-tuning cheap and fast — but this amplifies the importance of DATA QUALITY. When you can fine-tune in hours on a single GPU, the bottleneck shifts entirely to the data:
>
> **"Is High Quality Data All We Need?" (Sachdeva, Mishra et al., 2022):** This question is even more relevant with LoRA. A LoRA adapter trained on 1000 high-quality examples can outperform one trained on 10,000 noisy examples — because the low-rank subspace captures the DATA's patterns, good or bad.
>
> **VAIDA (2023):** When creating fine-tuning datasets, the 45.8% artifact reduction from VAIDA's visual feedback directly translates to better LoRA adapters. With LoRA, you iterate fast — but you iterate on DATA, not architecture.
>
> **TarGEN (2023):** When you can't get enough real fine-tuning data, TarGEN generates synthetic data that performs 1-2% BETTER than original data. Combined with LoRA: synthetic data + LoRA = cheap, fast, effective fine-tuning.
>
> **"Generalized but not Robust?" (Sachdeva, Mishra et al., ACL 2022):** Found that data modification methods (augmentation, adversarial filtering) have different effects on out-of-distribution generalization vs. adversarial robustness. This is critical for LoRA practitioners: your adapter will be exactly as robust as your training data.
>
> [Paper: Sachdeva, Mishra et al., 2022, arXiv:2203.06404]
> [Paper: Arunkumar, Mishra, Sachdeva et al., 2023, "VAIDA," EACL]
> [Paper: Gupta, Mishra et al., 2023, "TarGEN," arXiv:2310.17876]
> [Paper: Sachdeva, Mishra et al., 2022, "Generalized but not Robust?" ACL Findings]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Task switching] | Create custom | Base model + multiple small adapters |
| ![Storage comparison] | Create custom | 1000 LoRAs (20GB) vs. 1000 full models (14TB) |
| ![Training flow] | Create custom | Data → LoRA training → Merge → Deploy |

- **[DIAGRAM TO CREATE]** Multi-tenant: one base model serving 1000 customers via LoRA adapters
- **[DIAGRAM TO CREATE]** The "fast iteration loop": data → LoRA → eval → improve data → repeat

---

## Slide 23: QLoRA — Democratizing Fine-Tuning

### Core Concept
QLoRA combines 4-bit quantization of the base model with LoRA adapters in 16-bit. This enables fine-tuning models that were previously inaccessible: a 65B model on a single 48GB GPU.

### The Three Innovations

**1. NormalFloat4 (NF4) — A Better 4-bit Format**

Pre-trained weights are approximately normally distributed. NF4 exploits this:
- Standard 4-bit integers: equal spacing → wasteful for bell curves
- NF4: spacing matches normal distribution → optimal information preservation
- Each bin captures equal probability mass of the expected distribution
- Result: information-theoretically optimal for normally-distributed data

```
Standard 4-bit:  [-8, -7, -6, ..., 6, 7]     ← uniform spacing
NF4:             [-1.0, -0.8, -0.5, ..., 0.5, 0.8, 1.0]  ← matches Gaussian
```

**2. Double Quantization**

Problem: 4-bit quantization uses per-block scale factors (32-bit floats per 64 weights).
Solution: Quantize the scale factors THEMSELVES to 8-bit.
- Saves ~0.37 bits per parameter
- 3GB savings for a 65B model

**3. Paged Optimizers**

Problem: GPU memory spikes during training (gradient computation, activation checkpoints).
Solution: Automatically page optimizer states to CPU RAM when GPU is full.
- Uses NVIDIA unified memory
- Prevents OOM crashes without manual memory management


### QLoRA Memory Breakdown

| Model | Full FT (FP16) | LoRA (FP16 base) | QLoRA (4-bit base) | Consumer GPU? |
|-------|---------------|------------------|-------------------|---------------|
| 7B | ~56GB | ~18GB | **~6GB** | ✅ RTX 3090 (24GB) |
| 13B | ~104GB | ~34GB | **~10GB** | ✅ RTX 4090 (24GB) |
| 33B | ~264GB | ~86GB | **~22GB** | ✅ A6000 (48GB) |
| 65B | ~520GB | ~170GB | **~42GB** | ✅ A100 (80GB) |
| 70B | ~560GB | ~180GB | **~48GB** | ⚠️ A100 (80GB, tight) |

### Guanaco: QLoRA's Flagship Result

- Base model: LLaMA 65B (4-bit quantized)
- Fine-tuning: LoRA adapters, 24 hours on a single A100
- Result: **99.3% of ChatGPT's performance** on Vicuna benchmark
- Total cost: ~$50 in GPU time vs. millions for training ChatGPT

### The Democratization Story

```
Before QLoRA (2022):                    After QLoRA (2023):
┌────────────────────────────┐         ┌────────────────────────────┐
│ Fine-tune 65B model:       │         │ Fine-tune 65B model:       │
│ • 8× A100 GPUs ($200K+)   │         │ • 1× A100 GPU ($15K)      │
│ • Multi-node cluster       │         │ • Single machine           │
│ • Enterprise only          │         │ • Graduate student budget   │
│ • Days of training         │         │ • 24 hours                 │
└────────────────────────────┘         └────────────────────────────┘
```

### Connection to Accessibility & Multilingual Research (Your Work)

> **Research connection:** QLoRA's democratization aligns with the broader theme of making AI accessible — directly connecting to Bhavdeep Sachdeva's work as a **Language Ambassador for Panjabi at Cohere AI (Project Aya)**:
>
> - Project Aya focused on bringing LLMs to underrepresented languages
> - QLoRA makes this economically feasible: fine-tune a multilingual model for Panjabi on a single consumer GPU
> - Before QLoRA: only well-funded labs could adapt models to low-resource languages
> - After QLoRA: any researcher with a single GPU can create a language-specific adapter
>
> The combination of **TarGEN** (synthetic data for low-resource scenarios) + **QLoRA** (cheap fine-tuning) is particularly powerful for languages like Panjabi where labeled data is scarce.
>
> [Experience: Bhavdeep Sachdeva, Language Ambassador for Panjabi, Cohere AI Project Aya, 2023]
> [Paper: Gupta, Mishra et al., 2023, "TarGEN," arXiv:2310.17876]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Memory comparison chart] | Create custom | Bar chart: Full FT vs. LoRA vs. QLoRA |
| ![NF4 vs standard 4-bit] | Dettmers et al. 2023 | Distribution-aware quantization bins |
| ![Democratization timeline] | Create custom | Before/after QLoRA accessibility |
| ![Guanaco vs ChatGPT] | Dettmers et al. 2023 | 99.3% performance at fraction of cost |

- **[SOURCE: Dettmers et al., 2023, "QLoRA," arXiv:2305.14314, Figure 1]** — Memory comparison
- **[SOURCE: Dettmers et al., 2023, Table 1]** — Guanaco benchmark results
- **[DIAGRAM TO CREATE]** Memory waterfall: Full model → LoRA savings → QLoRA savings (stacked bars)

---

## Slide 24: The LoRA Family & Decision Guide

### Core Concept
LoRA spawned an entire family of variants, each addressing different limitations. Understanding which to use when is a practical skill for anyone building with LLMs.

### The LoRA Family Tree (Expanded)

```
Full Fine-Tuning (100% params, maximum control)
    │
    ├── Adapters (Houlsby 2019) — adds bottleneck layers, inference latency penalty
    │
    ├── Prefix Tuning (Li & Liang 2021) — learnable prefix tokens, limited capacity
    │
    ├── LoRA (Hu et al. 2021) — low-rank updates, zero inference latency
    │       │
    │       ├── QLoRA (Dettmers 2023) — 4-bit quantized base + FP16 LoRA
    │       │       └── GGML/GPTQ + LoRA — various quantization + LoRA combos
    │       │
    │       ├── AdaLoRA (Zhang 2023) — adaptive rank per layer (more rank where needed)
    │       │
    │       ├── LoRA-XS (2024) — SVD-frozen outer matrices, tiny trainable core (100x less)
    │       │
    │       ├── DoRA (Liu 2024) — decompose into magnitude + direction
    │       │       │              direction updated via LoRA; magnitude separately
    │       │       └── Closer to full FT behavior with LoRA efficiency
    │       │
    │       ├── LoRA+ (Hayou 2024) — different learning rates for A and B matrices
    │       │
    │       ├── rsLoRA (2024) — rank-stabilized scaling for high ranks
    │       │
    │       └── GLoRA, VeRA, LoRA-FA, etc. — various specialized variants
    │
    └── (IA)³ (2022) — learned rescaling of activations, even fewer params than LoRA
```

### DoRA: The Current Best Variant (2024)

**Insight:** Full fine-tuning decomposes weight updates into MAGNITUDE and DIRECTION changes. Standard LoRA only captures the direction component well.

**How DoRA works:**
```
W = m · (V / ||V||)    where m = magnitude, V/||V|| = direction

LoRA updates:    W' = W + BA         (conflates magnitude and direction)
DoRA updates:    W' = m' · (V + BA) / ||V + BA||   (separates them)
```

**Results (DoRA vs. LoRA on LLaMA 7B):**
| Task | LoRA | DoRA | Improvement |
|------|------|------|-------------|
| Commonsense reasoning | 78.2 | 79.4 | +1.2% |
| Visual instruction tuning | 82.1 | 83.0 | +0.9% |
| Average across tasks | — | — | +0.5-1.5% |

### Practical Decision Matrix

| Your Situation | Recommended Method | Why |
|---------------|-------------------|-----|
| Single GPU, <24GB | QLoRA (4-bit) | Only option that fits |
| Single GPU, 48-80GB | LoRA or DoRA (FP16 base) | Full quality, fast iteration |
| Multiple GPUs available | LoRA with larger rank (32-64) | Best quality per dollar |
| Maximum quality needed | DoRA on all linear layers | Closest to full FT |
| Extreme parameter budget | LoRA-XS | 100x fewer than LoRA |
| Need to serve 100+ variants | LoRA (easily swappable) | Storage/serving efficiency |
| Model >100B params | QLoRA (absolutely necessary) | Memory is the constraint |

### Rank Selection Guide

| Rank (r) | Trainable Params (7B model) | Performance | Use When |
|----------|---------------------------|-------------|----------|
| 4 | ~1.6M | 95-97% of full FT | Quick experiments, many tasks |
| 8 | ~3.3M | 97-99% of full FT | Standard production |
| 16 | ~6.6M | 98-99.5% of full FT | High-quality production |
| 32 | ~13.2M | 99-99.8% of full FT | Complex tasks, large data |
| 64 | ~26.4M | ~99.9% of full FT | Diminishing returns territory |
| 128+ | ~53M+ | ~full FT | Rarely justified |


### Connection to Robustness Research (Your Work)

> **Research connection:** The choice of fine-tuning method interacts with robustness in ways Bhavdeep Sachdeva and Swaroop Mishra's research illuminates:
>
> **"Pretrained Transformers Do Not Always Improve Robustness" (Sachdeva, Mishra et al., 2023):**
> - Found that pre-trained transformers can be LESS robust than traditional models on noisy data
> - Augmented with adversarial filtering (AF), robustness improves
> - Implication for LoRA: a LoRA adapter trained on clean data may not generalize to noisy real-world inputs
> - Practitioners should evaluate LoRA adapters on adversarial/OOD data, not just in-distribution test sets
>
> **"Generalized but not Robust?" (ACL 2022):**
> - Data modification methods (backtranslation, adversarial examples, etc.) improve OOD generalization
> - But the same methods can HURT adversarial robustness
> - For LoRA fine-tuning: you need to choose between "works well on new domains" and "works well under attack" — they may require different data strategies
>
> **Practical recommendation:** Fine-tune LoRA adapters on BOTH clean data AND adversarial/noisy examples. The low-rank subspace is small — if it only encodes clean patterns, it fails on real-world noise.
>
> [Paper: Sachdeva, Mishra et al., 2023, "Pretrained Transformers Do Not Always Improve Robustness," arXiv:2210.07663]
> [Paper: Sachdeva, Mishra et al., 2022, "Generalized but not Robust?" ACL Findings, 2022]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![LoRA family tree] | Create custom | Full tree diagram with years |
| ![Decision matrix] | Create custom | Flow chart: "what should I use?" |
| ![Rank vs performance] | Create custom | Curve showing diminishing returns |
| ![DoRA vs LoRA] | Liu et al. 2024 | Magnitude/direction decomposition |

- **[SOURCE: Liu et al., 2024, "DoRA: Weight-Decomposed Low-Rank Adaptation," arXiv:2402.09353, ICML 2024]**
- **[SOURCE: Hu et al., 2021, Figure 1]** — Original LoRA diagram (still the canonical reference)
- **[DIAGRAM TO CREATE]** Decision flowchart: GPU memory → model size → quality needs → recommended method
- **[DIAGRAM TO CREATE]** Rank vs. performance curve (log-scale x, showing diminishing returns)

---

## Summary: Key Numbers for Section 4

### LoRA Efficiency
- Parameter reduction: 10,000-37,000x fewer trainable params
- Storage per adapter: ~4-50MB (vs. 14-810GB for full model)
- Performance gap: <0.3% vs. full fine-tuning (for appropriate rank)
- Inference latency added: **zero** (merged weights)

### QLoRA Memory Savings
- 65B model: 520GB → 42GB (12x reduction)
- 7B model: 56GB → 6GB (9x reduction)
- Guanaco 65B: 99.3% of ChatGPT quality, trained in 24h on 1 GPU

### Training Costs (2025-2026 estimates)
| Model | Method | GPU(s) | Time | Approx Cost |
|-------|--------|--------|------|-------------|
| 7B | QLoRA | 1× RTX 4090 | 2-4 hours | ~$5 |
| 7B | LoRA | 1× A100 | 1-2 hours | ~$10 |
| 70B | QLoRA | 1× A100 80GB | 12-24 hours | ~$50-100 |
| 70B | LoRA | 4× A100 | 4-8 hours | ~$100-200 |
| 405B | QLoRA | 4× A100 | 24-48 hours | ~$200-500 |

### Data vs. Method (from your research)
- 1 instruction ≈ 200 data examples (Mishra, 2023)
- TarGEN synthetic data: +1-2% over original data (Mishra, 2023)
- VAIDA artifact reduction: 45.8% fewer artifacts (Mishra, Sachdeva, 2023)
- Robustness gap: pre-trained models can be LESS robust than BoW on noisy data (Sachdeva, 2023)

---

## Visual Index: All Images for Section 4

### From Original Papers

| # | Visual | Paper | Figure | Slide |
|---|--------|-------|--------|-------|
| 1 | LoRA mechanism (W + BA) | Hu et al. 2021 | Fig 1 | Slide 21 |
| 2 | QLoRA memory comparison | Dettmers et al. 2023 | Fig 1 | Slide 23 |
| 3 | NF4 quantization | Dettmers et al. 2023 | Fig 2 | Slide 23 |
| 4 | DoRA decomposition | Liu et al. 2024 | Fig 1 | Slide 24 |
| 5 | Intrinsic dimensionality | Aghajanyan et al. 2021 | Fig 1 | Slide 21 |

### Diagrams to Create

| # | Visual | Description | Slide | Tool |
|---|--------|-------------|-------|------|
| 1 | Fine-tuning spectrum | Full FT → LoRA → QLoRA → Prompt Engineering | Slide 19 | Excalidraw |
| 2 | Memory breakdown | Weights + Gradients + Optimizer stacked bar | Slide 20 | Python |
| 3 | Parameter savings | 175B vs. 4.7M side-by-side bars | Slide 21 | Python |
| 4 | Training vs inference | Parallel path (train) → Merged (infer) | Slide 21 | Excalidraw |
| 5 | Multi-tenant deployment | 1 base model + N adapters | Slide 22 | Excalidraw |
| 6 | Memory waterfall | Full → LoRA → QLoRA progressive savings | Slide 23 | Python |
| 7 | Before/after QLoRA | Enterprise cluster vs. single GPU | Slide 23 | PowerPoint |
| 8 | LoRA family tree | Full tree with years and relationships | Slide 24 | draw.io |
| 9 | Decision flowchart | Memory → Size → Quality → Recommendation | Slide 24 | Excalidraw |
| 10 | Rank vs. performance | Diminishing returns curve | Slide 24 | Python |
| 11 | Data quality matters | "What matters most in fine-tuning?" pie | Slide 19 | PowerPoint |

---

## Hype vs. Reality for Section 4

### What's Real (Evidence-Backed)
- ✅ LoRA achieves <0.3% gap from full fine-tuning for most tasks (verified)
- ✅ QLoRA enables 65B fine-tuning on a single GPU (demonstrated)
- ✅ Merged LoRA weights add zero inference latency (mathematically guaranteed)
- ✅ Task switching via adapter swapping works in production
- ✅ Low-rank hypothesis is empirically validated across model sizes
- ✅ DoRA improves over LoRA by 0.5-1.5% on average (ICML 2024)

### What's Overstated
- ⚠️ "LoRA is always as good as full fine-tuning" — for complex creative tasks, full FT can still win
- ⚠️ "r=4 is always enough" — depends on task complexity; some need r=16-32
- ⚠️ "Fine-tuning fixes hallucination" — it does NOT; it changes style, not reliability
- ⚠️ "QLoRA has zero quality loss" — there IS a small gap (~0.5-1%) in some benchmarks
- ⚠️ "LoRA adapters compose linearly" — merging multiple LoRAs is non-trivial
- ⚠️ "More data always helps" — quality > quantity (Sachdeva, Mishra 2022)

### What's Hype
- ❌ "Fine-tune your way to GPT-4" — fine-tuning adjusts behavior, not fundamental capability
- ❌ "Anyone can build a custom ChatGPT in an afternoon" — data curation takes weeks
- ❌ "Open-source fine-tuned models match proprietary" — for specific tasks yes, generally no
- ❌ "LoRA solves all deployment challenges" — serving infrastructure still matters

---

## Appendix: Full Citation List for Section 4

1. Houlsby et al., 2019, "Parameter-Efficient Transfer Learning for NLP," ICML
2. Aghajanyan et al., 2021, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning," ACL 2021, arXiv:2012.13255
3. Li & Liang, 2021, "Prefix-Tuning," arXiv:2101.00190
4. Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685 (Microsoft)
5. **Mishra et al., 2021, "Cross-Task Generalization via Natural Language Crowdsourcing Instructions," arXiv:2104.08773**
6. **Wang, Mishra et al., 2022, "Super-NaturalInstructions," EMNLP 2022, arXiv:2204.07705**
7. **Sachdeva, Mishra et al., 2022, "Is High Quality Data All We Need?" arXiv:2203.06404**
8. **Sachdeva, Mishra et al., 2022, "Generalized but not Robust?" ACL Findings 2022**
9. Zhang et al., 2023, "AdaLoRA: Adaptive Budget Allocation for PEFT," arXiv:2303.10512
10. Dettmers et al., 2023, "QLoRA: Efficient Finetuning of Quantized LLMs," arXiv:2305.14314
11. **Arunkumar, Mishra, Sachdeva et al., 2023, "VAIDA," EACL 2023, arXiv:2302.04434**
12. **Wang, Mishra et al., 2023, "Self-Instruct," ACL 2023, arXiv:2212.10560**
13. **Gupta, Mishra et al., 2023, "TarGEN," arXiv:2310.17876**
14. **Sachdeva, Mishra et al., 2023, "Pretrained Transformers Do Not Always Improve Robustness," arXiv:2210.07663**
15. **Mishra et al., 2023, "Instruction Tuned Models are Quick Learners," arXiv:2306.05539**
16. Liu et al., 2024, "DoRA: Weight-Decomposed Low-Rank Adaptation," ICML 2024, arXiv:2402.09353
17. LoRA-XS, 2024, arXiv:2405.17604
18. Hayou et al., 2024, "LoRA+: Efficient Low Rank Adaptation of Large Models," arXiv:2402.12354

---

## Key Narrative Arc for Section 4

**The story to tell:**

1. **The problem (Slide 19-20):** Full fine-tuning is too expensive for everyone except mega-labs. 175B params × storage/compute requirements = impossible for most.

2. **The insight (Slide 21-22):** Fine-tuning updates live in a LOW-RANK subspace. You don't need to update all 175B parameters — 4.7M is enough. This is mathematically elegant and practically transformative.

3. **The democratization (Slide 23):** QLoRA combines quantization + LoRA to bring 65B fine-tuning to a single GPU. $50 of compute vs. millions. Graduate students can now do what only OpenAI could in 2020.

4. **The practical guide (Slide 24):** A family of methods exists — choose based on your constraints. DoRA is best quality; QLoRA is most accessible; LoRA is the workhorse.

5. **The real bottleneck (throughout):** With LoRA making fine-tuning cheap and fast, the DOMINANT factor is now **data quality** — exactly what your research addresses. TarGEN for generating it, VAIDA for quality-checking it, and robustness research for stress-testing it.

---

*Document created: July 2, 2026*
*Last updated: July 2, 2026*
