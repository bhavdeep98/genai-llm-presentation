# LoRA & Variants — Summary

---

## Paper 1: LoRA — Low-Rank Adaptation (Hu et al., 2021)

**Paper:** "LoRA: Low-Rank Adaptation of Large Language Models"  
**ArXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)  
**Authors:** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (Microsoft)  

### The Problem
- Full fine-tuning updates ALL parameters of a large model
- For GPT-3 (175B): each fine-tuned version needs ~350GB of storage
- Cannot serve multiple fine-tuned variants efficiently
- Training requires enormous GPU memory (full gradient computation)

### The Key Insight
**Pre-trained weight matrices have low "intrinsic rank" for downstream tasks.**

When you fine-tune a model, the weight update ΔW has a much lower rank than the full matrix. So instead of updating the full d×d matrix, decompose the update into two smaller matrices:

```
W' = W + ΔW = W + BA

Where: 
  W ∈ R^(d×k)     — frozen pre-trained weights
  B ∈ R^(d×r)     — trainable (initialized to zero)
  A ∈ R^(r×k)     — trainable (initialized randomly from Gaussian)
  r << min(d, k)  — the rank (typically r = 4, 8, or 16)
```

### How It Works Step-by-Step
1. Freeze all pre-trained model weights
2. For each target weight matrix (typically attention Q, K, V, O projections):
   - Add parallel path: input → A (down-project to rank r) → B (up-project back)
3. During training, only update A and B (tiny compared to W)
4. During inference: merge W + BA → single matrix (zero added latency!)
5. To switch tasks: swap out the A, B matrices (small files)

### Why Rank r Works
- Aghajanyan et al. (2020) showed pre-trained models have low "intrinsic dimensionality"
- Most of the information for a new task can be captured in a low-rank subspace
- Higher ranks capture more but with diminishing returns
- Typical effective rank: 4-16 for most tasks

### Results
| Model | Full FT Params | LoRA Params (r=4) | Reduction | Performance |
|-------|---------------|-------------------|-----------|-------------|
| GPT-3 175B | 175B | 4.7M | 10,000x | Matches/exceeds full FT |
| RoBERTa-base | 125M | 0.3M | 417x | Within 0.1% of full FT |
| GPT-2 Medium | 354M | 0.35M | 1,000x | Matches full FT |

### Advantages
1. **Storage efficiency:** Each LoRA adapter is ~MB instead of ~GB
2. **No inference latency:** Merged weights are same size as original
3. **Task switching:** Swap adapters, not models
4. **Lower GPU memory:** Only gradients for r×(d+k) params, not d×k
5. **Composable:** Can potentially combine multiple LoRA adapters

### Where to Apply LoRA
- Original paper tested applying to different attention matrices
- Best results: apply to **both Q and V** projections
- More recent work: apply to **all linear layers** (Q, K, V, O, FFN up, FFN down)
- Rank choice: r=4 competitive for most tasks, r=16 slightly better, r>64 rarely helps

---

## Paper 2: QLoRA (Dettmers et al., 2023)

**Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs"  
**ArXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)  
**Authors:** Dettmers, Pagnoni, Holtzman, Zettlemoyer (University of Washington)  

### The Problem
- LoRA reduces trainable params but the frozen model still takes massive memory
- A 65B model in 16-bit needs ~130GB VRAM just to load
- Most researchers have 1-2 GPUs (24-48GB each)
- How to fine-tune large models on consumer hardware?

### The Key Innovations

**1. 4-bit NormalFloat (NF4)**
- New quantization data type optimized for normally-distributed weights
- Pre-trained weights are approximately normally distributed
- NF4 provides information-theoretically optimal quantization for this distribution
- Better than standard 4-bit integer or 4-bit float

**2. Double Quantization**
- Quantize the quantization constants themselves
- Reduces memory overhead of storing scale factors
- Saves an additional ~0.37 bits per parameter

**3. Paged Optimizers**
- Use CPU RAM when GPU memory spikes (during gradient checkpointing)
- Automatically manages memory transfers
- Prevents OOM during training spikes

### The Result
- **65B model fine-tuned on single 48GB GPU** (previously impossible)
- Preserves full 16-bit fine-tuning performance
- Guanaco (their best model) reaches 99.3% of ChatGPT performance on Vicuna benchmark
- Only 24 hours of training on a single GPU

### QLoRA Memory Comparison
| Model Size | Full FT (16-bit) | LoRA (16-bit) | QLoRA (4-bit) |
|-----------|-----------------|---------------|---------------|
| 7B | ~28GB | ~18GB | ~6GB |
| 13B | ~52GB | ~34GB | ~10GB |
| 33B | ~132GB | ~86GB | ~22GB |
| 65B | ~260GB | ~170GB | ~42GB |

---

## Paper 3: LoRA-XS (2024)

**ArXiv:** [2405.17604](https://arxiv.org/abs/2405.17604)  

### The Idea
- Regular LoRA: A (r×k) and B (d×r) are both trainable
- LoRA-XS: Compute SVD of pre-trained W, use the top-r singular vectors as FROZEN A and B
- Insert a tiny trainable r×r matrix between them
- **100x fewer parameters** than standard LoRA for 7B models

---

## The LoRA Family Tree

```
Full Fine-Tuning (100% params)
    │
    ├── Adapters (Houlsby et al., 2019) — adds layers, has inference latency
    │
    ├── LoRA (2021) — low-rank weight updates, no inference latency
    │       │
    │       ├── QLoRA (2023) — quantized base + LoRA (memory efficiency)
    │       │
    │       ├── LoRA-XS (2024) — SVD-initialized, fewer params
    │       │
    │       ├── DoRA (2024) — decomposes into magnitude + direction
    │       │
    │       ├── AdaLoRA (2023) — adaptive rank allocation
    │       │
    │       └── LoRA+ (2024) — different learning rates for A and B
    │
    └── Prefix Tuning / Prompt Tuning — prepend learnable tokens
```

---

## Practical Decision Guide

**When to use full fine-tuning:**
- You have a small model (<1B params) AND plenty of GPU memory
- You need maximum possible performance on a single task
- You're creating a base model, not adapting one

**When to use LoRA:**
- Multiple tasks/clients needing different model behaviors
- Limited GPU memory (single GPU training)
- Rapid iteration on fine-tuning experiments
- Need to preserve base model capabilities

**When to use QLoRA:**
- Large models (33B+) on consumer hardware (24-48GB GPU)
- Budget constraints — can't afford multi-GPU setup
- Acceptable to have slightly slower training (quantization overhead)
- Democratizing access to large model fine-tuning

---

## Hype vs. Reality

### What's Real
- LoRA genuinely matches full fine-tuning for most tasks (verified across many studies)
- QLoRA genuinely enables 65B fine-tuning on single GPU
- The low-rank hypothesis is empirically validated
- LoRA adapters are practical for production deployment

### What's Overhyped
- "LoRA is always as good as full fine-tuning" — for some complex tasks, full FT still wins
- "r=4 is always enough" — depends on task complexity and model size
- "LoRA composes linearly" — combining multiple LoRA adapters is non-trivial
- "QLoRA has no quality loss" — there IS a small gap for certain benchmarks, usually <1%
- "Fine-tuning solves hallucination" — it does NOT; it adjusts style and knowledge surface

---

## Visual Suggestions
- [DIAGRAM] LoRA mechanism: frozen W with parallel A→B path
- [DIAGRAM] QLoRA: 4-bit quantized model + LoRA adapters
- [CHART] Memory usage comparison: Full FT vs. LoRA vs. QLoRA
- [TREE] LoRA family evolution diagram
- [TABLE] When to use what — decision matrix
- [FORMULA] W' = W + BA with dimensional annotations
