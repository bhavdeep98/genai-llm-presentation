# Alternative Architectures & What's Next — Summary

---

## Architecture 1: Mamba — Selective State Space Models (Gu & Dao, 2023)

**Paper:** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"  
**ArXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)  

### The Problem with Transformers
- Self-attention is O(n²) in sequence length
- For a 100K token context: attention matrix is 100K × 100K = 10 billion entries
- Memory grows quadratically — limits practical sequence length
- Inference cost scales linearly with sequence length (KV cache)

### The State Space Model (SSM) Idea
Map sequences through a continuous-time system, then discretize:
```
x'(t) = Ax(t) + Bu(t)    — state evolution
y(t) = Cx(t) + Du(t)     — output
```

Key properties:
- Linear time complexity O(n) — processes each token in constant time
- Constant memory at inference (fixed-size hidden state)
- Can be computed as a convolution during training (parallelizable!)

### Mamba's Innovation: Selectivity
Previous SSMs (S4) used fixed parameters A, B, C → content-agnostic.

Mamba makes B, C, and Δ (discretization step) **input-dependent:**
- The model can "select" what information to remember or forget
- Similar to how gating works in LSTMs, but more principled
- Enables content-aware processing while maintaining linear time

### Results
- **Mamba-3B outperforms Transformers of the same size** on language modeling
- **Matches Transformers of twice its size** (3B Mamba ≈ 6B Transformer)
- 5x higher throughput than Transformers at inference for long sequences
- Strong on language, audio, and genomics

### Limitations (Critical — from "Illusion of State" paper)
- Despite the "state" formulation, SSMs have **similar expressiveness limitations to Transformers** for state-tracking problems
- Cannot efficiently track state across very long contexts in practice
- Not a universal replacement for attention (some tasks still need it)
- Less community tooling and ecosystem compared to Transformers
- Hybrid architectures (Mamba + attention layers) often outperform pure Mamba

---

## Architecture 2: Mixture of Experts (MoE)

### Original Concept
- Shazeer et al., "Outrageously Large Neural Networks" (2017)
- Router selects which "expert" sub-networks to activate per token
- Only a fraction of total parameters active at any time

### How MoE Works
```
Input Token → Router (learned gating network) → Select Top-k Experts → 
Weighted Sum of Expert Outputs → Output

Total params: 100B
Active params per token: 12B (with top-2 routing from 64 experts)
```

### Why MoE Matters
- **Scale without proportional compute:** Can have trillion-parameter models with sub-100B active compute
- **Specialization:** Different experts learn different patterns/domains
- **Training efficiency:** More total capacity for same training FLOP budget

### Key MoE Models (2024-2026)
| Model | Total Params | Active Params | Experts | Architecture |
|-------|-------------|---------------|---------|--------------|
| Mixtral 8x7B | ~47B | ~13B | 8 experts, top-2 | Transformer + MoE FFN |
| DeepSeek-V2 | 236B | ~21B | 160 experts, top-6 | MLA + MoE |
| DeepSeek-V3 | 671B | ~37B | 256 experts, top-8 | MLA + MoE |
| GPT-4 (rumored) | ~1.8T | ~200B | MoE | Transformer + MoE |

### MoE Challenges
1. **Load balancing:** Some experts get overloaded, others underused
2. **Communication overhead:** In distributed training, routing tokens to experts across GPUs
3. **Training instability:** Router can collapse to always selecting same experts
4. **Inference complexity:** Need all expert weights in memory even if only a few are active
5. **Evaluation difficulty:** Standard benchmarks don't test expert specialization

---

## Architecture 3: Multi-head Latent Attention (MLA)

### DeepSeek's Innovation
- Standard attention: stores K, V caches → O(n × d × h) memory per layer
- MLA: compress K, V into low-rank latent representations
- Significant KV cache reduction without quality loss
- Combined with MoE for the full DeepSeek architecture

---

## The Architecture Landscape (2026)

```
┌────────────────────────────────────────────────┐
│            Current Architecture Zoo              │
├────────────────────────────────────────────────┤
│                                                  │
│  Dense Transformers ──→ MoE Transformers         │
│  (GPT-4o, Claude 3.5)   (DeepSeek, Mixtral)    │
│                                                  │
│  State Space Models ──→ Hybrid (SSM + Attn)     │
│  (Mamba, Mamba-2)       (Jamba, Zamba)          │
│                                                  │
│  New Attention Variants                          │
│  (MLA, Linear Attention, Flash Attention)       │
│                                                  │
│  Emerging: Test-Time Compute                     │
│  (o1-style reasoning, search at inference)      │
│                                                  │
└────────────────────────────────────────────────┘
```

### Test-Time Compute (2024-2026 Frontier)
- Instead of bigger models → use more compute at inference time
- Model "thinks longer" on harder problems (chain-of-thought, search)
- OpenAI o1, o3: spend more FLOPs at inference rather than pre-training
- Shifts the scaling curve: smaller models + more inference compute can match larger models

---

## Hype vs. Reality

### What's Real
- MoE genuinely enables larger models at manageable compute costs
- Linear-time alternatives to attention exist and work
- Hybrid architectures often beat pure approaches
- Test-time compute is a genuinely new scaling axis

### What's Overhyped
- "Mamba replaces Transformers" → Hybrid is winning, not pure replacement
- "SSMs solve the context length problem" → They have their own limitations
- "MoE is free scaling" → Communication costs, load balancing, memory for all experts
- "The architecture matters most" → Data, training recipe, and scale often matter more

### What To Watch
1. Will pure SSMs catch up to Transformers at scale? (Unclear as of 2026)
2. How far can test-time compute push reasoning? (Active research)
3. Will new architectures require new fine-tuning methods? (LoRA works for MoE, but differently)
4. Energy efficiency of different architectures at scale

---

## Visual Suggestions
- [DIAGRAM] Transformer O(n²) vs. SSM O(n) scaling curves
- [DIAGRAM] MoE architecture: router selecting experts
- [TABLE] Architecture comparison: params, active compute, speed
- [TIMELINE] Architecture evolution 2017-2026
- [CHART] Accuracy vs. compute for different architectures
