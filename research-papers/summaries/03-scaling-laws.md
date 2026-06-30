# Scaling Laws — Summary

---

## Paper 1: Scaling Laws for Neural Language Models (Kaplan et al., 2020)

**ArXiv:** [2001.08361](https://arxiv.org/abs/2001.08361)  
**Authors:** Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (OpenAI)  

### The Problem
- No principled way to predict how model performance scales with resources
- Should you make the model bigger? Train longer? Get more data?
- Everyone was guessing

### The Key Insight
Loss follows smooth power-law relationships across 7+ orders of magnitude:
- **L(N) ∝ N^(-0.076)** — Loss scales with parameter count
- **L(D) ∝ D^(-0.095)** — Loss scales with dataset size  
- **L(C) ∝ C^(-0.050)** — Loss scales with compute budget

These relationships are remarkably predictable and consistent.

### Practical Implications
1. Bigger models are more sample-efficient (learn more per token)
2. For a fixed compute budget, it's better to train a large model for fewer steps
3. Model shape (width vs. depth) matters less than total parameter count
4. Returns are diminishing but predictable

### Limitation/Controversy
- Suggested making models very large and training for fewer steps
- Didn't account for inference cost (a huge model trained briefly is expensive to deploy)
- Later contradicted by Chinchilla

---

## Paper 2: Training Compute-Optimal Large Language Models — "Chinchilla" (Hoffmann et al., 2022)

**ArXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)  
**Authors:** Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, et al. (DeepMind)  

### The Problem
- Kaplan's laws suggested bigger models are always better
- But 175B+ parameter models were expensive to serve
- Was there a more efficient balance?

### The Key Insight
**For compute-optimal training, model size and training tokens should scale equally.**

For every doubling of model size, you should also double the training data. Previous models (like GPT-3) were significantly under-trained relative to their size.

### Evidence
- Trained 400+ models from 70M to 16B parameters
- Tested on 5 to 500 billion tokens
- Found the optimal ratio: ~20 tokens per parameter

### Results: Chinchilla (70B) vs. Gopher (280B)
- Chinchilla: 70B parameters, 1.4T tokens
- Gopher: 280B parameters, 300B tokens
- **Chinchilla outperformed Gopher** despite being 4x smaller
- Used the same compute budget more efficiently

### Impact on the Field
1. Shifted focus from "make it bigger" to "train it longer on more data"
2. Influenced LLaMA (Meta): 7B-65B models trained on 1-1.4T tokens
3. Made smaller, better-trained models practical for deployment
4. Changed the industry's approach to model development

---

## Kaplan vs. Chinchilla — The Reconciliation

| Aspect | Kaplan (2020) | Chinchilla (2022) |
|--------|--------------|-------------------|
| Optimal N scaling | N ∝ C^0.73 | N ∝ C^0.50 |
| Optimal D scaling | D ∝ C^0.27 | D ∝ C^0.50 |
| Implication | Favor larger models | Balance model & data |
| Parameter counting | Non-embedding params | Total params |

The discrepancy was partially due to Kaplan counting non-embedding parameters and analyzing at smaller scales (as shown in "Reconciling Kaplan and Chinchilla Scaling Laws," 2024, arXiv:2406.12907).

---

## Hype vs. Reality

### What's Real
- Power-law scaling is empirically robust across many orders of magnitude
- Compute-optimal training genuinely saves resources
- These laws guided practical decisions at major labs

### What's Overhyped
- "Just scale and capabilities will emerge" — emergent capabilities are unpredictable and may be artifacts of metric choice
- Scaling laws for loss don't directly translate to task performance
- Individual tasks can degrade with scale (inverse scaling)
- The laws break down at the extremes and don't predict qualitative capability jumps

## Visual Suggestions
- [CHART] Log-log plot of loss vs. compute (the classic power-law curve)
- [COMPARISON] Chinchilla (70B) vs. Gopher (280B) benchmark scores
- [TABLE] Kaplan vs. Chinchilla optimal ratios
- [DIAGRAM] Decision tree: Given X compute, how to allocate between N and D
