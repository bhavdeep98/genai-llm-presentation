# GPT Series Evolution — Summary

---

## GPT-1 (2018)

**Paper:** "Improving Language Understanding by Generative Pre-Training"  
**Authors:** Radford, Narasimhan, Salimans, Sutskever (OpenAI)  
**Source:** [OpenAI](https://openai.com/index/language-unsupervised/)  

### The Problem
- NLP relied on task-specific labeled data (expensive, limited)
- No general-purpose language understanding model existed

### The Insight
Pre-train a decoder-only transformer on unlabeled text (next-token prediction), then fine-tune on labeled data for specific tasks. The pre-trained representations transfer well across tasks.

### Architecture
- 12-layer decoder-only Transformer
- 117M parameters
- Trained on BooksCorpus (~7,000 unpublished books)
- Fine-tuned with task-specific linear heads

### Results
- Beat task-specific models on 9 out of 12 NLP benchmarks
- Showed that unsupervised pre-training provides useful inductive bias

### Limitation
- Still needed fine-tuning for each downstream task
- Small scale by today's standards

---

## GPT-2 (2019)

**Paper:** "Language Models are Unsupervised Multitask Learners"  
**Authors:** Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI)  
**Source:** [OpenAI](https://openai.com/research/better-language-models)  

### The Problem
- GPT-1 still required fine-tuning per task
- Could a model learn tasks implicitly from diverse enough pre-training data?

### The Insight
With enough diverse data and scale, the model learns to perform tasks zero-shot — no fine-tuning needed. "All tasks can be framed as language modeling."

### Architecture
- 48-layer decoder-only Transformer
- 1.5B parameters (13x GPT-1)
- Trained on WebText (40GB of curated internet text, ~8M documents)
- Modified: Layer normalization moved to input, additional normalization added

### Results
- Zero-shot SOTA on 7 out of 8 language modeling benchmarks
- Could generate coherent paragraphs of text
- Demonstrated emerging capabilities not explicitly trained for

### Limitation
- Zero-shot performance was inconsistent across tasks
- Still far from supervised SOTA on most benchmarks
- Generated text could be incoherent over longer passages

---

## GPT-3 (2020)

**Paper:** "Language Models are Few-Shot Learners"  
**ArXiv:** [2005.14165](https://arxiv.org/abs/2005.14165)  
**Authors:** Brown, Mann, Ryder, et al. (OpenAI)  

### The Problem
- Zero-shot was unreliable, fine-tuning required task-specific data
- Could models learn from just a few examples in context?

### The Insight
At sufficient scale (175B parameters), models can learn new tasks from just a few demonstration examples provided in the prompt — "in-context learning." No weight updates needed.

### Architecture
- 96-layer decoder-only Transformer
- 175B parameters (100x GPT-2)
- 2048-token context window
- Trained on ~300B tokens from filtered internet, books, Wikipedia
- Alternating dense and sparse attention patterns

### Results
- Few-shot: Near or exceeds fine-tuned models on many tasks
- Demonstrated translation, arithmetic, coding, and creative writing
- Introduced scaling as a viable path to capability

### Critical Analysis
- Massive compute cost (~$4.6M estimated training cost in 2020)
- Still hallucinates confidently
- Not aligned — could produce harmful/biased outputs
- In-context learning is poorly understood theoretically

---

## The Evolution Pattern

| Model | Year | Parameters | Key Innovation |
|-------|------|-----------|----------------|
| GPT-1 | 2018 | 117M | Pre-train + fine-tune paradigm |
| GPT-2 | 2019 | 1.5B | Zero-shot via scale |
| GPT-3 | 2020 | 175B | In-context few-shot learning |
| GPT-4 | 2023 | ~1.8T (rumored MoE) | Multimodal, reasoning |

### What We Learn From This Progression
1. **Scale enables emergent capabilities** — but not predictably
2. **Data quality matters as much as quantity** (WebText curation, filtering)
3. **Architecture stayed mostly constant** — the transformer didn't change much
4. **The real innovation was in training methodology** — what to train on, how to use the model

## Hype vs. Reality
- **Hype:** "GPT-3 understands language" → It pattern-matches at scale, not "understanding"
- **Reality:** In-context learning works but is sensitive to prompt formatting
- **Hype:** "Scaling is all you need" → Chinchilla showed we were doing it wrong
- **Reality:** Scale helps but isn't sufficient for reliability or alignment

## Visual Suggestions
- [CHART] Parameter count growth: log scale, GPT-1 → GPT-3
- [DIAGRAM] Training paradigm shift: fine-tuning → prompting → in-context learning
- [TABLE] Comparison of capabilities across GPT versions
- [TIMELINE] 2018-2023 evolution with key milestones
