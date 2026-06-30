# Learning Path — Step-by-Step Understanding

This document is for YOUR learning. It maps the "why" chain so you deeply understand each concept before teaching it.

---

## The Chain of Ideas (How Each Thing Led to the Next)

```
RNNs are slow & forget (1990s-2016)
    ↓ Insight: "What if we look at everything at once?"
Transformers — Attention Is All You Need (2017)
    ↓ Insight: "What if we pre-train on all text, then fine-tune?"
GPT-1 — Pre-train + Fine-tune (2018)
    ↓ Insight: "What if we just scale it up?"
GPT-2 — Zero-shot from scale (2019)
    ↓ Insight: "What if we scale even more?"
GPT-3 — In-context learning, few-shot (2020)
    ↓ Problem: "But it's not actually helpful or safe"
InstructGPT/RLHF — Alignment via human feedback (2022)
    ↓ Problem: "RLHF is complex and expensive"
DPO — Direct preference optimization (2023)
    ↓ Problem: "Full fine-tuning is too expensive"
LoRA — Low-rank adaptation (2021)
    ↓ Problem: "Even loading the model is too expensive"
QLoRA — Quantized + LoRA (2023)
    ↓ Question: "What about knowledge the model doesn't have?"
RAG — Retrieve, then generate (2020)
    ↓ Question: "Can models use tools and take actions?"
ReAct/Agents — Reasoning + Acting (2022)
    ↓ Question: "Does it have to be a transformer?"
Mamba/SSMs — Linear-time alternatives (2023)
    ↓ Question: "Can we scale without proportional cost?"
MoE — Sparse activation (2017→2024)
    ↓ Current frontier: "Can we think harder, not bigger?"
Test-Time Compute — o1/o3 reasoning (2024-2025)
```

---

## Key Conceptual Building Blocks

### Block 1: The Attention Mechanism
**Analogy:** Imagine reading a book. Instead of reading word-by-word (RNN), you can glance at any previous word instantly. Self-attention lets every word "look at" every other word to decide its meaning in context.

**Why it works:** Language is full of long-range dependencies. "The cat sat on the mat because **it** was tired" — "it" refers to "cat" which might be 10 words away. Attention handles this in O(1) path length.

**Why it's expensive:** n tokens × n tokens = n² comparisons. This is the fundamental cost of transformers and drives all efficiency research.

### Block 2: The Pre-training Insight
**Analogy:** Learning a language by reading millions of books vs. learning from a textbook. The first gives you intuition for how language works; the second gives you specific task knowledge. GPT does both: pre-train (read books), then fine-tune (study textbook).

**Why it works:** Next-token prediction is a surprisingly rich training signal. To predict the next word, you must understand syntax, semantics, facts, reasoning patterns — all emerge from this simple objective.

### Block 3: Scaling Laws
**Analogy:** If training a model is like filling a bucket with knowledge, scaling laws tell you: "Use a bigger bucket (more params) AND more water (more data) at the same rate."

**Why it matters:** Before scaling laws, training large models was expensive guesswork. After: you can predict performance before spending $10M on training.

### Block 4: Alignment (RLHF/DPO)
**Analogy:** A gifted student who knows everything but doesn't know how to communicate or behave appropriately. RLHF is like teaching them manners and helpfulness through feedback: "This response was helpful" / "This one wasn't."

**Why it works:** The model already has the capability; alignment just reshapes the output distribution toward human preferences. That's why a tiny 1.3B aligned model beats a 175B unaligned one.

### Block 5: LoRA
**Analogy:** You're a native English speaker learning French. You don't rewire your entire brain (full fine-tuning) — you add a small "adapter" that maps between what you know and the new language. LoRA adds a tiny parallel path that captures what's different.

**Why it works mathematically:** When you fine-tune for a specific task, most of the weight change can be captured by a low-rank approximation. The "rank" of the change is much smaller than the matrix dimensions because the task is much simpler than "all of language."

### Block 6: RAG
**Analogy:** An expert who can reason brilliantly but has a perfect reference library. Instead of memorizing every fact, they know how to look things up and incorporate what they find into their reasoning.

**Why it works:** Separates "knowing how to think" (model weights) from "having information" (retrieved documents). You can update the library without retraining the expert.

---

## Common Misconceptions to Avoid

1. **"LLMs understand language"** → They predict likely continuations. Understanding is our anthropomorphic projection onto effective pattern matching.

2. **"More parameters = smarter"** → More params = more capacity, but how you use that capacity (data quality, training recipe, alignment) matters as much or more.

3. **"LoRA fine-tuning teaches new knowledge"** → It primarily adjusts style, format, and preference. For genuinely new facts, you need RAG or more pre-training data.

4. **"RLHF solves safety"** → It makes models more pleasant and instruction-following. Safety is a much harder, unsolved problem.

5. **"Benchmarks tell the full story"** → Benchmark contamination is widespread. Always test on your own data.

6. **"Transformers will always be the architecture"** → They're dominant now, but hybrids and alternatives are evolving. Stay open.

---

## Recommended Reading Order (If Starting From Scratch)

1. **Start here:** Attention Is All You Need — read Section 3 (architecture) carefully
2. **Then:** GPT-3 paper — focus on Section 1 (introduction) and Section 3 (results)
3. **Then:** LoRA paper — read all of it (short and well-written)
4. **Then:** InstructGPT paper — focus on Section 3 (methodology)
5. **Then:** DPO paper — read Section 3 (derivation) slowly
6. **Then:** QLoRA paper — Section 3 (method) and Section 5 (results)
7. **Optional depth:** Scaling Laws, Mamba, RAG original paper

---

## Questions to Ask Yourself (Self-Test)

After studying each topic, you should be able to answer:

### Transformers
- Why is O(n²) the cost? What creates the quadratic scaling?
- What would happen without positional encodings?
- Why multi-head instead of single-head attention?

### Scaling
- If I have $1M in compute, should I train a 7B or 70B model? (Chinchilla answer)
- Why did scaling laws surprise people?
- What can't scaling laws predict?

### Alignment
- Why is a 1.3B aligned model better than 175B unaligned? What does this tell us?
- What's the key difference between RLHF and DPO mathematically?
- Can you reward-hack DPO? How?

### LoRA
- Why rank 4? Why not rank 1 or rank 1000?
- Where in the transformer should you apply LoRA? Why?
- Why is there zero inference latency with LoRA?

### RAG
- When does RAG fail?
- Why not just make the context window infinitely large?
- What determines retrieval quality?
