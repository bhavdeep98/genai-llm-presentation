# Generative AI & LLMs: From Foundations to Innovation
## Presentation Outline (Step-by-Step)

---

## Slide 1: Title
**Generative AI & LLMs: Architectures, Fine-Tuning, and Building What's Next**
- Subtitle: A research-grounded guide for builders

---

## Slide 2: What This Talk Is About
- We'll trace the entire LLM stack from first principles
- Every claim backed by original papers
- Goal: You leave knowing *why* things work, not just what they do
- By the end: You'll have the foundation to innovate

---

## Section 1: The Foundation — Transformers (15 min)

### Slide 3: Before Transformers — The Problem
- RNNs: Sequential (slow), forget long-range patterns
- Visual: Sequential vs. parallel processing diagram
- Key question: "What if we could look at everything at once?"

### Slide 4: The Attention Mechanism
- Q, K, V intuition: "What am I looking for?" / "What do I offer?" / "What information do I give?"
- Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Visual: Attention weights matrix showing which words attend to which
- [Paper: Vaswani et al., 2017, arXiv:1706.03762]

### Slide 5: Multi-Head Attention
- Why multiple heads: different heads learn different relationships
- One head for syntax, another for semantics, another for position
- Visual: Split → parallel attention → concatenate
- 8 heads in original paper, 32-128 in modern models

### Slide 6: The Full Transformer
- Encoder-Decoder architecture diagram
- Key components: positional encoding, residual connections, layer norm, FFN
- Visual: The iconic Figure 1 from the paper

### Slide 7: Why This Changed Everything
- Fully parallelizable → 100x faster training than RNNs
- O(1) path length between any two positions
- BLEU 28.4 on English-German (new SOTA at fraction of cost)
- Set the stage for everything that follows

---

## Section 2: Scaling — The GPT Story (12 min)

### Slide 8: The Pre-train + Fine-tune Paradigm
- GPT-1 (2018): 117M params, decoder-only
- Key idea: Learn language first (unsupervised), then adapt (supervised)
- [Paper: Radford et al., 2018]
- Visual: Two-phase training diagram

### Slide 9: Scale Unlocks Capabilities
- GPT-2 (2019): 1.5B params → zero-shot abilities emerge
- GPT-3 (2020): 175B params → in-context learning (few-shot)
- Visual: Log-scale parameter growth with capability milestones
- [Paper: Brown et al., 2020, arXiv:2005.14165]

### Slide 10: Scaling Laws — The Science of "Bigger"
- Loss ∝ N^(-0.076) (model size) | Loss ∝ D^(-0.095) (data)
- Kaplan (2020): Favor bigger models
- Chinchilla (2022): Balance model size and data equally
- Visual: Power-law curves on log-log plot
- [Papers: arXiv:2001.08361, arXiv:2203.15556]

### Slide 11: The Chinchilla Moment
- Chinchilla 70B beat Gopher 280B (same compute, better allocation)
- Rule: ~20 tokens per parameter for optimal training
- Shifted industry from "make it bigger" to "train it better"
- Visual: Chinchilla vs. Gopher comparison chart

---

## Section 3: Alignment — Making Models Useful (12 min)

### Slide 12: The Alignment Problem
- Raw GPT-3: Powerful but uncontrollable
- Hallucination, toxicity, not following instructions
- More params ≠ more helpful
- Key question: "How do we make it do what users actually want?"

### Slide 13: InstructGPT & RLHF Pipeline
- Stage 1: SFT (demonstrate good behavior)
- Stage 2: Train reward model (rank outputs)
- Stage 3: PPO optimization (maximize reward)
- Visual: 3-stage pipeline diagram
- Key result: 1.3B InstructGPT > 175B GPT-3 (human preference)
- [Paper: Ouyang et al., 2022, arXiv:2203.02155]

### Slide 14: DPO — Simplifying Alignment
- Skip the reward model entirely
- Direct optimization from preference pairs
- One hyperparameter (β) instead of many
- Same results, much simpler training
- [Paper: Rafailov et al., 2023, arXiv:2305.18290]
- Visual: RLHF (3 models) vs. DPO (2 models) pipeline

### Slide 15: Alignment Limitations (Honest Assessment)
- Reward hacking: models game the reward function
- Scalable oversight: Can humans evaluate complex outputs?
- Doesn't solve safety — just adjusts output distribution
- [Paper: Casper et al., 2023, arXiv:2307.15217]

---

## Section 4: Fine-Tuning — Making Models Your Own (15 min)

### Slide 16: Why Fine-Tune?
- Pre-trained model knows language; you teach it YOUR task
- Format, style, domain knowledge, specific behaviors
- Spectrum: Full fine-tuning → Parameter-efficient → Prompt tuning

### Slide 17: The Problem with Full Fine-Tuning
- GPT-3: 175B parameters × 2 bytes = 350GB per variant
- Need gradient storage → 3-4x model size in memory
- 10 tasks = 10 full copies of the model
- Impractical for most organizations

### Slide 18: LoRA — The Key Insight
- Pre-trained weight updates have LOW intrinsic rank
- Instead of updating W (d×d), decompose: ΔW = B×A (d×r) × (r×d)
- r=4-16 instead of d=4096 → 10,000x fewer parameters
- Visual: Weight matrix with parallel low-rank path
- [Paper: Hu et al., 2021, arXiv:2106.09685]

### Slide 19: LoRA — How It Works
- Freeze W, train A and B only
- At inference: merge W + BA → zero extra latency!
- Switch tasks: swap the tiny A, B matrices
- Visual: Forward pass with LoRA path highlighted

### Slide 20: QLoRA — Democratizing Fine-Tuning
- Base model in 4-bit (NF4 quantization) + LoRA adapters in 16-bit
- 65B model on single 48GB GPU (previously needed ~260GB)
- Innovations: NF4, double quantization, paged optimizers
- [Paper: Dettmers et al., 2023, arXiv:2305.14314]
- Visual: Memory comparison chart (Full FT vs. LoRA vs. QLoRA)

### Slide 21: The LoRA Family & Decision Guide
- LoRA-XS, DoRA, AdaLoRA, QLoRA — evolution tree
- When to use what: decision matrix
- Practical tips: rank selection, layer selection, learning rate

---

## Section 5: Building with LLMs — API & RAG (12 min)

### Slide 22: The API Economy
- Integration patterns: Direct call, RAG, Agents, Chains
- Cost structure: Input tokens, output tokens, latency trade-offs
- Visual: 4 patterns as architecture diagrams

### Slide 23: RAG — Retrieval-Augmented Generation
- Problem: Models can't know everything; knowledge goes stale
- Solution: Retrieve relevant documents, condition generation on them
- Architecture: Query → Embed → Retrieve → Generate
- [Paper: Lewis et al., 2020, arXiv:2005.11401]
- Visual: RAG pipeline diagram

### Slide 24: Agents — Reasoning + Acting
- ReAct pattern: Think → Act → Observe → Repeat
- Tool use: Search, code execution, API calls
- [Paper: Yao et al., 2022, arXiv:2210.03629]
- Visual: Agent loop diagram with example

### Slide 25: Production Reality Check
- What works: RAG for QA, structured extraction, simple tool use
- What doesn't: Long-horizon agents, guaranteed accuracy
- Error compounding: 90% per step × 5 steps = 59% success
- Visual: Success rate decay chart

---

## Section 6: Alternative Architectures (8 min)

### Slide 26: Beyond Dense Transformers
- Mixture of Experts: Scale params without scaling compute
- State Space Models (Mamba): O(n) instead of O(n²)
- Test-Time Compute: Think harder, not bigger
- Visual: Architecture landscape map

### Slide 27: MoE — How the Industry Actually Scales
- Router selects top-k experts per token
- DeepSeek-V3: 671B total, ~37B active per token
- Trade-off: Memory for all experts, but compute for few
- Visual: Router → expert selection diagram

### Slide 28: Mamba & Linear-Time Alternatives
- SSMs: Process sequences in O(n) time
- Mamba-3B ≈ Transformer-6B (2x efficiency)
- But: Similar expressiveness limits for state-tracking
- Hybrid models (SSM + Attention) currently winning
- [Paper: Gu & Dao, 2023, arXiv:2312.00752]

---

## Section 7: Hype vs. Reality (8 min)

### Slide 29: The 5 Questions Framework
1. Is there a peer-reviewed paper?
2. Are benchmarks reproducible?
3. What's the failure mode?
4. Who benefits from this claim?
5. What's the base rate (vs. simpler methods)?

### Slide 30: What's Real vs. What's Not
- ✅ Real: Scaling improves capability, LoRA works, RAG helps factuality
- ⚠️ Overstated: "Emergence," "LLMs reason," "Agents do everything"
- ❌ Hype: "AGI in 2 years," "Models understand," "Benchmarks = real performance"
- Visual: 3-column table with evidence links

### Slide 31: The Social Media Filter
- Paper says "reduces hallucination by 20%" → Tweet says "Solves hallucination!"
- Paper says "matches GPT-4 on this benchmark" → Tweet says "GPT-4 killer!"
- Always read the limitations section
- Visual: Paper abstract vs. social media interpretation comparison

---

## Section 8: Call to Action — Build Something (5 min)

### Slide 32: Where Innovation Happens
- Real gaps: Evaluation methods, efficiency, retrieval quality, reliability
- Methodology: Identify limitation → Hypothesize mechanism → Test → Compare to baselines
- You now have the foundations — use them

### Slide 33: Your Innovation Toolkit
- Papers: You have the full catalog (12+ foundational papers)
- Methods: Attention, scaling, LoRA, RLHF, RAG, agents
- Critical thinking: The 5 Questions Framework
- Community: arXiv, HuggingFace, open-source models

### Slide 34: Closing
- LLMs are powerful tools with real limitations
- The basics matter more than the hype
- Understanding *why* enables you to build *what's next*
- "The best way to predict the future is to invent it"

---

## Timing Summary
| Section | Duration | Slides |
|---------|----------|--------|
| Transformers | 15 min | 3-7 |
| Scaling/GPT | 12 min | 8-11 |
| Alignment | 12 min | 12-15 |
| Fine-Tuning/LoRA | 15 min | 16-21 |
| APIs/RAG/Agents | 12 min | 22-25 |
| Architectures | 8 min | 26-28 |
| Hype vs. Reality | 8 min | 29-31 |
| Call to Action | 5 min | 32-34 |
| **Total** | **~87 min** | **34 slides** |
