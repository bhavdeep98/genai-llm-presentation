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

## Section 1: The Foundation — LLM Architecture (20 min)

### Slide 3: What Is an LLM, Really?
- A next-token prediction machine — given context, predict what comes next
- **Mystery novel analogy:** The story is the context, the clues are the relevant information, the last word to predict is "the murderer is ___"
- There's a probability distribution over many possible answers — but only one is correct
- The architecture's job: figure out which clue matters most for the prediction
- Visual: A sentence with the last word blanked, probability bars over candidates

### Slide 4: Embeddings — Words as Vectors in Space
- Words aren't symbols to a model — they become points in high-dimensional space
- Intuition: "King" − "Man" + "Woman" ≈ "Queen" (directions encode meaning)
- Similar words cluster together; dissimilar words are far apart
- Each dimension captures some feature (formal/informal, concrete/abstract, etc.)
- Modern LLMs use ~4096-12288 dimensional spaces
- [Paper: Mikolov et al., 2013, "Efficient Estimation of Word Representations in Vector Space," arXiv:1301.3781]
- Visual: 2D/3D projection of word clusters, showing relationships as geometric directions

### Slide 5: The Attention Mechanism — Who Should I Listen To?
- The core question: given a word, which OTHER words in the context matter most?
- **Q, K, V intuition (credit: 3Blue1Brown):**
  - Query: "What am I looking for?"
  - Key: "What do I offer?"
  - Value: "What information do I give if you pick me?"
- Analogy: Q is the detective's question, K is which clues respond to that question, V is what those clues actually reveal
- Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- The softmax creates a probability distribution — a weighted vote on relevance
- [Paper: Vaswani et al., 2017, "Attention Is All You Need," arXiv:1706.03762]
- [Visual reference: 3Blue1Brown "Attention in transformers, step-by-step" (2024)]
- Visual: Attention weights matrix showing which words attend to which

### Slide 6: Multi-Head Attention — Multiple Perspectives
- Why multiple heads: different heads learn different relationships
- One head for syntax ("what grammatically connects"), another for coreference ("who does 'they' refer to"), another for semantic similarity
- Like having 8-128 detectives each following a different line of investigation
- Visual: Split → parallel attention → concatenate
- 8 heads in original paper, 32-128 in modern models

### Slide 7: The MLP Layers — Where Facts Live
- Attention finds relationships; MLP layers store knowledge
- Think of MLPs as the model's "memory of facts and nuance"
- Research shows: factual associations (e.g., "Eiffel Tower is in ___") are encoded in MLP weights
- MLP acts like a key-value store — input pattern triggers stored knowledge
- Attention = routing/wiring; MLP = the actual content being routed
- [Paper: Dai et al., 2022, "Knowledge Neurons in Pretrained Transformers," arXiv:2104.08696]
- [Paper: Geva et al., 2021, "Transformer Feed-Forward Layers Are Key-Value Memories," arXiv:2012.14913]
- Visual: Simplified diagram showing Attention → MLP interplay in one transformer block

### Slide 8: The Full Transformer Architecture
- Stack attention + MLP + residual connections + layer norm = one block
- Repeat 32-128 times = a full LLM
- Encoder-Decoder (original) → Decoder-only (GPT-style) is what most modern LLMs use
- Key components: positional encoding, residual connections, layer normalization
- Visual: The iconic Figure 1 from Vaswani et al. (adapted for decoder-only)

### Slide 9: Why Transformers Won (But They're Not the End)
- Fully parallelizable → 100x faster training than RNNs
- O(1) path length between any two positions (vs. O(n) for RNNs)
- BLEU 28.4 on English-German (new SOTA at fraction of cost)
- **But:** O(n²) attention complexity limits context length
- **But:** Autoregressive prediction ≠ understanding/planning
- These limitations motivate the next frontier...

### Slide 10: The Frontier — Beyond Token Prediction (JEPA)
- LLMs predict the *next token* — but is that how intelligence works?
- Yann LeCun's argument: humans build *world models*, not word predictors
- **JEPA (Joint Embedding Predictive Architecture):** predict in *representation space*, not pixel/token space
- Key difference: JEPA learns abstract representations that discard irrelevant details
- I-JEPA (images, 2023): predicts missing patches in latent space, not pixel space
- V-JEPA (video, 2024): learns temporal structure without generating pixels
- Not a replacement for transformers today — but a research frontier addressing fundamental limitations
- [Paper: LeCun, 2022, "A Path Towards Autonomous Machine Intelligence" (position paper)]
- [Paper: Assran et al., 2023, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," arXiv:2301.08243]
- [Org: Meta AI — I-JEPA, V-JEPA]
- Visual: Autoregressive (predict token by token) vs. JEPA (predict in latent space) comparison diagram

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
| LLM Architecture | 20 min | 3-10 |
| Scaling/GPT | 12 min | 11-14 |
| Alignment | 12 min | 15-18 |
| Fine-Tuning/LoRA | 15 min | 19-24 |
| APIs/RAG/Agents | 12 min | 25-28 |
| Architectures | 8 min | 29-31 |
| Hype vs. Reality | 8 min | 32-34 |
| Call to Action | 5 min | 35-37 |
| **Total** | **~92 min** | **37 slides** |
