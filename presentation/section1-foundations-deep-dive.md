# Section 1: LLM Architecture — Deep Dive Research Document

## Purpose
This document expands each slide in Section 1 (The Foundation — LLM Architecture) into research-paper-level depth with visual references, source citations, and placement of Bhavdeep Singh Sachdeva and Swaroop Mishra's research in the timeline.

---

## Timeline of Key Contributions (Your Research in Context)

| Year | Paper/Contribution | Authors | Relevance |
|------|-------------------|---------|-----------|
| 2013 | Word2Vec | Mikolov et al. | Embeddings foundation |
| 2017 | Attention Is All You Need | Vaswani et al. | Transformer architecture |
| 2018 | GPT-1 | Radford et al. | Decoder-only pre-train + fine-tune |
| 2019 | GPT-2 | Radford et al. | Zero-shot emergence |
| 2020 | GPT-3 | Brown et al. | Few-shot, in-context learning |
| 2020 | Scaling Laws | Kaplan et al. | Power-law relationships |
| 2021 | **Natural Instructions** | **Swaroop Mishra** et al. | Pioneered instruction-tuning field |
| 2021 | Reframing Instructional Prompts to GPTk's Language | **Swaroop Mishra** et al. | How to turn task instructions into effective prompts |
| 2021 | LoRA | Hu et al. | Parameter-efficient fine-tuning |
| 2022 | **NumGLUE** | **Swaroop Mishra, Bhavdeep Sachdeva** et al. | Multi-task math reasoning benchmark |
| 2022 | **Cross-Task Generalization via NL Instructions** | **Swaroop Mishra** et al. | ACL 2022 — instruction-following generalization |
| 2022 | **Super-NaturalInstructions** | **Swaroop Mishra** et al. | EMNLP 2022 — 1,616 tasks benchmark |
| 2022 | **Generalized but not Robust?** | **Bhavdeep Sachdeva**, Mishra et al. | ACL Findings 2022 — data modification & robustness |
| 2022 | **LILA: Unified Math Reasoning Benchmark** | **Swaroop Mishra** et al. | EMNLP 2022 — 20 datasets, Python programs |
| 2022 | **"Is High Quality Data All We Need?"** | **Bhavdeep Sachdeva**, Mishra et al. | Data quality hypothesis |
| 2022 | Chinchilla | Hoffmann et al. | Compute-optimal training |
| 2022 | InstructGPT / RLHF | Ouyang et al. | Alignment pipeline |
| 2022 | LeCun — JEPA Position Paper | LeCun | World models vs. token prediction |
| 2023 | **VAIDA (Benchmark Creation)** | Arunkumar, **Mishra, Sachdeva** et al. | EACL 2023 — human-and-metric-in-the-loop |
| 2023 | **Self-Instruct** | Wang, **Mishra** et al. | ACL 2023 — LLMs generate own training data |
| 2023 | **InstructABSA** | Scaria, **Mishra** et al. | NAACL 2024 — instruction-tuning for sentiment |
| 2023 | **Pretrained Transformers Do Not Always Improve Robustness** | **Sachdeva**, Mishra et al. | Robustness analysis of pretrained models |
| 2023 | **Step-Back Prompting** | Zheng, **Mishra** et al. | ICLR 2024 — reasoning via abstraction |
| 2023 | **TarGEN** | Gupta, **Mishra** et al. | Seedless synthetic data generation |
| 2023 | DPO | Rafailov et al. | Simplified alignment |
| 2023 | QLoRA | Dettmers et al. | Democratized fine-tuning |
| 2023 | I-JEPA | Assran et al. (Meta) | Image prediction in latent space |
| 2023 | Mamba | Gu & Dao | Linear-time sequence modeling |
| 2024 | **Self-Discover** | Zhou, **Mishra** et al. | Google DeepMind — self-composed reasoning structures |
| 2024 | V-JEPA | Bardes et al. (Meta) | Video prediction in latent space |
| 2024 | IMO Silver Medal (AlphaProof) | Google DeepMind (incl. **Swaroop Mishra**) | Math reasoning system |

---

## Slide 3: What Is an LLM, Really?

### Core Concept
An LLM is fundamentally a **next-token prediction machine**. Given a sequence of tokens (context), it produces a probability distribution over the vocabulary for what comes next.

### The Mystery Novel Analogy (Presentation Framing)

> Imagine a mystery novel. The **story so far** is the context window. The **clues scattered throughout** are the relevant tokens the model attends to. The final prediction — "The murderer is ___" — is the next token the model must output.
>
> There's a probability distribution over many suspects (tokens), but the architecture's job is to figure out **which clues matter most** to make the right prediction.

This analogy works because:
1. It captures the probabilistic nature (many candidates, one correct answer)
2. It motivates attention (not all context is equally important)
3. It sets up the "relevance finding" problem that attention solves
4. It gives students an intuitive frame before any math appears

### Technical Depth

**Autoregressive Factorization:**
```
P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁)
```

The entire LLM is trained to minimize:
```
L = -∑ᵢ log P(xᵢ | x₁, ..., xᵢ₋₁; θ)
```

This is cross-entropy loss — the model learns to assign high probability to the actual next token.

**Why this matters for the audience:**
- Every capability (reasoning, coding, translation) emerges from this single objective
- The architecture doesn't "know" it's doing reasoning — it's predicting tokens
- This is both the power (generality) and the limitation (no explicit world model)

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Next token prediction diagram] | Create custom | Sentence with last word blanked + probability bars over candidates |
| ![Autoregressive generation] | 3Blue1Brown "But what is a GPT?" (2024) | Shows token-by-token generation process |
| ![Temperature/sampling visual] | Create custom | How temperature affects the probability distribution shape |

- **[SOURCE: 3Blue1Brown, "Transformers, the tech behind LLMs" (2024)](https://www.3blue1brown.com/lessons/gpt/)** — Excellent visual of token prediction as probability distributions
- **[DIAGRAM TO CREATE]** Mystery novel visual: story pages → highlighted clues → "The murderer is ___" with probability bars

---

## Slide 4: Embeddings — Words as Vectors in Space

### Core Concept
Before any attention or computation, words must become numbers. Embeddings map each token to a point in a high-dimensional vector space where **geometric relationships encode meaning**.

### Key Intuitions for Students

1. **Directions encode relationships:**
   - "King" − "Man" + "Woman" ≈ "Queen"
   - "Paris" − "France" + "Italy" ≈ "Rome"
   - These aren't hard-coded — they're *learned* from co-occurrence patterns

2. **Similarity = proximity:**
   - Words that appear in similar contexts cluster together
   - "happy", "joyful", "elated" form a cluster
   - "cat" and "dog" are closer than "cat" and "democracy"

3. **Each dimension captures something:**
   - No single dimension is "formality" or "concreteness" — they're distributed
   - But directions in the space do correspond to meaningful features
   - A model with 4096 dimensions has 4096 features per token to work with

### Technical Depth

**From Word2Vec to Modern Embeddings:**

| Era | Method | Dimensions | Key Property |
|-----|--------|-----------|-------------|
| 2013 | Word2Vec | 100-300 | Static: one vector per word |
| 2014 | GloVe | 100-300 | Static: global co-occurrence matrix |
| 2018+ | Contextual (BERT, GPT) | 768-12288 | Dynamic: vector changes with context |


Modern LLMs use **contextual embeddings** — the same word "bank" gets different vectors in "river bank" vs. "bank account". This is crucial: the embedding layer gives the initial representation, then attention layers refine it with context.

**Embedding dimensions in modern models:**
- GPT-3: d = 12288
- Llama 2 (70B): d = 8192
- GPT-4 (estimated): d ≈ 12288+

**The embedding matrix:** E ∈ ℝ^(V × d) where V = vocabulary size (32K-100K+ tokens), d = embedding dimension. This single matrix has V × d parameters — often 300M+ parameters just for embeddings.

### Connection to NumGLUE (Your Research)

> **Research connection:** In NumGLUE (Mishra, Sachdeva et al., 2022), one finding was that models struggle with numerical reasoning partly because standard tokenization and embedding schemes don't represent number magnitudes well. The number "1000" and "1001" might have very different embeddings despite being semantically close numerically. This is a fundamental limitation of how LLMs embed tokens.
>
> [Paper: Mishra et al., 2022, "NumGLUE: A Suite of Fundamental yet Challenging Mathematical Reasoning Tasks," arXiv:2204.05660, ACL 2022]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Word2Vec analogy diagram] | Mikolov et al. 2013, Figure 2 | King-Queen, Man-Woman vector arithmetic |
| ![2D/3D word clusters] | Create (use t-SNE/PCA projection) | Show semantic clusters in reduced space |
| ![Embedding dimension visualization] | Create custom | Show how 4096 dims compress meaning |


- **[SOURCE: Mikolov et al., 2013, "Efficient Estimation of Word Representations in Vector Space," arXiv:1301.3781]** — Original Word2Vec paper with vector arithmetic examples
- **[SOURCE: 3Blue1Brown, "But what is a GPT?" (2024)](https://www.3blue1brown.com/lessons/gpt/)** — Excellent visualization of embeddings as high-dimensional vectors
- **[BLOG: Chris Olah, "Deep Learning, NLP, and Representations" (2014)](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)** — Classic visual explanation of embedding spaces
- **[DIAGRAM TO CREATE]** 2D projection showing: animal cluster, country cluster, emotion cluster — with arrows showing relational directions

---

## Slide 5: The Attention Mechanism — Who Should I Listen To?

### Core Concept
Given a token, which other tokens in the context matter most for predicting what comes next? Attention is the mechanism that answers this by computing a relevance score between every pair of tokens.

### The 3Blue1Brown Intuition (Key Framing)

Credit: [3Blue1Brown, "Attention in transformers, step-by-step" (2024)](https://www.3blue1brown.com/lessons/attention)

The Q, K, V framework:
- **Query (Q):** "What am I looking for?" — Each token asks a question about what information it needs
- **Key (K):** "What do I offer?" — Each token advertises what information it contains
- **Value (V):** "What information do I actually give if you pick me?" — The actual content transferred

### Extended Mystery Novel Analogy

> **Q** is the detective's question: "Who had a motive?"
> **K** is which clues respond to that question — some clues are about motive, others about alibis
> **V** is what those relevant clues actually reveal when you examine them closely
>
> The dot product Q·K measures how well a question matches a clue. High score = this clue is relevant to this question. The softmax normalizes these scores into a probability distribution — a **weighted vote** on which clues matter most.


### Technical Depth

**The Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Step by step:
1. **QK^T** — Dot product between all queries and keys. Result: n×n matrix of "relevance scores"
2. **/ √d_k** — Scale factor prevents softmax saturation for large d_k (dimension of keys)
3. **softmax** — Converts raw scores into probability distribution (rows sum to 1)
4. **× V** — Weight the values by their relevance scores → weighted sum of information

**Where Q, K, V come from:**
```
Q = X · W_Q    (linear projection of input)
K = X · W_K    (linear projection of input)
V = X · W_V    (linear projection of input)
```

The W_Q, W_K, W_V matrices are learned parameters — the model learns WHAT to ask for (Q), WHAT to advertise (K), and WHAT to provide (V).

**Why √d_k matters:**
- Without scaling, dot products grow with d_k → softmax becomes nearly one-hot
- With scaling, gradients remain healthy even for large dimensions
- Original paper: d_k = 64 per head

### Connection to Instruction-Following (Your Mentor's Research)

> **Research connection:** Swaroop Mishra's work on instruction-tuning (Natural Instructions, 2021; Cross-Task Generalization, ACL 2022) showed that instruction-tuned models develop better attention patterns — they learn to attend to the task instruction and relevant input portions more effectively than base models. The attention mechanism is what enables instruction-following: Q from the current generation position queries K from the instruction text to route relevant information.
>
> [Paper: Mishra et al., 2022, "Cross-Task Generalization via Natural Language Crowdsourcing Instructions," ACL 2022, arXiv:2104.08773]


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Scaled dot-product attention] | Vaswani et al. 2017, Figure 2 (left) | The canonical Q,K,V flow diagram |
| ![Attention heatmap] | 3Blue1Brown "Attention" (2024) | Shows which words attend to which |
| ![Q,K,V intuition diagram] | Create custom (3B1B style) | Question/Key/Value with detective analogy |
| ![Softmax visualization] | Create custom | How raw scores become a probability distribution |

- **[SOURCE: Vaswani et al., 2017, "Attention Is All You Need," arXiv:1706.03762, Figure 2]** — Canonical attention mechanism diagram
- **[SOURCE: 3Blue1Brown, "Attention in transformers, step-by-step" (2024)](https://www.3blue1brown.com/lessons/attention)** — Step-by-step visual walkthrough of Q, K, V
- **[SOURCE: 3Blue1Brown Substack, "Visualizing Attention" (2024)](https://3blue1brown.substack.com/p/visualizing-attention)** — Companion written explanation
- **[DIAGRAM TO CREATE]** Detective analogy: Q = detective's question bubble, K = clue response indicators, V = actual revealed information

---

## Slide 6: Multi-Head Attention — Multiple Perspectives

### Core Concept
One attention head can only capture one type of relationship at a time. Multiple heads run in parallel, each learning to look for **different patterns** in the same input.

### Key Intuitions for Students

Think of it as having **8-128 detectives** each following a different line of investigation simultaneously:
- Head 1: Tracks **syntactic relationships** ("what grammatically connects to what")
- Head 2: Tracks **coreference** ("who does 'they' refer to?")
- Head 3: Tracks **semantic similarity** ("which words mean related things")
- Head 4: Tracks **positional patterns** ("what's nearby")
- Head 5: Tracks **negation scope** ("what does 'not' apply to")
- etc.

No head is explicitly programmed for these roles — they **emerge** from training.


### Technical Depth

**Multi-Head Attention formula:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

**Dimensions:**
- d_model = 512 (original), 4096-12288 (modern)
- h = 8 (original), 32-128 (modern)
- d_k = d_v = d_model / h = 64 per head (original)

**What this means computationally:**
- Each head operates on a **lower-dimensional subspace** (d_model/h dimensions)
- All heads run in parallel → same compute as a single large attention
- W_O projects concatenated outputs back to d_model dimensions

**Evidence heads specialize (from interpretability research):**
- Clark et al. (2019): Found specific heads for syntax (subject-verb), positional (adjacent tokens), and delimiter patterns
- Voita et al. (2019): Showed that many heads can be pruned without quality loss — a few heads carry most information

### Head Count Evolution

| Model | Year | Heads | d_model | d_per_head |
|-------|------|-------|---------|-----------|
| Original Transformer | 2017 | 8 | 512 | 64 |
| GPT-2 | 2019 | 12 | 768 | 64 |
| GPT-3 | 2020 | 96 | 12288 | 128 |
| Llama 2 (70B) | 2023 | 64 | 8192 | 128 |
| Llama 3 (405B) | 2024 | 128 | 16384 | 128 |

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Multi-head attention diagram] | Vaswani et al. 2017, Figure 2 (right) | Split → parallel heads → concat |
| ![Head specialization] | Clark et al. 2019 | Individual head attention patterns |
| ![Parallel detectives] | Create custom | 8 detectives each following different leads |

- **[SOURCE: Vaswani et al., 2017, Figure 2 (right panel)]** — Multi-head attention architecture
- **[SOURCE: Clark et al., 2019, "What Does BERT Look At?" arXiv:1906.04341]** — Head specialization evidence
- **[DIAGRAM TO CREATE]** Split/parallel/concat flow with example attention patterns per head

---

## Slide 7: The MLP Layers — Where Facts Live

### Core Concept
Attention **routes information** between tokens. MLP (feed-forward) layers **store and transform knowledge**. Together they form the two complementary operations in each transformer block.

### Key Intuitions for Students

- Attention = the **wiring/routing** — deciding what information flows where
- MLP = the **content/memory** — the actual facts and transformations being applied
- Think of MLPs as the model's **"memory of facts and nuance"**

Research has shown:
- "Eiffel Tower is in ___" → The answer "Paris" is encoded in specific MLP neurons
- You can literally find neurons that activate for specific factual associations
- Editing those neurons changes the model's factual outputs

### Technical Depth

**MLP Structure (per position, per layer):**
```
MLP(x) = W₂ · σ(W₁ · x + b₁) + b₂
```
Where:
- W₁ ∈ ℝ^(d_ff × d_model) — "up projection" (expand to 4x)
- W₂ ∈ ℝ^(d_model × d_ff) — "down projection" (compress back)
- σ = activation function (ReLU original, GeLU/SwiGLU modern)
- d_ff = 4 × d_model typically (2048 in original, 16384+ in modern)

**Why MLPs store knowledge (key papers):**

1. **Knowledge Neurons (Dai et al., 2022):** Identified specific neurons in MLP layers that correspond to factual knowledge. Suppressing these neurons removes the fact; amplifying them strengthens it.
   - [Paper: arXiv:2104.08696]


2. **FFN as Key-Value Memories (Geva et al., 2021):** Showed that each row of W₁ acts as a "key" (pattern detector) and the corresponding column of W₂ acts as a "value" (information to output). The MLP is essentially a large, learned key-value store.
   - [Paper: arXiv:2012.14913]

3. **Constructing Fact-Storing MLPs (Hazy Research / Stanford, 2025):** Recent work explicitly constructing MLPs that implement key-value fact mappings, confirming the theoretical framework.
   - [Source: hazyresearch.stanford.edu]

**MLP vs. Attention — complementary roles:**
```
One Transformer Block:
  x → LayerNorm → Multi-Head Attention → + residual → LayerNorm → MLP → + residual → output
       (routing)                                         (knowledge)
```

### Connection to Data Quality Research (Your Research)

> **Research connection:** Bhavdeep Sachdeva and Swaroop Mishra's work on "Is High Quality Data All We Need?" (2022) and "Generalized but not Robust?" (ACL Findings 2022) directly relates to what MLPs learn. If training data contains artifacts or low-quality patterns, the MLP layers will memorize those artifacts as "facts." This explains why models trained on cleaner data show better generalization — the MLP weights encode more genuine knowledge rather than spurious correlations.
>
> The VAIDA benchmark creation paradigm (Arunkumar, Mishra, Sachdeva et al., EACL 2023) addresses this at the source: ensuring training data quality so that what MLPs learn is actually correct.
>
> [Paper: Sachdeva et al., 2022, "Generalized but not Robust?" ACL Findings]
> [Paper: Sachdeva & Mishra, 2022, "Is High Quality Data All We Need?" arXiv:2203.06404]
> [Paper: Arunkumar, Mishra, Sachdeva et al., 2023, "VAIDA," EACL, arXiv:2302.04434]


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Attention vs MLP roles] | Create custom | Side-by-side: attention=routing, MLP=memory |
| ![Key-value memory diagram] | Geva et al. 2021, Figure 1 | FFN as key-value lookup |
| ![Knowledge neuron activation] | Dai et al. 2022 | Specific neurons for specific facts |
| ![Transformer block] | Create custom | Full block showing Attn→MLP interplay |

- **[SOURCE: Dai et al., 2022, "Knowledge Neurons in Pretrained Transformers," arXiv:2104.08696]** — Neuron-level fact storage
- **[SOURCE: Geva et al., 2021, "Transformer Feed-Forward Layers Are Key-Value Memories," arXiv:2012.14913]** — MLP as key-value store
- **[SOURCE: Hazy Research, Stanford, 2025, "Our In-House Recipe for Juicy, Fact-Stuffed MLPs"](https://hazyresearch.stanford.edu/blog/2025-12-01-mlps-p2)** — Latest understanding of MLP fact storage
- **[DIAGRAM TO CREATE]** Simple two-panel: (1) Attention weights showing routing, (2) MLP activation showing fact retrieval

---

## Slide 8: The Full Transformer Architecture

### Core Concept
Stack attention + MLP + residual connections + layer normalization = one transformer block. Repeat 32-128 times = a full LLM.

### Key Points for Students

**One block:**
```
Input → Layer Norm → Multi-Head Attention → + → Layer Norm → MLP → + → Output
                                           ↑                       ↑
                                      (residual)              (residual)
```

**Full model (decoder-only, GPT-style):**
```
Token IDs → Embedding + Position → [Block₁] → [Block₂] → ... → [Block_N] → LayerNorm → Linear → Softmax → Next Token Probs
```


### Architecture Evolution

| Architecture | Used By | Key Difference |
|-------------|---------|----------------|
| Encoder-Decoder | Original Transformer, T5, BART | Separate encoder and decoder stacks |
| Encoder-only | BERT, RoBERTa | Bidirectional; good for understanding, not generation |
| **Decoder-only** | **GPT, Llama, Claude, Gemini** | **Autoregressive; what all modern LLMs use** |

**Why decoder-only won:**
- Simpler architecture (one stack, not two)
- Naturally autoregressive (generates token by token)
- Scales better empirically
- Causal mask prevents "cheating" — can only attend to past tokens

**Key components explained:**
1. **Positional Encoding:** Injects position information (sinusoidal original → RoPE modern)
2. **Residual Connections:** Allow gradients to flow through deep stacks (critical for training 100+ layers)
3. **Layer Normalization:** Stabilizes activations (Pre-LN in modern models vs. Post-LN in original)
4. **Causal Mask:** Upper-triangular mask prevents attending to future tokens

### Model Scale Comparison

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|-----------|
| Original Transformer | 6+6 | 512 | 8 | ~65M |
| GPT-2 | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |
| Llama 2 (70B) | 80 | 8192 | 64 | 70B |
| Llama 3 (405B) | 126 | 16384 | 128 | 405B |

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Original Transformer Figure 1] | Vaswani et al. 2017 | The iconic encoder-decoder diagram |
| ![Decoder-only adaptation] | Create custom | Adapted for GPT-style (decoder only, causal mask) |
| ![Scale comparison] | Create custom | Stacked blocks visualizing depth differences |

- **[SOURCE: Vaswani et al., 2017, Figure 1]** — The canonical architecture diagram (adapt for decoder-only)
- **[DIAGRAM TO CREATE]** Modern decoder-only version with annotations showing RoPE, Pre-LN, SwiGLU

---

## Slide 9: Why Transformers Won (But They're Not the End)

### Core Concept
Transformers dominate because of parallelization and direct long-range connections, but they have fundamental limitations that motivate ongoing research.

### What Made Transformers Win

**vs. RNNs:**
| Property | RNNs | Transformers |
|----------|------|-------------|
| Training | Sequential (slow) | Fully parallel |
| Long-range deps | O(n) path length | O(1) path length |
| Gradient flow | Vanishing/exploding | Stable (residuals + LN) |
| BLEU (EN→DE) | ~24 (2016 SOTA) | 28.4 (Vaswani 2017) |
| Training time | Weeks | 3.5 days (8 P100s) |

**Key results from original paper:**
- BLEU 28.4 English→German (beat all previous models)
- BLEU 41.8 English→French (new SOTA at 1/4 training cost)
- Training: 3.5 days on 8 P100 GPUs vs. weeks for RNNs

### The Limitations (Motivating Next Frontier)

1. **O(n²) attention complexity:**
   - 100K tokens → 10 billion entries in attention matrix
   - Memory grows quadratically with sequence length
   - Practical limit on context windows (even with tricks)

2. **Autoregressive prediction ≠ understanding/planning:**
   - LLMs predict one token at a time, left to right
   - No explicit planning or "thinking ahead"
   - Can't revise earlier outputs based on later reasoning
   - This is Yann LeCun's core critique

3. **No explicit world model:**
   - Knowledge is implicitly encoded in weights
   - Cannot be inspected, verified, or updated easily
   - Leads to hallucination and inconsistency


### Connection to Reasoning Research (Your Mentor's Research)

> **Research connection:** Swaroop Mishra's reasoning work directly addresses the autoregressive limitations of transformers:
>
> - **Reframing (2021):** Showed that HOW you frame instructions dramatically affects model performance — because autoregressive models can only condition on what came before
> - **Step-Back Prompting (2023, ICLR 2024):** Enables LLMs to "step back" and derive abstract principles before reasoning — a workaround for the lack of planning
> - **Self-Discover (2024, Google DeepMind):** LLMs compose their own reasoning structures — addressing the limitation that autoregressive generation can't naturally decompose problems
>
> These methods are all **patches** to the fundamental limitation of next-token prediction. They work, but they motivate the question: can we build architectures that plan natively?
>
> [Paper: Mishra et al., 2021, "Reframing Instructional Prompts," arXiv:2109.07830]
> [Paper: Zheng, Mishra et al., 2023, "Take a Step Back," arXiv:2310.06117, ICLR 2024]
> [Paper: Zhou, Mishra et al., 2024, "Self-Discover," arXiv:2402.03620, Google DeepMind]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Parallel vs sequential] | Create custom | RNN sequential chain vs. Transformer parallel |
| ![O(n²) scaling curve] | Create custom | Memory/compute vs. sequence length |
| ![Limitation diagram] | Create custom | Three boxes: O(n²), no planning, no world model |

- **[SOURCE: Vaswani et al., 2017, Table 1]** — Complexity comparison across architectures
- **[DIAGRAM TO CREATE]** Three-panel showing: (1) O(n²) growth, (2) autoregressive left-to-right limitation, (3) "what's next?" arrow pointing to JEPA

---

## Slide 10: The Frontier — Beyond Token Prediction (JEPA)

### Core Concept
LLMs predict the next token. JEPA (Joint Embedding Predictive Architecture) proposes predicting in **representation space** instead — learning abstract world models rather than surface-level token sequences.

### Yann LeCun's Argument

From "A Path Towards Autonomous Machine Intelligence" (2022):

1. **Humans don't predict pixels or words** — we predict at an abstract level
2. **Generative models waste capacity** on predicting irrelevant details
3. **Planning requires a world model** — predicting consequences of actions in abstract space
4. **Token prediction is necessary but not sufficient** for intelligence

### How JEPA Works (Simplified)

```
Traditional LLM (Autoregressive):
  Context tokens → Predict NEXT TOKEN (in vocabulary space)
  "The cat sat on the" → P("mat") = 0.3, P("chair") = 0.2, ...

JEPA (Joint Embedding Predictive):
  Context representation → Predict TARGET REPRESENTATION (in latent space)
  [abstract features of scene] → [predicted abstract features of next state]
```

**Key difference:** JEPA discards irrelevant details and predicts only the **abstract meaning** of the target, not its exact surface form.

### I-JEPA (Images, 2023)
- Given an image with patches masked, predict the **representation** of the masked patches
- Not pixel prediction (like MAE) — representation prediction
- Learns semantic features without needing pixel-level detail
- Outperforms pixel-reconstruction methods on downstream tasks
- [Paper: Assran et al., 2023, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," arXiv:2301.08243]


### V-JEPA (Video, 2024)
- Extends I-JEPA to video — predicts representations of masked spatiotemporal regions
- Learns temporal dynamics without generating pixels
- Captures motion, causality, and object persistence in abstract space
- Meta AI: "The next step toward advanced machine intelligence"
- [Source: Meta AI Blog, "V-JEPA" (Feb 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)

### VL-JEPA (Vision-Language, 2024-2025)
- Combines vision and language in a single JEPA framework
- Predicts continuous embeddings of target text instead of autoregressively generating tokens
- Focuses on task-relevant semantics while abstracting away surface-level linguistic variability
- [Paper: arXiv:2512.10942]

### Why JEPA Matters for This Presentation

**As a frontier for students to understand:**
1. It names the **specific limitation** of LLMs (next-token prediction → no world model)
2. It proposes a **concrete alternative paradigm** (predict in representation space)
3. It's backed by **real papers with results** (I-JEPA, V-JEPA) — not just speculation
4. It connects to the **broader question** of what intelligence requires beyond pattern matching

**What JEPA is NOT (honest framing):**
- Not a replacement for transformers today
- Not yet proven at LLM scale for language tasks
- Not guaranteed to solve the alignment/safety problem
- Still early research — promising direction, not a solution

### Connection to LLM-JEPA (Emerging Work)

Recent work (arXiv:2509.14252) combines LLM training objectives with JEPA principles, showing improvements over standard autoregressive training across multiple model families (Llama3, Gemma2, Olmo). This suggests JEPA ideas may eventually merge back into the LLM paradigm rather than replacing it entirely.

### Connection to Self-Improvement Research (Your Mentor's Work)

> **Research connection:** Swaroop Mishra's work on Self-Instruct (2023) and Self-Discover (2024) represents an intermediate step between pure autoregressive models and world-model architectures like JEPA. These methods enable LLMs to:
>
> - **Self-Instruct:** Generate their own training data — a form of "self-play" that approaches JEPA's self-supervised learning philosophy
> - **Self-Discover:** Compose reasoning structures before solving — mimicking the "predict in abstract space first, then execute" pattern that JEPA formalizes
>
> Both represent the field moving toward models that build internal representations and plans, even within the autoregressive framework.
>
> [Paper: Wang, Mishra et al., 2023, "Self-Instruct," ACL 2023, arXiv:2212.10560]
> [Paper: Zhou, Mishra et al., 2024, "Self-Discover," arXiv:2402.03620]


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Autoregressive vs JEPA] | Create custom | Side-by-side: token prediction vs. latent prediction |
| ![I-JEPA architecture] | Assran et al. 2023, Figure 1 | Context encoder → predictor → target encoder |
| ![V-JEPA masking] | Meta AI Blog (2024) | Video with masked spatiotemporal regions |
| ![World model diagram] | LeCun 2022 position paper | The full cognitive architecture proposal |

- **[SOURCE: LeCun, 2022, "A Path Towards Autonomous Machine Intelligence" (position paper)](https://openreview.net/forum?id=BZ5a1r-kVsf)** — Full architecture proposal
- **[SOURCE: Assran et al., 2023, arXiv:2301.08243, Figure 1]** — I-JEPA architecture diagram
- **[SOURCE: Meta AI Blog, V-JEPA (Feb 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)** — V-JEPA visual explanation
- **[BLOG: Rohit Bandaru, "Deep Dive into Yann LeCun's JEPA"](https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/)** — Accessible explanation with diagrams
- **[DIAGRAM TO CREATE]** Two-track comparison: (top) LLM token-by-token prediction, (bottom) JEPA abstract-space prediction with "irrelevant details discarded" annotation

---

## Summary: Research Papers Referenced in Section 1

### Foundational Papers (External)

| # | Paper | Authors | Year | ArXiv | Relevance |
|---|-------|---------|------|-------|-----------|
| 1 | Efficient Estimation of Word Representations | Mikolov et al. | 2013 | 1301.3781 | Embeddings/Word2Vec |
| 2 | Attention Is All You Need | Vaswani et al. | 2017 | 1706.03762 | Transformer architecture |
| 3 | What Does BERT Look At? | Clark et al. | 2019 | 1906.04341 | Head specialization |
| 4 | Transformer FFN as Key-Value Memories | Geva et al. | 2021 | 2012.14913 | MLP knowledge storage |
| 5 | Knowledge Neurons in Pretrained Transformers | Dai et al. | 2022 | 2104.08696 | Neuron-level facts |
| 6 | A Path Towards Autonomous Machine Intelligence | LeCun | 2022 | (position paper) | JEPA framework |
| 7 | I-JEPA | Assran et al. | 2023 | 2301.08243 | Image JEPA |
| 8 | Mamba | Gu & Dao | 2023 | 2312.00752 | Linear-time alternative |


### Your Research & Your Mentor's Research (Placed in Timeline)

| # | Paper | Authors | Year | Venue | Connection to Section 1 |
|---|-------|---------|------|-------|------------------------|
| 1 | Natural Instructions | Mishra et al. | 2021 | — | Pioneered instruction-tuning; showed attention can learn to follow instructions |
| 2 | Reframing Instructional Prompts | Mishra et al. | 2021 | ACL 2022 | How prompt framing affects autoregressive model behavior |
| 3 | NumGLUE | Mishra, Sachdeva et al. | 2022 | ACL 2022 | Reveals embedding limitations for numerical reasoning |
| 4 | Cross-Task Generalization | Mishra et al. | 2022 | ACL 2022 | Instruction-following generalization via attention |
| 5 | Super-NaturalInstructions | Mishra et al. | 2022 | EMNLP 2022 | 1,616 tasks — scaling instruction data |
| 6 | LILA | Mishra et al. | 2022 | EMNLP 2022 | Unified math reasoning benchmark |
| 7 | Generalized but not Robust? | Sachdeva, Mishra et al. | 2022 | ACL Findings | What MLPs learn from noisy data |
| 8 | "Is High Quality Data All We Need?" | Sachdeva, Mishra et al. | 2022 | arXiv:2203.06404 | Data quality → MLP knowledge quality |
| 9 | VAIDA | Arunkumar, Mishra, Sachdeva et al. | 2023 | EACL 2023 | Benchmark creation quality → training quality |
| 10 | Self-Instruct | Wang, Mishra et al. | 2023 | ACL 2023 | Self-generated training data |
| 11 | InstructABSA | Scaria, Mishra et al. | 2023 | NAACL 2024 | Instruction-tuning for sentiment analysis |
| 12 | Pretrained Transformers Not Always Robust | Sachdeva, Mishra et al. | 2023 | arXiv:2210.07663 | Robustness limitations of transformer representations |
| 13 | Step-Back Prompting | Zheng, Mishra et al. | 2023 | ICLR 2024 | Reasoning via abstraction — workaround for autoregressive limits |
| 14 | TarGEN | Gupta, Mishra et al. | 2023 | arXiv:2310.17876 | Seedless synthetic data generation |
| 15 | Self-Discover | Zhou, Mishra et al. | 2024 | Google DeepMind | Self-composed reasoning structures |
| 16 | IMO 2024 Silver Medal | Google DeepMind (incl. Mishra) | 2024 | — | Math reasoning system (AlphaProof) |

---

## Visual Index: All Images & Diagrams for Section 1

### From Original Papers (Cite & Reference Directly)

| # | Visual | Paper | Figure # | Slide |
|---|--------|-------|----------|-------|
| 1 | Scaled dot-product attention flow | Vaswani et al. 2017 | Fig 2 (left) | Slide 5 |
| 2 | Multi-head attention architecture | Vaswani et al. 2017 | Fig 2 (right) | Slide 6 |
| 3 | Full Transformer architecture | Vaswani et al. 2017 | Fig 1 | Slide 8 |
| 4 | Word2Vec analogy vectors | Mikolov et al. 2013 | Fig 2 | Slide 4 |
| 5 | FFN key-value memory diagram | Geva et al. 2021 | Fig 1 | Slide 7 |
| 6 | Knowledge neuron activation | Dai et al. 2022 | Fig 2 | Slide 7 |
| 7 | I-JEPA architecture | Assran et al. 2023 | Fig 1 | Slide 10 |

### From Blogs & Videos (Reference with Attribution)

| # | Visual | Source | URL | Slide |
|---|--------|--------|-----|-------|
| 1 | Embedding space visualization | 3Blue1Brown "GPT" (2024) | 3blue1brown.com/lessons/gpt | Slide 4 |
| 2 | Q,K,V step-by-step | 3Blue1Brown "Attention" (2024) | 3blue1brown.com/lessons/attention | Slide 5 |
| 3 | Attention heatmap | 3Blue1Brown "Attention" (2024) | 3blue1brown.com/lessons/attention | Slide 5 |
| 4 | V-JEPA demo | Meta AI Blog (2024) | ai.meta.com/blog/v-jepa-... | Slide 10 |
| 5 | JEPA deep dive diagrams | Rohit Bandaru blog | rohitbandaru.github.io/blog/JEPA-Deep-Dive | Slide 10 |
| 6 | Word embedding clusters | Chris Olah blog (2014) | colah.github.io/posts/2014-07-NLP-RNNs-Representations | Slide 4 |

### Diagrams to Create

| # | Visual | Description | Slide | Suggested Tool |
|---|--------|-------------|-------|---------------|
| 1 | Mystery novel analogy | Story → Clues highlighted → "The murderer is ___" + probability bars | Slide 3 | Excalidraw |
| 2 | Token prediction | Sentence with blanked word, probability distribution bars | Slide 3 | PowerPoint |
| 3 | Embedding space (2D) | t-SNE/PCA clusters: animals, countries, emotions + direction arrows | Slide 4 | Python matplotlib |
| 4 | Q,K,V detective analogy | Detective question bubble → clue responses → revealed info | Slide 5 | Excalidraw |
| 5 | Softmax visualization | Raw scores → normalized probability distribution | Slide 5 | Python |
| 6 | Multiple detectives | 8 detectives each following different leads in parallel | Slide 6 | Excalidraw |
| 7 | Attention vs MLP roles | Two-panel: routing vs. memory | Slide 7 | Excalidraw |
| 8 | Full transformer block | Annotated modern decoder-only block | Slide 8 | draw.io/TikZ |
| 9 | RNN vs Transformer | Sequential chain vs. parallel all-pairs | Slide 9 | Excalidraw |
| 10 | O(n²) growth curve | Memory/compute vs. sequence length | Slide 9 | Python matplotlib |
| 11 | Three limitations panel | O(n²), no planning, no world model | Slide 9 | PowerPoint |
| 12 | Autoregressive vs JEPA | Token-by-token vs. latent prediction side-by-side | Slide 10 | Excalidraw |
| 13 | Research timeline | Your papers placed on a 2017-2024 timeline with major LLM milestones | All | draw.io/Mermaid |

---

## Appendix: Key Quotes & Numbers for Slides

### For Slide 3 (What is an LLM)
- "Language modeling is the task of assigning a probability to a sequence of words" — foundation of all LLMs
- GPT-3 vocabulary: 50,257 tokens; each prediction is a choice among 50K+ options
- Modern context windows: 4K (GPT-3) → 128K (GPT-4) → 1M+ (Gemini 1.5)

### For Slide 4 (Embeddings)
- Word2Vec (2013): "King − Man + Woman ≈ Queen" (Mikolov et al., arXiv:1301.3781)
- Modern embedding dimensions: GPT-3 = 12,288; Llama 2 = 8,192
- Embedding matrix for GPT-3: 50,257 tokens × 12,288 dims = ~617M parameters just for embeddings

### For Slide 5 (Attention)
- Original paper d_k = 64, scaling factor √64 = 8
- Attention matrix for 4K context: 4,096 × 4,096 = 16.7M entries per head per layer
- "Attention is all you need" — the insight that attention alone (without recurrence) suffices

### For Slide 6 (Multi-Head)
- GPT-3: 96 heads × 96 layers = 9,216 attention operations per forward pass
- Each head sees 128 dimensions (12,288 / 96)
- Clark et al. (2019): found syntax heads, positional heads, delimiter heads in BERT

### For Slide 7 (MLP)
- GPT-3 MLP: W₁ is 12,288 × 49,152 and W₂ is 49,152 × 12,288 per layer
- That's ~1.2B parameters per MLP layer × 96 layers = ~115B of GPT-3's 175B params are in MLPs
- Dai et al. (2022): Identified specific neurons for "Eiffel Tower → Paris" type facts

### For Slide 8 (Full Architecture)
- Original: 6 layers, 512 dims, 65M params, 3.5 days training
- GPT-3: 96 layers, 12,288 dims, 175B params, ~$4.6M training cost
- Llama 3 (405B): 126 layers, 16,384 dims, estimated $100M+ training

### For Slide 9 (Why Transformers Won)
- RNN training: weeks; Transformer training: 3.5 days (8 P100 GPUs) for same task
- BLEU improvement: 24 (RNN SOTA) → 28.4 (Transformer) = +18% on EN→DE
- O(n²) problem: 128K context → 16.4 billion attention entries per head per layer

### For Slide 10 (JEPA)
- LeCun (2022): "Current AI systems lack the ability to plan, to reason by analogy, or to transfer learning"
- I-JEPA (2023): Matches or exceeds pixel-reconstruction methods on ImageNet
- V-JEPA (2024): Learns video dynamics without any pixel-level generation
- LLM-JEPA (2025): Outperforms standard LLM objectives across Llama3, Gemma2, Olmo families

---

## Next Steps

1. **Expand this format** to Sections 2-8 (Scaling, Alignment, Fine-Tuning, RAG/Agents, Architectures, Hype vs Reality, Call to Action)
2. **Download/screenshot** key figures from papers for actual slides
3. **Create custom diagrams** (start with the mystery novel analogy and embedding space)
4. **Record which of your papers** to mention verbally vs. show on slides
5. **Map your mentor's IMO 2024 contribution** to the "reasoning" narrative in Sections 3 and 9

---

*Document created: July 2, 2026*
*Last updated: July 2, 2026*
