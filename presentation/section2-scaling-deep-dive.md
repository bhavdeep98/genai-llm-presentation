# Section 2: Scaling — The GPT Story — Deep Dive Research Document

## Purpose
This document expands Section 2 (Scaling — The GPT Story) into research-paper-level depth. It covers the pre-train + fine-tune paradigm, the GPT progression, scaling laws, the Chinchilla moment, and how instruction-tuning (pioneered by Swaroop Mishra) became the critical bridge between raw scale and usable capabilities.

---

## Timeline: Scaling Milestones + Your Research

| Year | Milestone | Authors | Key Insight |
|------|-----------|---------|-------------|
| 2018 | GPT-1 (117M) | Radford et al. | Pre-train + fine-tune paradigm |
| 2019 | GPT-2 (1.5B) | Radford et al. | Zero-shot via scale |
| 2020 | GPT-3 (175B) | Brown et al. | In-context few-shot learning |
| 2020 | Scaling Laws | Kaplan et al. | Power-law relationships |
| 2021 | **Natural Instructions** | **Swaroop Mishra** et al. | First paper on instruction-tuning — the missing link |
| 2021 | **Reframing Prompts to GPTk's Language** | **Swaroop Mishra** et al. | +12.5% GPT-3 performance via prompt reframing |
| 2022 | Chinchilla (70B) | Hoffmann et al. | Compute-optimal: balance model size & data |
| 2022 | **Cross-Task Generalization** | **Swaroop Mishra** et al. | ACL 2022 — 19% gain from instruction-following |
| 2022 | **Super-NaturalInstructions** | **Mishra**, Wang et al. | EMNLP 2022 — 1,616 tasks, Tk-Instruct |
| 2022 | **NumGLUE** | **Mishra, Sachdeva** et al. | ACL 2022 — Math reasoning benchmark (humans 46.4% ahead) |
| 2022 | **"Is High Quality Data All We Need?"** | **Sachdeva, Mishra** et al. | Data quality > data quantity hypothesis |
| 2022 | **LILA** | **Mishra** et al. | EMNLP 2022 — 20-dataset unified math benchmark |
| 2023 | Llama 2 (7B-70B) | Meta | Open-source scaling + Chinchilla-optimal training |
| 2023 | **Instruction-Example Equivalence** | **Mishra** et al. | One instruction ≈ 200 data samples |
| 2023 | **Self-Instruct** | Wang, **Mishra** et al. | LLMs bootstrap their own instruction data |
| 2023 | GPT-4 (~1.8T MoE) | OpenAI | Multimodal, MoE architecture |
| 2023 | "Emergent Abilities a Mirage?" | Schaeffer et al. | Metric choice creates illusion of emergence |
| 2024 | Llama 3 (8B-405B) | Meta | 15T tokens training, open frontier model |
| 2024 | **Self-Discover** | Zhou, **Mishra** et al. | DeepMind — reasoning structures at scale |

---

## Slide 11: The Pre-train + Fine-tune Paradigm (GPT-1)

### Core Concept
GPT-1 established the two-phase approach that underlies all modern LLMs: (1) learn general language from massive unlabeled text, then (2) adapt to specific tasks with labeled data.

### Why This Was Revolutionary

**Before GPT-1 (2018):**
- Each NLP task required its own architecture and labeled dataset
- Models trained from scratch per task (sentiment, QA, summarization)
- Limited labeled data → limited performance
- No transfer of knowledge between tasks

**After GPT-1:**
- One pre-trained model → many tasks via fine-tuning
- Unsupervised pre-training provides a "foundation" of language understanding
- Fine-tuning on small labeled datasets works because the foundation is strong
- Knowledge transfers: learning from books helps with sentiment classification

### Technical Details

**Architecture:**
- 12-layer decoder-only Transformer (not encoder-decoder)
- 117M parameters
- 768-dim embeddings, 12 attention heads
- 512-token context window
- BPE tokenization (~40K vocabulary)

**Training:**
- Pre-training: BooksCorpus (~7,000 unpublished books, ~800M words)
- Objective: Standard next-token prediction (causal LM)
- Fine-tuning: Add task-specific linear head, fine-tune all parameters

**Results:**
- Beat task-specific models on 9/12 NLP benchmarks
- First evidence that generative pre-training transfers to discriminative tasks


### The Paradigm Shift Visualized

```
Pre-GPT-1 Era:
  Task A Data → Model A (from scratch) → Task A Output
  Task B Data → Model B (from scratch) → Task B Output
  Task C Data → Model C (from scratch) → Task C Output

GPT-1 Era:
  Unlabeled Text → Pre-trained LM → Fine-tune (Task A) → Task A Output
                                   → Fine-tune (Task B) → Task B Output
                                   → Fine-tune (Task C) → Task C Output
```

### Connection to Instruction-Tuning (Your Mentor's Pioneering Work)

> **Research connection:** GPT-1 established that fine-tuning works. But fine-tuning requires task-specific labeled data for EACH task. Swaroop Mishra's **Natural Instructions (2021)** — the first paper on instruction-tuning — asked a different question: what if you fine-tune on the *instructions themselves* rather than just input-output pairs?
>
> This was the critical bridge: instead of fine-tuning GPT for "sentiment analysis" or "QA" separately, you fine-tune it to *follow instructions* — then it can do ANY task described by an instruction. This insight eventually led to ChatGPT and the entire instruction-following paradigm.
>
> **The evolution:**
> - GPT-1 (2018): Fine-tune per task (needs task-specific data)
> - Natural Instructions (2021): Fine-tune to follow instructions (needs instruction data)
> - InstructGPT/ChatGPT (2022): RLHF on top of instruction-tuning (needs human preferences)
>
> [Paper: Mishra et al., 2021, "Cross-Task Generalization via Natural Language Crowdsourcing Instructions," arXiv:2104.08773]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Two-phase training] | Create custom | Pre-train (unlabeled) → Fine-tune (labeled) pipeline |
| ![Before/After comparison] | Create custom | Task-specific models vs. one foundation + fine-tune |
| ![GPT-1 architecture] | Radford et al. 2018, Figure 1 | Transformer decoder with task heads |

- **[SOURCE: Radford et al., 2018, "Improving Language Understanding by Generative Pre-Training," Figure 1]**
- **[DIAGRAM TO CREATE]** Three-era timeline: task-specific → pre-train+fine-tune → instruction-following

---

## Slide 12: Scale Unlocks Capabilities (GPT-2 → GPT-3)

### Core Concept
As models scale from 117M → 1.5B → 175B parameters, qualitatively new capabilities appear: first zero-shot performance, then few-shot in-context learning. The architecture barely changed — scale itself drove the breakthroughs.

### GPT-2 (2019): Zero-Shot Emergence

**What changed from GPT-1:**
- 13x more parameters (117M → 1.5B)
- 10x more data (800M words → 8B words from WebText)
- No fine-tuning needed for some tasks — "zero-shot"

**Key results:**
- Zero-shot SOTA on 7/8 language modeling benchmarks
- Could write coherent paragraphs, do basic translation, answer questions
- OpenAI initially withheld release due to "misuse concerns" (controversial)

**The insight:** With enough diverse pre-training data, task knowledge is implicitly encoded. The model doesn't need to be told "this is a sentiment task" — it infers from context.

### GPT-3 (2020): In-Context Learning

**What changed from GPT-2:**
- 117x more parameters (1.5B → 175B)
- ~300B training tokens (filtered internet, books, Wikipedia)
- 2048-token context window (vs. 1024 for GPT-2)

**The in-context learning breakthrough:**
```
Prompt: "Translate English to French:
  sea otter → loutre de mer
  cheese → fromage
  plaid shirt →"

Model output: "chemise à carreaux"
```

No gradient updates. No fine-tuning. Just examples in the prompt.

**Three evaluation modes GPT-3 introduced:**
| Mode | Description | Performance |
|------|-------------|-------------|
| Zero-shot | Task description only | Inconsistent but impressive |
| One-shot | One example | Significant improvement |
| Few-shot | 10-100 examples in prompt | Near fine-tuned quality |


### The Scale Numbers

| Model | Year | Params | Training Data | Context | Est. Cost |
|-------|------|--------|---------------|---------|-----------|
| GPT-1 | 2018 | 117M | ~800M words | 512 | ~$10K |
| GPT-2 | 2019 | 1.5B | ~8B words | 1024 | ~$50K |
| GPT-3 | 2020 | 175B | ~300B tokens | 2048 | ~$4.6M |
| GPT-4 | 2023 | ~1.8T (MoE) | undisclosed | 8K-128K | ~$100M+ |
| Llama 3 | 2024 | 405B | 15T tokens | 128K | ~$100M+ |

### What Architecture Actually Changed (Very Little)

| Component | GPT-1 | GPT-3 | Change? |
|-----------|-------|-------|---------|
| Base architecture | Decoder Transformer | Decoder Transformer | Same |
| Attention | Standard multi-head | Standard multi-head | Same |
| Activation | GELU | GELU | Same |
| Normalization | Post-LN | Pre-LN | Minor tweak |
| Position encoding | Learned | Learned | Same |

**The lesson:** The Transformer architecture from 2017 was already powerful enough. The breakthroughs came from **scale** (more parameters, more data, more compute), not architectural innovation.

### Connection to Your Research: What Scale Alone Cannot Do

> **Research connection:** While GPT-3 demonstrated that scale enables in-context learning, Swaroop Mishra's research showed that scale alone has critical limitations:
>
> **Reframing (2021):** Showed that GPT-3's performance is highly sensitive to prompt format. The same task can swing ±30% depending on phrasing. Scale doesn't automatically give robustness to prompt variation. Reframing prompts boosted GPT-3 by 12.5% averaged across tasks.
>
> **Instruction-Example Equivalence (2023):** Found that a single well-written instruction can be worth ~200 training examples. This means the DATA EFFICIENCY gains from good instructions can outweigh brute-force scaling — cheaper and more accessible.
>
> **NumGLUE (Mishra, Sachdeva et al., 2022):** Demonstrated that even at 175B scale, GPT-3 performs 46.4% worse than humans on basic math reasoning. Scale doesn't solve everything — architecture and training approach matter for specific capabilities.
>
> [Paper: Mishra et al., 2021, "Reframing Instructional Prompts to GPTk's Language," arXiv:2109.07830]
> [Paper: Mishra et al., 2023, "Instruction Tuned Models are Quick Learners," arXiv:2306.05539]
> [Paper: Mishra, Sachdeva et al., 2022, "NumGLUE," arXiv:2204.05660, ACL 2022]


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Parameter growth log-scale] | Create custom | Log-scale chart: 117M → 1.5B → 175B → 1.8T |
| ![In-context learning example] | Brown et al. 2020, Figure 1 | Few-shot prompt → completion |
| ![Capability milestones] | Create custom | Which abilities appeared at which scale |
| ![GPT-3 accuracy vs shots] | Brown et al. 2020, Figure 3 | Zero/one/few-shot performance curves |

- **[SOURCE: Brown et al., 2020, "Language Models are Few-Shot Learners," arXiv:2005.14165, Figures 1, 3]**
- **[SOURCE: Radford et al., 2019, "Language Models are Unsupervised Multitask Learners"]**
- **[BLOG: Lilian Weng, "The Transformer Family Version 2.0" (2023)](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)** — Comprehensive overview with diagrams
- **[DIAGRAM TO CREATE]** Side-by-side: GPT-1 (fine-tune per task) vs. GPT-3 (few-shot in prompt) workflow

---

## Slide 13: Scaling Laws — The Science of "Bigger"

### Core Concept
Performance of language models follows smooth, predictable power-law relationships with model size, dataset size, and compute. This transforms "how big should we build?" from guessing into engineering.

### Kaplan et al. (2020): The Discovery

**The three power laws:**
```
L(N) ∝ N^(-0.076)    — Loss scales with model parameters
L(D) ∝ D^(-0.095)    — Loss scales with dataset size
L(C) ∝ C^(-0.050)    — Loss scales with compute budget
```

**What this means in plain language:**
- Every 10x increase in model size reduces loss by ~17%
- Every 10x increase in data reduces loss by ~20%
- Every 10x increase in compute reduces loss by ~11%
- These hold across 7+ orders of magnitude (remarkably smooth)

**Practical implications (Kaplan's view):**
1. Bigger models are more sample-efficient (learn more from each token)
2. For fixed compute, favor LARGER models trained SHORTER
3. Model shape (width/depth ratio) matters less than total parameter count
4. You can predict training loss before spending the compute


### The Log-Log Plot (Key Visual)

On a log-log scale, power laws appear as straight lines. The scaling laws paper showed that across model sizes from 768 to 1.5 billion parameters, the loss-vs-compute curve is a near-perfect straight line.

```
Log(Loss)
  |
  |  ·
  |    ·
  |      ·
  |        ·
  |          ·    ← Remarkably linear on log-log scale
  |            ·
  |              ·
  |________________ Log(Compute)
```

### The "Emergent Abilities" Debate

**Original claim (Wei et al., 2022):**
- Certain capabilities appear suddenly at specific scale thresholds
- Below threshold: near-zero performance; above: strong performance
- Arithmetic, multi-step reasoning, word unscrambling — all "emerge" at 10B+ params

**The counter-argument (Schaeffer et al., 2023):**
- "Emergent abilities are a mirage" — artifact of non-linear metrics
- With linear/continuous metrics, performance improves smoothly
- The "emergence" is in the MEASUREMENT, not the MODEL
- [Paper: Schaeffer et al., 2023, "Are Emergent Abilities of Large Language Models a Mirage?" arXiv:2304.15004]

**The nuanced view (what to tell students):**
- Performance on individual tasks does improve smoothly with scale (Schaeffer is right)
- But the UTILITY of that performance can be threshold-like (going from 40%→60% on math isn't useful; 60%→80% is)
- "Emergence" is better understood as "crossing a usefulness threshold" rather than a phase transition

### Connection to Benchmark Research (Your Work)

> **Research connection:** The emergence debate connects directly to HOW we measure model capabilities. Bhavdeep Sachdeva and Swaroop Mishra's benchmark research exposed this:
>
> **NumGLUE (2022):** Showed that state-of-the-art LLMs (including GPT-3) perform 46.4% worse than humans on basic math. The "emergent" math ability people claimed for GPT-3 looks very different when measured on a carefully constructed benchmark vs. cherry-picked examples.
>
> **LILA (2022):** Unified 20 math reasoning datasets and showed that multi-task training improves performance by 21.83% — but even the best model only reaches 60.4%. The gap between "appears to do math" and "reliably does math" is enormous.
>
> **"Is High Quality Data All We Need?" (2022):** Hypothesized that a smaller, high-quality dataset can outperform larger noisy ones — directly challenging the "just scale the data" approach.
>
> This work provides the HONEST assessment needed in the "hype vs. reality" framing: scaling helps, but benchmarks reveal the real story.
>
> [Paper: Mishra, Sachdeva et al., 2022, "NumGLUE," ACL 2022]
> [Paper: Mishra et al., 2022, "LILA," EMNLP 2022, arXiv:2210.17517]
> [Paper: Sachdeva, Mishra et al., 2022, arXiv:2203.06404]


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Power-law curves] | Kaplan et al. 2020, Figure 1 | Log-log loss vs. compute/params/data |
| ![Emergence mirage] | Schaeffer et al. 2023, Figure 1 | Nonlinear vs. linear metric comparison |
| ![Scaling comparison table] | Create custom | Kaplan vs. Chinchilla optimal ratios |
| ![NumGLUE gap] | Mishra, Sachdeva et al. 2022 | Human vs. model performance on math |

- **[SOURCE: Kaplan et al., 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361, Figure 1]** — The canonical power-law curves
- **[SOURCE: Schaeffer et al., 2023, "Are Emergent Abilities a Mirage?" arXiv:2304.15004, Figure 1]** — Metric choice creates emergence illusion
- **[SOURCE: Wei et al., 2022, "Emergent Abilities of Large Language Models," arXiv:2206.07682]** — Original emergence claims
- **[DIAGRAM TO CREATE]** Annotated log-log plot with GPT-1/2/3/4 placed on the curve + "where would infinite scale take us?" arrow

---

## Slide 14: The Chinchilla Moment

### Core Concept
DeepMind's Chinchilla paper showed that the industry was building models that were too large and training them too little. A 70B model trained on 4x more data beat a 280B model trained on less data — using the same compute budget.

### The Key Finding

**Kaplan (2020):** "For a fixed compute budget, use most of it on model size."
- Optimal allocation: N ∝ C^0.73 (params grow fast), D ∝ C^0.27 (data grows slow)
- Implication: Build very large models, train them briefly

**Chinchilla (2022):** "Balance model size and data equally."
- Optimal allocation: N ∝ C^0.50, D ∝ C^0.50
- Rule of thumb: ~20 tokens per parameter for optimal training
- Implication: Build moderate models, train them thoroughly

### The Evidence

**Chinchilla (70B) vs. Gopher (280B):**
| Metric | Chinchilla 70B | Gopher 280B | Winner |
|--------|---------------|-------------|--------|
| Training tokens | 1.4T | 300B | Chinchilla 4.7x more |
| Parameters | 70B | 280B | Gopher 4x larger |
| Training compute | Same | Same | Tie (same budget) |
| Language modeling | Better | Worse | **Chinchilla** |
| MMLU | 67.6% | 60.0% | **Chinchilla** |
| Reading comp | Better | Worse | **Chinchilla** |

**The takeaway:** A model 4x smaller, trained on 4.7x more data, wins — at the same cost.


### Impact on the Industry

**Before Chinchilla (2020-2022):**
- GPT-3: 175B params, ~300B tokens (undertrained by Chinchilla standards)
- Gopher: 280B params, ~300B tokens (severely undertrained)
- PaLM: 540B params, 780B tokens (still undertrained)
- Industry consensus: "bigger model = better"

**After Chinchilla (2022-present):**
- Llama 1 (Meta, 2023): 7B-65B, trained on 1-1.4T tokens (Chinchilla-optimal)
- Llama 2 (Meta, 2023): 7B-70B, trained on 2T tokens (slightly over-trained for inference efficiency)
- Llama 3 (Meta, 2024): 8B-405B, trained on **15T tokens** (massively over-trained for deployment)
- Industry consensus: "train it longer, even beyond Chinchilla optimal, for better inference efficiency"

**The new wrinkle (2023-2024): Over-training for deployment**
- Chinchilla gives compute-optimal training (cheapest training cost)
- But inference cost depends on model SIZE, not training duration
- A smaller model trained longer is cheaper to deploy
- Llama 3 8B trained on 15T tokens (1875 tokens/param) vs. Chinchilla's 20 tokens/param
- This "over-training" makes deployment much cheaper at slight training cost increase

### The Chinchilla Ratio Across Models

| Model | Params | Tokens | Tokens/Param | vs. Chinchilla Optimal |
|-------|--------|--------|-------------|----------------------|
| GPT-3 | 175B | 300B | 1.7 | 11x undertrained |
| Chinchilla | 70B | 1.4T | 20 | Optimal |
| Llama 1 (65B) | 65B | 1.4T | 21.5 | ~Optimal |
| Llama 2 (70B) | 70B | 2T | 28.6 | 1.4x over-trained |
| Llama 3 (8B) | 8B | 15T | 1,875 | 94x over-trained (for deployment) |
| Llama 3 (405B) | 405B | 15T | 37 | 1.9x over-trained |

### Connection to Data Quality Research (Your Work)

> **Research connection:** Chinchilla said "more data." But Bhavdeep Sachdeva and Swaroop Mishra's research asks: "more data OF WHAT QUALITY?"
>
> **"Is High Quality Data All We Need?" (Sachdeva, Mishra et al., 2022):** Directly challenges the "just scale the data" approach. Hypothesizes that a smaller, high-quality dataset can outperform larger noisy ones. This is orthogonal to Chinchilla — you need enough data, but QUALITY may matter more than raw quantity.
>
> **VAIDA (2023):** Provides the methodology. If you're going to train on more data (Chinchilla), you need tools to ensure that data is high quality. VAIDA's visual-analytics-driven benchmark creation reduces artifacts by 45.8% — that's 45.8% less noise entering the model's MLP memories.
>
> **Super-NaturalInstructions (2022):** 1,616 tasks with expert-written instructions covering 76 task types and 55 languages. Showed that Tk-Instruct (a smaller model) outperformed larger models when given diverse, high-quality instruction data. This directly demonstrates: quality + diversity > raw scale.
>
> The synthesis: **Chinchilla says scale data. Your research says scale QUALITY data. The two together are more powerful than either alone.**
>
> [Paper: Sachdeva, Mishra et al., 2022, arXiv:2203.06404]
> [Paper: Arunkumar, Mishra, Sachdeva et al., 2023, "VAIDA," EACL, arXiv:2302.04434]
> [Paper: Wang, Mishra et al., 2022, "Super-NaturalInstructions," EMNLP 2022, arXiv:2204.07705]


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Chinchilla vs Gopher] | Hoffmann et al. 2022 | Bar chart comparison on benchmarks |
| ![Optimal ratio curves] | Hoffmann et al. 2022, Figure 3 | Isoflop curves showing optimal N/D balance |
| ![Tokens-per-param ratio] | Create custom | Table/chart of models vs. Chinchilla ratio |
| ![Training regime evolution] | Create custom | Timeline: undertrained → optimal → over-trained |

- **[SOURCE: Hoffmann et al., 2022, "Training Compute-Optimal LLMs," arXiv:2203.15556, Figures 3-4]** — Isoflop curves
- **[SOURCE: Meta, 2024, "The Llama 3 Herd of Models"](https://ai.meta.com/blog/meta-llama-3-1/)** — 15T token training details
- **[BLOG: Nostalgebraist, "Chinchilla's Wild Implications" (2022)](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)** — Accessible analysis
- **[DIAGRAM TO CREATE]** Two-axis chart: x=params, y=tokens, with models plotted and "Chinchilla line" showing optimal ratio. Models above line = over-trained (for deployment); below = undertrained.

---

## Slide 14b: The Instruction-Tuning Revolution (Bridging Scale to Usability)

### Core Concept
Scale gave us capable models (GPT-3). But scale alone didn't make them USABLE. Instruction-tuning — pioneered by Swaroop Mishra — was the missing link between "powerful language model" and "useful assistant."

### The Problem Scale Didn't Solve

GPT-3 at 175B parameters could do amazing things... if you knew the right prompt:
```
Bad prompt: "What is the capital of France?"
GPT-3 might continue: "What is the capital of Germany? What is the capital of..."

Good prompt: "Answer the following question: What is the capital of France?\nAnswer:"
GPT-3: "Paris"
```

The model had knowledge but no instruction-following ability.

### The Natural Instructions Breakthrough (Mishra et al., 2021)

**First paper on instruction-tuning:**
- Collected 61 existing NLP tasks and their crowdsourcing instructions
- Mapped them to a unified schema: {instruction, input, output}
- Fine-tuned models on this meta-dataset
- Result: 19% better generalization to UNSEEN tasks when instructions are included

**Key finding:** Models benefit from instructions even for tasks they've never seen. The instruction acts as a "routing signal" telling the model what behavior is expected.

### Super-NaturalInstructions: Scaling Instructions (Mishra, Wang et al., 2022)

**Scaling the instruction-tuning paradigm:**
- 1,616 diverse NLP tasks with expert-written instructions
- 76 distinct task types (classification, extraction, rewriting, etc.)
- 55 languages
- Result: Tk-Instruct (smaller model) outperformed much larger models on cross-task generalization


### The Instruction-Example Equivalence (Mishra et al., 2023)

**A remarkable finding:**
- Adding one additional well-written instruction to a task improves performance equivalent to adding ~200 training examples
- This means: a single sentence of guidance ≈ 200 labeled data points
- Implication: INSTRUCTIONS are an extremely efficient form of scaling

```
Traditional scaling:     200 more examples → +X% performance
Instruction scaling:     1 instruction     → +X% performance (same gain, much cheaper!)
```

**Why this matters for the scaling story:**
- Chinchilla said "scale data" — expensive, resource-intensive
- Instruction-tuning says "scale instruction quality" — cheap, accessible
- This democratizes the benefits of scaling: you don't need trillion-token datasets

### The Path from Instructions to ChatGPT

```
Natural Instructions (Mishra, 2021)    → "Models can follow instructions"
Super-NaturalInstructions (2022)       → "At scale: 1,616 tasks"
Self-Instruct (Wang, Mishra, 2023)     → "Models can GENERATE their own instruction data"
InstructGPT (OpenAI, 2022)             → "Add RLHF on top → ChatGPT"
```

Swaroop Mishra's instruction-tuning work is a direct ancestor of the techniques that made ChatGPT possible. The ChatGPT revolution was built on:
1. Scale (GPT-3.5 as base) — Brown et al.
2. Instruction-tuning (teaching models to follow instructions) — **Mishra et al.**
3. RLHF (human preference alignment) — Ouyang et al.

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Instruction-tuning pipeline] | Mishra et al. 2021 | Tasks → Instructions → Unified training |
| ![Tk-Instruct vs larger models] | Wang, Mishra et al. 2022 | Smaller model + instructions beats larger |
| ![Instruction-example equivalence] | Mishra et al. 2023 | 1 instruction ≈ 200 examples visual |
| ![Path to ChatGPT] | Create custom | Timeline from NatInst → ChatGPT |

- **[SOURCE: Mishra et al., 2022, "Cross-Task Generalization," ACL 2022, arXiv:2104.08773]**
- **[SOURCE: Wang, Mishra et al., 2022, "Super-NaturalInstructions," EMNLP 2022, arXiv:2204.07705]**
- **[SOURCE: Mishra et al., 2023, "Instruction Tuned Models are Quick Learners," arXiv:2306.05539]**
- **[AWARD: AI2 Lasting Impact Paper Award — Natural Instructions, 3.5 years after publication]**
- **[DIAGRAM TO CREATE]** Efficiency comparison: bar chart showing "200 examples" vs. "1 instruction" achieving same gain

---

## Summary: Key Numbers for Section 2

### Parameter Scale
- GPT-1: 117M (2018)
- GPT-2: 1.5B (2019) — 13x growth
- GPT-3: 175B (2020) — 117x growth
- GPT-4: ~1.8T MoE (2023) — ~10x total growth (but sparse)
- Llama 3: 405B dense (2024) — largest open model

### Training Data Scale
- GPT-1: ~800M words (BooksCorpus)
- GPT-2: ~8B words (WebText, 40GB)
- GPT-3: ~300B tokens (filtered internet, ~570GB)
- Llama 3: 15T tokens (~multi-PB scale)
- Scale of data: ~50,000x growth from GPT-1 to Llama 3

### Cost Scale
- GPT-1: ~$10K (estimated)
- GPT-3: ~$4.6M (estimated, 2020)
- GPT-4: ~$100M+ (estimated, 2023)
- Llama 3 (405B): ~$100M+ (16K H100 GPUs, weeks)

### Key Ratios
- Chinchilla optimal: ~20 tokens per parameter
- Llama 3 (8B): 1,875 tokens per parameter (94x Chinchilla)
- Instruction-example equivalence: 1 instruction ≈ 200 examples (Mishra 2023)
- Reframing boost: +12.5% on GPT-3 (Mishra 2021)
- NumGLUE human-model gap: 46.4% (Mishra, Sachdeva 2022)

### Performance Milestones
- GPT-1: 9/12 benchmarks beat with fine-tuning
- GPT-2: 7/8 LM benchmarks zero-shot SOTA
- GPT-3: Few-shot competitive with fine-tuned models
- GPT-4: Bar exam top 10%, multimodal, reasoning
- Tk-Instruct (11B): Outperformed 175B GPT-3 on cross-task generalization with instructions

---

## Visual Index: All Images for Section 2

### From Original Papers

| # | Visual | Paper | Figure | Slide |
|---|--------|-------|--------|-------|
| 1 | Pre-train + fine-tune diagram | Radford et al. 2018 | Fig 1 | Slide 11 |
| 2 | In-context learning example | Brown et al. 2020 | Fig 1 | Slide 12 |
| 3 | Few-shot performance curves | Brown et al. 2020 | Fig 3 | Slide 12 |
| 4 | Power-law loss curves | Kaplan et al. 2020 | Fig 1 | Slide 13 |
| 5 | Isoflop optimal allocation | Hoffmann et al. 2022 | Fig 3 | Slide 14 |
| 6 | Chinchilla vs. Gopher results | Hoffmann et al. 2022 | Table 4 | Slide 14 |
| 7 | Tk-Instruct generalization | Wang, Mishra et al. 2022 | Fig 3 | Slide 14b |
| 8 | NatInst cross-task results | Mishra et al. 2022 | Table 2 | Slide 14b |

### From Blogs & Resources

| # | Visual | Source | URL | Slide |
|---|--------|--------|-----|-------|
| 1 | GPT family tree | Multiple | Various infographics | Slide 12 |
| 2 | Chinchilla analysis | Nostalgebraist blog | lesswrong.com | Slide 14 |
| 3 | Llama 3 training details | Meta AI Blog | ai.meta.com | Slide 14 |
| 4 | Emergence mirage | Schaeffer et al. | arXiv:2304.15004, Fig 1 | Slide 13 |

### Diagrams to Create

| # | Visual | Description | Slide | Tool |
|---|--------|-------------|-------|------|
| 1 | Three-era timeline | Task-specific → Pre-train+FT → Instructions | Slide 11 | Excalidraw |
| 2 | Parameter growth chart | Log-scale GPT-1→4 with Llama 3 | Slide 12 | Python/matplotlib |
| 3 | Architecture stability | Table showing "nothing changed architecturally" | Slide 12 | PowerPoint |
| 4 | Annotated log-log plot | GPT models placed on the scaling curve | Slide 13 | Python |
| 5 | Chinchilla ratio chart | Params vs. tokens with optimal line | Slide 14 | Python |
| 6 | Undertrained → optimal → over-trained | Training regime evolution timeline | Slide 14 | Excalidraw |
| 7 | Instruction efficiency | "1 instruction ≈ 200 examples" visual | Slide 14b | PowerPoint |
| 8 | NatInst → ChatGPT path | Timeline showing direct lineage | Slide 14b | Excalidraw |

---

## Hype vs. Reality for Section 2

### What's Real (Evidence-Backed)
- ✅ Power-law scaling is empirically robust across 7+ orders of magnitude
- ✅ Chinchilla optimal training genuinely saves resources
- ✅ In-context learning works and scales with model size
- ✅ Instruction-tuning makes smaller models competitive with larger ones
- ✅ Data quality matters independently of data quantity

### What's Overstated
- ⚠️ "Emergent abilities appear suddenly" — likely a measurement artifact (Schaeffer 2023)
- ⚠️ "Scale solves everything" — NumGLUE shows 46.4% gap on basic math even at 175B
- ⚠️ "GPT-4 is 1.8T parameters" — unconfirmed; based on leaks, not official disclosure
- ⚠️ "Bigger is always better" — Tk-Instruct (11B) > GPT-3 (175B) with good instructions

### What's Hype
- ❌ "GPT-5 will solve everything" — no evidence extrapolating beyond current scaling laws
- ❌ "Scaling laws predict task performance" — they predict LOSS, not specific capabilities
- ❌ "We just need more compute" — alignment, data quality, and architecture all matter independently
- ❌ "Open models can't compete with closed" — Llama 3 405B rivals GPT-4 on many benchmarks

---

## Appendix: Full Citation List for Section 2

1. Radford et al., 2018, "Improving Language Understanding by Generative Pre-Training" (GPT-1)
2. Radford et al., 2019, "Language Models are Unsupervised Multitask Learners" (GPT-2)
3. Brown et al., 2020, "Language Models are Few-Shot Learners," arXiv:2005.14165 (GPT-3)
4. Kaplan et al., 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361
5. Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models," arXiv:2203.15556 (Chinchilla)
6. **Mishra et al., 2021, "Cross-Task Generalization via Natural Language Crowdsourcing Instructions," arXiv:2104.08773** (Natural Instructions)
7. **Mishra et al., 2022, "Reframing Instructional Prompts to GPTk's Language," ACL Findings, arXiv:2109.07830**
8. **Wang, Mishra et al., 2022, "Super-NaturalInstructions," EMNLP 2022, arXiv:2204.07705**
9. **Mishra, Sachdeva et al., 2022, "NumGLUE," ACL 2022, arXiv:2204.05660**
10. **Mishra et al., 2022, "LILA: A Unified Benchmark for Mathematical Reasoning," EMNLP 2022, arXiv:2210.17517**
11. **Sachdeva, Mishra et al., 2022, "Is High Quality Data All We Need?" arXiv:2203.06404**
12. **Mishra et al., 2023, "Instruction Tuned Models are Quick Learners," arXiv:2306.05539**
13. **Wang, Mishra et al., 2023, "Self-Instruct," ACL 2023, arXiv:2212.10560**
14. Schaeffer et al., 2023, "Are Emergent Abilities of Large Language Models a Mirage?" arXiv:2304.15004
15. OpenAI, 2023, "GPT-4 Technical Report," arXiv:2303.08774
16. Meta, 2024, "The Llama 3 Herd of Models" (technical report)
17. **Zhou, Mishra et al., 2024, "Self-Discover," arXiv:2402.03620** (Google DeepMind)

---

*Document created: July 2, 2026*
*Last updated: July 2, 2026*
