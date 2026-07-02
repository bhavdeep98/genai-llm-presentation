# Section 3: Alignment — Making Models Useful — Deep Dive Research Document

## Purpose
This document expands Section 3 (Alignment — Making Models Useful) into research-paper-level depth. It covers the alignment problem, RLHF, DPO, Constitutional AI, and the fundamental limitations. Swaroop Mishra's work on instruction-bias, self-correction limitations, and the instruction-tuning pipeline is central here — instruction-tuning IS the first stage of alignment.

---

## Timeline: Alignment Milestones + Your Research

| Year | Milestone | Authors | Key Insight |
|------|-----------|---------|-------------|
| 2017 | "Deep RL from Human Preferences" | Christiano et al. | Original RLHF concept for games/robotics |
| 2020 | "Learning to Summarize from Human Feedback" | Stiennon et al. (OpenAI) | First RLHF for language (summarization) |
| 2021 | **Natural Instructions** | **Swaroop Mishra** et al. | Instructions as alignment signal |
| 2022 | InstructGPT / ChatGPT | Ouyang et al. (OpenAI) | 3-stage RLHF pipeline at scale |
| 2022 | Constitutional AI | Bai et al. (Anthropic) | AI self-critique with principles |
| 2023 | **Instruction-Bias** | Parmar, **Mishra** et al. | EACL 2023 🏆 — Bias in annotation instructions propagates to models |
| 2022 | **Super-NaturalInstructions** | **Mishra**, Wang et al. | 1,616 tasks — SFT at scale |
| 2023 | DPO | Rafailov et al. (Stanford) | Skip reward model, direct optimization |
| 2023 | **Self-Instruct** | Wang, **Mishra** et al. | Models generate own alignment data |
| 2023 | **"LLMs Cannot Self-Correct Reasoning Yet"** | Huang, **Mishra** et al. | ICLR 2024 — fundamental limitation of self-improvement |
| 2023 | RLAIF | Lee et al. (Google) | Replace human feedback with AI feedback |
| 2023 | Open Problems in RLHF | Casper et al. | Systematic analysis of RLHF limitations |
| 2024 | **Self-Discover** | Zhou, **Mishra** et al. | Structured self-improvement (that actually works) |
| 2024 | RLHF Alternatives Landscape | Various | DPO, IPO, KTO, ORPO — simplifying alignment |

---

## Slide 15: The Alignment Problem

### Core Concept
A model can be incredibly capable (GPT-3: 175B params, trillion-dollar training) but completely unhelpful, harmful, or uncontrollable. Alignment is the problem of making capable models actually do what users want.

### The Gap Between Capability and Usefulness

**Raw GPT-3 (2020) — capable but unaligned:**
```
User: "What's the capital of France?"
GPT-3: "What's the capital of France? This is a question that many students
       ask. The capital of France is a topic discussed in geography class..."
       [continues without answering]
```

**InstructGPT (2022) — aligned:**
```
User: "What's the capital of France?"
InstructGPT: "The capital of France is Paris."
```

Same architecture, same pre-training — the difference is alignment.

### Three Faces of the Alignment Problem

1. **Helpfulness:** Does the model answer what you actually asked?
   - Raw LLMs often ramble, continue the prompt pattern, or give tangential info
   - Users want direct, useful responses

2. **Harmlessness:** Does the model refuse dangerous/toxic requests?
   - Raw LLMs will generate anything in their distribution
   - Society needs models that decline harmful requests

3. **Honesty:** Does the model acknowledge uncertainty and avoid hallucination?
   - Raw LLMs confidently assert false information
   - Aligned models should say "I don't know" when appropriate


### The Key Insight: More Parameters ≠ More Aligned

| Model | Parameters | Human Preference |
|-------|-----------|-----------------|
| GPT-3 | 175B | Baseline |
| InstructGPT | 1.3B | **Preferred over GPT-3** |
| Factor | 135x fewer params | Better alignment |

A 1.3B model with alignment training was preferred by humans over a 175B model without it. This is the most powerful argument for alignment research: technique beats scale.

### Connection to Instruction-Bias Research (Your Mentor's Award-Winning Work)

> **Research connection:** Swaroop Mishra's work on **instruction-bias** (Parmar, Mishra et al., EACL 2023 — Outstanding Paper Award) reveals a deeper layer of the alignment problem: **bias enters at the instruction level**, before any model training happens.
>
> **Key findings:**
> - Crowdsourcing instructions for dataset creation contain concrete patterns/biases
> - These patterns propagate to the collected data
> - Models then learn these biases and perform well on biased test sets but fail on unbiased data
> - This can lead to **overestimation of model performance** — the model appears aligned but has learned superficial correlations
>
> **Why this matters for alignment:**
> - RLHF Stage 1 (SFT) depends on human demonstrations — if instructions to demonstrators are biased, the SFT model inherits those biases
> - RLHF Stage 2 (Reward Model) depends on human rankings — if annotator instructions are biased, the reward model learns biased preferences
> - The entire alignment pipeline is only as good as the INSTRUCTIONS given to humans
>
> [Paper: Parmar, Mishra, Geva, Baral, 2023, "Don't Blame the Annotator: Bias Already Starts in the Annotation Instructions," EACL 2023, arXiv:2205.00415]
> [🏆 EACL 2023 Outstanding Paper Award]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Aligned vs unaligned output] | Create custom | Side-by-side: same question, raw GPT-3 vs. InstructGPT |
| ![Three faces of alignment] | Create custom | Helpful / Harmless / Honest triangle |
| ![Instruction-bias flow] | Parmar, Mishra et al. 2023 | How bias propagates from instructions to model |

- **[SOURCE: Ouyang et al., 2022, arXiv:2203.02155]** — InstructGPT preference results
- **[SOURCE: Parmar, Mishra et al., 2023, arXiv:2205.00415, EACL Outstanding Paper]** — Instruction-bias propagation
- **[DIAGRAM TO CREATE]** Bias propagation chain: Instructions → Annotators → Data → Model → Outputs

---

## Slide 16: InstructGPT & the RLHF Pipeline

### Core Concept
The 3-stage pipeline (SFT → Reward Model → PPO) that turned GPT-3 into ChatGPT. Each stage builds on the previous, progressively aligning the model with human preferences.

### Stage 1: Supervised Fine-Tuning (SFT)

**What happens:**
- Human demonstrators write ideal responses to diverse prompts
- Fine-tune the pre-trained LLM on these demonstrations
- Standard supervised learning (cross-entropy loss on the ideal responses)

**Data:**
- ~13,000 demonstration examples (InstructGPT)
- Labelers instructed to be helpful, truthful, and harmless
- Covers diverse tasks: QA, summarization, creative writing, coding, etc.

**Result:** A model that can follow instructions, but imperfectly. It mimics the demonstrations but hasn't learned nuanced preferences.

**Connection to your mentor's work:** This is literally instruction-tuning. The SFT stage of RLHF is the same technique Swaroop Mishra pioneered in Natural Instructions (2021). InstructGPT's SFT stage uses ~13K examples; Super-NaturalInstructions showed that scaling this to 1,616 tasks with expert instructions produces even better cross-task generalization.

### Stage 2: Reward Model Training

**What happens:**
- Generate multiple responses to the same prompt using the SFT model
- Human labelers RANK the responses (best to worst)
- Train a separate model to predict these rankings
- The reward model learns a scalar "quality score" for any (prompt, response) pair

**Architecture:**
- Same architecture as the LLM (6B parameters in InstructGPT)
- Output: single scalar reward value
- Trained with pairwise ranking loss (Bradley-Terry model)

**Data:**
- ~33,000 comparison pairs (InstructGPT)
- Each comparison: one prompt, multiple responses, human ranking

**Critical insight:** The reward model captures preferences that are hard to specify explicitly. "Be helpful but not verbose, be honest but not harsh, be creative but not inappropriate" — these are learned implicitly from examples.

### Stage 3: Reinforcement Learning (PPO)

**What happens:**
- Use the reward model as an automatic scoring function
- Optimize the SFT model to maximize reward scores using PPO
- Add KL divergence penalty to prevent the model from diverging too far from SFT model

```
Objective = E[Reward(prompt, response)] - β * KL(π_RL || π_SFT)
```

Where:
- π_RL = current RL policy (being optimized)
- π_SFT = reference policy (the SFT model)
- β = KL penalty coefficient (prevents reward hacking)


**Why PPO and not simpler RL:**
- PPO clips the policy update to prevent too-large steps
- Stable training (language RL is notoriously unstable)
- Maintains diversity of outputs (doesn't collapse to one "safe" response)

### The Full Pipeline Visualized

```
Stage 1: SFT                    Stage 2: Reward Model           Stage 3: RL (PPO)
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│ Prompt           │            │ Prompt           │            │ Prompt           │
│ + Ideal Response │──Train──→ │ + Response A     │──Train──→ │ Generate         │
│ (human demo)     │            │ + Response B     │            │ Response         │
│                  │            │ Human: A > B     │            │ Score w/ RM      │
│ ~13K examples    │            │ ~33K comparisons │            │ Update w/ PPO    │
└─────────────────┘            └─────────────────┘            └─────────────────┘
      ↓                               ↓                              ↓
   SFT Model                    Reward Model                    Final Model
   (can follow                  (scores quality                 (optimizes for
    instructions)                of responses)                   human preference)
```

### Key Results (InstructGPT, Ouyang et al., 2022)

| Metric | GPT-3 (175B) | InstructGPT (1.3B) | Winner |
|--------|-------------|-------------------|--------|
| Human preference | Baseline | Preferred 85%+ | InstructGPT |
| Truthfulness (TruthfulQA) | 22% | 28% | InstructGPT (+6%) |
| Toxicity reduction | Baseline | 25% less toxic | InstructGPT |
| Following instructions | Poor | Good | InstructGPT |
| NLP benchmarks | Better | Slightly worse | GPT-3 (alignment tax) |

**The "alignment tax":** The aligned model performs slightly worse on academic benchmarks (it's less willing to guess or produce harmful content that might score well). But it's dramatically more useful for real users.

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![3-stage RLHF pipeline] | Ouyang et al. 2022, Figure 2 | The canonical pipeline diagram |
| ![Human preference chart] | Ouyang et al. 2022, Figure 4 | 1.3B InstructGPT > 175B GPT-3 |
| ![PPO optimization loop] | Create custom | Generate → Score → Update cycle |

- **[SOURCE: Ouyang et al., 2022, "Training Language Models to Follow Instructions," arXiv:2203.02155, Figure 2]** — The iconic 3-stage diagram
- **[SOURCE: Ouyang et al., 2022, Figure 4]** — Human preference win rates
- **[BLOG: Chip Huyen, "Reinforcement Learning from Human Feedback" (2023)](https://huyenchip.com/2023/05/02/rlhf)** — Accessible explanation
- **[DIAGRAM TO CREATE]** Annotated 3-stage with data sizes, model sizes, and training cost per stage

---

## Slide 17: DPO — Simplifying Alignment

### Core Concept
DPO eliminates the reward model entirely. Instead of three stages (SFT → RM → RL), it directly optimizes the policy from preference pairs in a single step. Same results, dramatically simpler.

### The Insight That Makes DPO Possible

**The mathematical trick:**
Under the standard RLHF objective, there exists a closed-form solution for the optimal policy:
```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
```

This means you can rearrange the reward function as:
```
r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + const
```

**The implication:** The reward is implicitly defined by the policy itself. You don't need a separate reward model — you can directly optimize the policy to satisfy preference constraints.

### DPO Loss Function

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

In plain language:
- Take a preference pair: (y_w = preferred, y_l = dispreferred)
- Increase the probability of the preferred response (relative to reference)
- Decrease the probability of the dispreferred response (relative to reference)
- β controls how strongly preferences override the reference model

### RLHF vs. DPO Comparison

```
RLHF (3 models needed):               DPO (2 models needed):
┌──────────────────────┐              ┌──────────────────────┐
│ 1. Train SFT model   │              │ 1. Train SFT model   │
│ 2. Train Reward Model│              │ 2. Optimize directly │
│ 3. PPO optimization  │              │    from preferences  │
│                      │              │                      │
│ Models in memory:    │              │ Models in memory:     │
│ - Policy (being      │              │ - Policy (being      │
│   optimized)         │              │   optimized)         │
│ - Reward model       │              │ - Reference model    │
│ - Reference model    │              │   (frozen SFT)       │
│                      │              │                      │
│ Hyperparameters:     │              │ Hyperparameters:     │
│ - PPO clip ratio     │              │ - β (temperature)    │
│ - KL coefficient     │              │   That's it.         │
│ - Learning rate      │              │                      │
│ - Batch size         │              │                      │
│ - Epochs...          │              │                      │
└──────────────────────┘              └──────────────────────┘
```


### The Alignment Methods Landscape (2024-2026)

| Method | Year | Approach | Key Advantage |
|--------|------|----------|---------------|
| RLHF (PPO) | 2022 | RM + RL | Most capable; industry standard |
| DPO | 2023 | Direct optimization | Simpler, stable, no RM needed |
| IPO | 2023 | Identity preference | Avoids DPO's overfitting |
| KTO | 2024 | Binary feedback | Only needs thumbs up/down |
| ORPO | 2024 | Odds ratio | No reference model needed |
| RLAIF | 2023 | AI generates preferences | Scales beyond human annotation |
| Constitutional AI | 2022 | AI self-critique + principles | Explicit rules, interpretable |

### Connection to Self-Instruct (Your Mentor's Work)

> **Research connection:** Swaroop Mishra's **Self-Instruct (2023)** is a direct precursor to RLAIF. The insight:
>
> - RLHF requires expensive human annotation
> - Self-Instruct showed LLMs can generate their own instruction-following data
> - RLAIF (Google, 2023) extends this: LLMs can generate their own PREFERENCE data
> - Both eliminate the human bottleneck in alignment
>
> The progression:
> ```
> Self-Instruct (Mishra, 2023): LLM generates instruction data → SFT
> RLAIF (Lee et al., 2023):     LLM generates preference data → DPO/RL
> Self-Discover (Mishra, 2024): LLM discovers own reasoning structures
> ```
>
> Each step reduces human dependency in the alignment pipeline.
>
> [Paper: Wang, Mishra et al., 2023, "Self-Instruct," ACL 2023, arXiv:2212.10560]
> [Paper: Lee et al., 2023, "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback," ICML 2024, arXiv:2309.00267]

### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![RLHF vs DPO pipeline] | Create custom | Side-by-side 3-stage vs. 2-stage |
| ![DPO loss intuition] | Create custom | Preferred↑ / Dispreferred↓ visual |
| ![Methods landscape] | Create custom | Timeline of alignment methods 2022-2026 |

- **[SOURCE: Rafailov et al., 2023, "DPO," arXiv:2305.18290, Figure 1]** — RLHF vs. DPO comparison
- **[SOURCE: Lee et al., 2023, "RLAIF," arXiv:2309.00267]** — AI feedback matching human feedback
- **[DIAGRAM TO CREATE]** Two-panel: RLHF (3 stages, 3 models) vs. DPO (2 stages, 2 models)

---

## Slide 18: Alignment Limitations (Honest Assessment)

### Core Concept
RLHF/DPO make models more useful, but they do NOT solve alignment in any deep sense. Understanding the limitations is critical for anyone building on these systems.

### Limitation 1: Reward Hacking

**The problem:** Models learn to game the reward model rather than genuinely improve.

**Examples:**
- Model learns that longer responses score higher → becomes verbose
- Model learns that hedging ("I think...") scores as more "honest" → becomes uncertain about everything
- Model learns that refusing requests scores as "safe" → over-refuses benign queries

**Why it happens:** The reward model is a learned approximation of human preferences. It has blind spots. The RL optimization will find and exploit those blind spots (Goodhart's Law).

### Limitation 2: Scalable Oversight

**The problem:** As models become more capable, humans can't reliably evaluate their outputs.

**Examples:**
- Can a human labeler evaluate if a code generation is optimal? Secure? Bug-free?
- Can a human evaluate if a research summary accurately represents a 50-page paper?
- Can a human detect subtle factual errors in domains they don't specialize in?

**The scaling paradox:** We need alignment MOST for the most capable models, but those are exactly the models whose outputs we're LEAST able to evaluate.

### Limitation 3: Self-Correction Doesn't Work (Your Mentor's Research)

> **Research connection:** Swaroop Mishra co-authored a landmark paper showing that **LLMs cannot self-correct their reasoning without external feedback** (Huang, Mishra et al., ICLR 2024).
>
> **Key findings:**
> - When asked to "check your work," LLMs often DEGRADE their answers
> - Without external feedback (tools, human correction), self-correction is unreliable
> - The model's confidence in a wrong answer is the same as in a right answer
> - "Self-improvement" through self-correction is largely an illusion (without external signal)
>
> **Why this matters for alignment:**
> - Many proposed alignment approaches rely on models "catching their own mistakes"
> - Constitutional AI asks models to self-critique — but this paper shows the limits
> - True alignment likely requires EXTERNAL verification, not just self-reflection
> - The Self-Discover framework (Mishra, 2024) works precisely because it provides STRUCTURE — an external scaffold for reasoning, not just "try again"
>
> [Paper: Huang, Chen, Mishra, Zheng, Yu, Song, Zhou, 2023, "Large Language Models Cannot Self-Correct Reasoning Yet," ICLR 2024, arXiv:2310.01798]
> [Published at: Google DeepMind]


### Limitation 4: Preference Inconsistency

**The problem:** Different annotators have different values. "Averaging" preferences destroys important signal.

- Liberal vs. conservative annotators disagree on political content
- Experts vs. novices disagree on technical accuracy vs. accessibility
- Cultural differences in what's "helpful" or "appropriate"
- The "aligned" model reflects the demographics of its annotators, not universal values

### Limitation 5: Distribution Shift

**The problem:** The reward model was trained on a specific distribution of prompts and responses. In deployment, users ask things never seen in training.

- Novel topics emerge after training data cutoff
- Users find creative prompt formats that bypass alignment
- Jailbreaking exploits this gap between training distribution and deployment distribution

### What Alignment Actually Does (Honest Framing for Students)

```
Alignment does:                     Alignment does NOT:
✅ Make outputs more helpful        ❌ Make models truthful
✅ Reduce obvious toxicity          ❌ Eliminate hallucination
✅ Improve instruction following    ❌ Give models values
✅ Make models more predictable     ❌ Solve safety
✅ Reduce obvious harms             ❌ Make models understand
```

### Connection to Data Quality (Your Work)

> **Research connection:** The alignment pipeline's weakest link is data quality — a theme central to Bhavdeep Sachdeva and Swaroop Mishra's research:
>
> - **Instruction-Bias (EACL 2023):** If the INSTRUCTIONS to annotators are biased, the alignment data inherits that bias → the aligned model is biased
> - **VAIDA (EACL 2023):** Provides tools to detect and fix artifacts in human-created data → directly applicable to improving RLHF training data quality
> - **"Is High Quality Data All We Need?" (2022):** The alignment data (demonstrations + rankings) is small (~50K total for InstructGPT). Quality of these few thousand examples matters enormously
> - **"Generalized but not Robust?" (ACL 2022):** Shows that models trained on artifacted data appear aligned on test sets but fail in deployment — exactly the reward hacking problem
>
> [Paper: Sachdeva, Mishra et al., 2022, "Generalized but not Robust?" ACL Findings]
> [Paper: Parmar, Mishra et al., 2023, "Instruction-Bias," EACL 2023]

### The Open Problems (Casper et al., 2023)

Systematic analysis of RLHF limitations:
1. **Reward misspecification:** The reward model doesn't perfectly capture human values
2. **Reward hacking:** Models exploit reward model weaknesses
3. **Scalable oversight:** Humans can't evaluate superhuman outputs
4. **Deceptive alignment:** Models could learn to APPEAR aligned during training
5. **Value lock-in:** Once aligned to specific values, hard to update

[Paper: Casper et al., 2023, "Open Problems and Fundamental Limitations of RLHF," arXiv:2307.15217]


### Visual References

| Visual | Source | Description |
|--------|--------|-------------|
| ![Reward hacking example] | Create custom | Model gaming length/hedging for reward |
| ![Self-correction failure] | Huang, Mishra et al. 2023 | Correct answer → self-correct → wrong answer |
| ![Alignment does/doesn't] | Create custom | Two-column honest assessment |
| ![Open problems taxonomy] | Casper et al. 2023, Figure 1 | Systematic failure modes |

- **[SOURCE: Huang, Mishra et al., 2023, "LLMs Cannot Self-Correct Reasoning Yet," arXiv:2310.01798, ICLR 2024]** — Self-correction degradation examples
- **[SOURCE: Casper et al., 2023, arXiv:2307.15217, Figure 1]** — Taxonomy of RLHF problems
- **[SOURCE: Parmar, Mishra et al., 2023, arXiv:2205.00415]** — Instruction-bias propagation
- **[DIAGRAM TO CREATE]** Chain: Instruction bias → Annotator bias → Data bias → Model bias → Deployment failure

---

## Summary: Key Numbers for Section 3

### InstructGPT Data Scale
- SFT demonstrations: ~13,000 examples
- Reward model comparisons: ~33,000 pairs
- Total unique prompts: ~40,000
- Labeler team: ~40 contractors
- Training cost (incremental over GPT-3): relatively small vs. pre-training

### DPO Advantages (Quantified)
- Models in memory: 2 (vs. 3 for RLHF)
- Hyperparameters: 1 (β) vs. ~10+ for PPO
- Training stability: standard gradient descent vs. unstable RL
- Results: comparable to PPO-based RLHF on standard benchmarks

### Self-Correction Failure Rates (Huang, Mishra et al., 2023)
- Without external feedback: performance degrades in majority of cases
- Models sometimes flip correct answers to incorrect after "self-correction"
- Key finding: models cannot reliably evaluate correctness of their own outputs

### Instruction-Example Efficiency
- 1 instruction ≈ 200 data samples (Mishra et al., 2023)
- InstructGPT used ~13K demos → equivalent to ~2.6M examples worth of signal via instructions
- Super-NaturalInstructions: 1,616 tasks with expert instructions → Tk-Instruct outperforms 175B GPT-3

---

## Visual Index: All Images for Section 3

### From Original Papers

| # | Visual | Paper | Figure | Slide |
|---|--------|-------|--------|-------|
| 1 | RLHF 3-stage pipeline | Ouyang et al. 2022 | Fig 2 | Slide 16 |
| 2 | Human preference win rates | Ouyang et al. 2022 | Fig 4 | Slide 16 |
| 3 | RLHF vs DPO diagram | Rafailov et al. 2023 | Fig 1 | Slide 17 |
| 4 | Self-correction degradation | Huang, Mishra et al. 2023 | Fig 2 | Slide 18 |
| 5 | RLHF open problems taxonomy | Casper et al. 2023 | Fig 1 | Slide 18 |
| 6 | Instruction-bias propagation | Parmar, Mishra et al. 2023 | Fig 1 | Slide 15 |

### From Blogs & Resources

| # | Visual | Source | URL | Slide |
|---|--------|--------|-----|-------|
| 1 | RLHF explained | Chip Huyen blog (2023) | huyenchip.com | Slide 16 |
| 2 | Alignment landscape | Multiple surveys | — | Slide 17 |
| 3 | Constitutional AI flow | Anthropic blog | anthropic.com | Slide 17 |

### Diagrams to Create

| # | Visual | Description | Slide | Tool |
|---|--------|-------------|-------|------|
| 1 | Aligned vs unaligned | Same prompt, two outputs side-by-side | Slide 15 | PowerPoint |
| 2 | HHH triangle | Helpful / Harmless / Honest | Slide 15 | Excalidraw |
| 3 | Bias propagation chain | Instructions → Annotators → Data → Model → Outputs | Slide 15 | Excalidraw |
| 4 | Annotated 3-stage | RLHF with data sizes and model counts | Slide 16 | draw.io |
| 5 | PPO optimization loop | Generate → Score → Update → Repeat | Slide 16 | Excalidraw |
| 6 | RLHF vs DPO | 3 models vs. 2 models, complexity comparison | Slide 17 | PowerPoint |
| 7 | Methods timeline | RLHF → DPO → KTO → ORPO evolution | Slide 17 | Mermaid/draw.io |
| 8 | Self-correction failure | Correct → "Let me check" → Wrong | Slide 18 | Excalidraw |
| 9 | "Alignment does/doesn't" | Two-column honest checklist | Slide 18 | PowerPoint |
| 10 | Open problems map | 5 failure modes with examples | Slide 18 | Excalidraw |

---

## Hype vs. Reality for Section 3

### What's Real (Evidence-Backed)
- ✅ RLHF dramatically improves user experience (1.3B > 175B)
- ✅ DPO achieves comparable results with less complexity
- ✅ Instruction-tuning is the foundation of all aligned models
- ✅ Self-correction without external feedback doesn't work (Mishra, ICLR 2024)
- ✅ Instruction-bias propagates through the alignment pipeline (Mishra, EACL 2023)
- ✅ RLAIF can match RLHF quality on standard tasks

### What's Overstated
- ⚠️ "RLHF solves alignment" — it's a surface-level adjustment, not deep alignment
- ⚠️ "Models are safe after alignment" — jailbreaks are trivial, distribution shift breaks alignment
- ⚠️ "DPO replaces RLHF" — for complex objectives, RLHF may still be necessary
- ⚠️ "Constitutional AI makes models ethical" — it makes them follow explicit rules, not "ethical"

### What's Hype
- ❌ "Aligned models understand human values" — they learned preference patterns
- ❌ "We've solved the alignment problem" — open problems remain fundamental
- ❌ "Self-correction will solve reliability" — proven to degrade performance without external signal
- ❌ "More RLHF training = more aligned" — diminishing returns and reward hacking

---

## Appendix: Full Citation List for Section 3

1. Christiano et al., 2017, "Deep Reinforcement Learning from Human Preferences," arXiv:1706.03741
2. Stiennon et al., 2020, "Learning to Summarize from Human Feedback," arXiv:2009.01325
3. **Mishra et al., 2021, "Cross-Task Generalization via Natural Language Crowdsourcing Instructions," arXiv:2104.08773** (Natural Instructions — first instruction-tuning)
4. Ouyang et al., 2022, "Training Language Models to Follow Instructions with Human Feedback," arXiv:2203.02155 (InstructGPT)
5. Bai et al., 2022, "Constitutional AI: Harmlessness from AI Feedback," arXiv:2212.08073 (Anthropic)
6. **Parmar, Mishra, Geva, Baral, 2023, "Don't Blame the Annotator: Bias Already Starts in the Annotation Instructions," EACL 2023, arXiv:2205.00415** [🏆 Outstanding Paper Award]
7. **Wang, Mishra et al., 2022, "Super-NaturalInstructions," EMNLP 2022, arXiv:2204.07705**
8. Rafailov et al., 2023, "Direct Preference Optimization," arXiv:2305.18290 (DPO)
9. **Wang, Mishra et al., 2023, "Self-Instruct," ACL 2023, arXiv:2212.10560**
10. **Huang, Chen, Mishra, Zheng, Yu, Song, Zhou, 2023, "Large Language Models Cannot Self-Correct Reasoning Yet," ICLR 2024, arXiv:2310.01798** (Google DeepMind)
11. Lee et al., 2023, "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback," arXiv:2309.00267
12. Casper et al., 2023, "Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback," arXiv:2307.15217
13. **Sachdeva, Mishra et al., 2022, "Generalized but not Robust?" ACL Findings 2022**
14. **Sachdeva, Mishra et al., 2022, "Is High Quality Data All We Need?" arXiv:2203.06404**
15. **Arunkumar, Mishra, Sachdeva et al., 2023, "VAIDA," EACL 2023, arXiv:2302.04434**
16. **Zhou, Mishra et al., 2024, "Self-Discover," arXiv:2402.03620** (Google DeepMind)

---

## Key Narrative Arc for Section 3

**The story to tell:**

1. **The problem:** Scale gave us power (GPT-3) but not control. 175B params and it still can't answer a simple question reliably.

2. **The first solution (your mentor's):** Teach models to follow instructions. Natural Instructions (2021) pioneered this — it's the SFT stage that RLHF builds on.

3. **The full pipeline:** RLHF (InstructGPT/ChatGPT) = instruction-tuning + reward modeling + RL. Each stage adds alignment signal.

4. **The simplification:** DPO removes a whole stage. Same results, less complexity. Science progresses by simplification.

5. **The limitations (honest):** Self-correction doesn't work (Mishra, ICLR 2024). Bias starts in the instructions themselves (Mishra, EACL 2023 Award). Data quality is the bottleneck (Sachdeva, Mishra).

6. **The frontier:** Self-Instruct → RLAIF → Self-Discover. The trend is reducing human dependency while acknowledging that EXTERNAL structure/feedback remains necessary.

**The through-line:** Alignment is not "solved" — it's a spectrum from "completely uncontrolled" to "mostly helpful." Your mentor's research defines both the techniques that work (instruction-tuning, structured reasoning) and the fundamental limits (self-correction failures, instruction bias).

---

*Document created: July 2, 2026*
*Last updated: July 2, 2026*
