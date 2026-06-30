# Research Paper Catalog

This document catalogs all foundational and key papers for our presentation, organized by topic.
Each entry includes: title, authors, year, arXiv/source link, and a brief note on why it matters.

---

## 1. Transformer Architecture (Foundation)

### 1.1 Attention Is All You Need
- **Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
- **Year:** 2017
- **ArXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)
- **Published at:** NeurIPS 2017
- **Why it matters:** Introduced the Transformer architecture — self-attention mechanism replacing RNNs entirely. This is the foundation of ALL modern LLMs.
- **Key insight:** Attention mechanisms alone (without recurrence or convolution) are sufficient for sequence-to-sequence tasks, enabling massive parallelization.

---

## 2. GPT Series (Scaling Language Models)

### 2.1 GPT-1: Improving Language Understanding by Generative Pre-Training
- **Authors:** Radford, Narasimhan, Salimans, Sutskever (OpenAI)
- **Year:** 2018
- **Source:** [OpenAI Blog](https://openai.com/index/language-unsupervised/)
- **Parameters:** 117M, 12-layer decoder-only Transformer
- **Why it matters:** First demonstration that unsupervised pre-training + supervised fine-tuning on a decoder-only transformer could beat task-specific models. Established the "pre-train then fine-tune" paradigm.

### 2.2 GPT-2: Language Models are Unsupervised Multitask Learners
- **Authors:** Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI)
- **Year:** 2019
- **Source:** [OpenAI Blog](https://openai.com/research/better-language-models)
- **Parameters:** 1.5B
- **Why it matters:** Showed that scaling up pre-training data and model size leads to emergent zero-shot capabilities. The model could do tasks it was never explicitly trained for.

### 2.3 GPT-3: Language Models are Few-Shot Learners
- **Authors:** Brown, Mann, Ryder, Subbiah, Kaplan, et al. (OpenAI)
- **Year:** 2020
- **ArXiv:** [2005.14165](https://arxiv.org/abs/2005.14165)
- **Parameters:** 175B
- **Why it matters:** Demonstrated that massive scale enables in-context learning (few-shot prompting). No gradient updates needed — just examples in the prompt. This changed how people thought about using language models.

---

## 3. Scaling Laws

### 3.1 Scaling Laws for Neural Language Models
- **Authors:** Kaplan, McCandlish, Henighan, Brown, et al. (OpenAI)
- **Year:** 2020
- **ArXiv:** [2001.08361](https://arxiv.org/abs/2001.08361)
- **Why it matters:** Empirically showed that loss scales as a power law with model size, dataset size, and compute. Provided a principled way to predict model performance before training.

### 3.2 Training Compute-Optimal Large Language Models (Chinchilla)
- **Authors:** Hoffmann, Borgeaud, Mensch, et al. (DeepMind)
- **Year:** 2022
- **ArXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)
- **Why it matters:** Showed previous models were over-parameterized and under-trained. For compute-optimal training, model size and training tokens should scale equally. Led to smaller but better-trained models.

---

## 4. Alignment & Instruction Following

### 4.1 InstructGPT: Training Language Models to Follow Instructions with Human Feedback
- **Authors:** Ouyang, Wu, Jiang, Almeida, et al. (OpenAI)
- **Year:** 2022
- **ArXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)
- **Published at:** NeurIPS 2022
- **Why it matters:** Introduced the 3-stage RLHF pipeline (SFT → Reward Model → PPO). A 1.3B InstructGPT was preferred over 175B GPT-3 by human evaluators. Proved alignment > scale for user-facing quality.

### 4.2 Direct Preference Optimization (DPO)
- **Authors:** Rafailov, Sharma, Mitchell, et al. (Stanford)
- **Year:** 2023
- **ArXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)
- **Why it matters:** Showed you can skip the reward model entirely — reparameterize RLHF as a simple classification loss. Dramatically simpler than PPO-based RLHF while achieving comparable results.

### 4.3 Open Problems and Fundamental Limitations of RLHF
- **Authors:** Casper, Davies, Shi, et al.
- **Year:** 2023
- **ArXiv:** [2307.15217](https://arxiv.org/abs/2307.15217)
- **Why it matters:** Critical analysis of RLHF limitations — reward hacking, scalable oversight problems, distributional challenges. Essential for our "hype vs. reality" section.

---

## 5. Parameter-Efficient Fine-Tuning

### 5.1 LoRA: Low-Rank Adaptation of Large Language Models
- **Authors:** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (Microsoft)
- **Year:** 2021
- **ArXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)
- **Why it matters:** Core innovation: freeze pre-trained weights, inject trainable low-rank matrices (A×B decomposition) into transformer layers. Reduces trainable params by 10,000x while matching full fine-tuning quality. No inference latency added.

### 5.2 QLoRA: Efficient Finetuning of Quantized LLMs
- **Authors:** Dettmers, Pagnoni, Holtzman, Zettlemoyer (UW)
- **Year:** 2023
- **ArXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)
- **Why it matters:** Combined 4-bit quantization (NF4 data type) with LoRA. Fine-tune a 65B model on a single 48GB GPU. Introduced double quantization and paged optimizers. Democratized LLM fine-tuning.

### 5.3 LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters
- **Authors:** (Multiple authors)
- **Year:** 2024
- **ArXiv:** [2405.17604](https://arxiv.org/abs/2405.17604)
- **Why it matters:** Reduces LoRA parameters by 100x using SVD-derived frozen matrices with a tiny trainable matrix between them. Pushes the efficiency frontier further.

### 5.4 Low-Rank Adaptation for Foundation Models (Survey)
- **Authors:** (Multiple authors)
- **Year:** 2025
- **ArXiv:** [2501.00365](https://arxiv.org/abs/2501.00365)
- **Why it matters:** Comprehensive survey of LoRA and all its variants — useful for understanding the full landscape of parameter-efficient methods.

---

## 6. Retrieval-Augmented Generation (RAG)

### 6.1 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Authors:** Lewis, Perez, Piktus, et al. (Meta AI)
- **Year:** 2020
- **ArXiv:** [2005.11401](https://arxiv.org/abs/2005.11401)
- **Why it matters:** Original RAG paper. Combines a pre-trained retriever (DPR) with a pre-trained generator (BART). The model retrieves relevant documents and conditions generation on them — reducing hallucination and enabling knowledge updates without retraining.

### 6.2 Retrieval-Augmented Generation for AI-Generated Content (Survey)
- **Authors:** (Multiple authors)
- **Year:** 2024
- **ArXiv:** [2402.19473](https://arxiv.org/abs/2402.19473)
- **Why it matters:** Comprehensive survey of RAG approaches across modalities — covers advanced retrieval, generation, and augmentation strategies.

---

## 7. Agents & Tool Use

### 7.1 ReAct: Synergizing Reasoning and Acting in Language Models
- **Authors:** Yao, Zhao, Yu, et al. (Princeton/Google)
- **Year:** 2022
- **ArXiv:** [2210.03629](https://arxiv.org/abs/2210.03629)
- **Why it matters:** Introduced interleaving reasoning traces with actions. LLMs can "think" about what to do, then execute tools, observe results, and iterate. Foundation for modern LLM agents.

### 7.2 The Evolution of Tool Use in LLM Agents
- **Authors:** (Multiple authors)
- **Year:** 2025
- **ArXiv:** [2603.22862](https://arxiv.org/abs/2603.22862)
- **Why it matters:** Latest survey on how tool use has evolved from single-tool selection to complex multi-step agent workflows.

---

## 8. Alternative Architectures

### 8.1 Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors:** Gu, Dao
- **Year:** 2023
- **ArXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)
- **Why it matters:** Proposed selective state space models as a transformer alternative. Linear-time complexity vs. quadratic attention. Mamba-3B matches transformers twice its size on language modeling. Challenges the "transformers are the only way" assumption.

### 8.2 The Illusion of State in State-Space Models
- **Authors:** (Multiple authors)
- **Year:** 2024
- **ArXiv:** [2404.08819](https://arxiv.org/abs/2404.08819)
- **Why it matters:** Critical analysis showing SSMs have similar expressiveness limitations to transformers for state-tracking. Important for "hype vs. reality" — SSMs aren't a magic bullet.

### 8.3 Mixture of Experts (MoE) in LLMs
- **Authors:** Various (Shazeer et al. 2017 original, recent: DeepSeek, Mixtral)
- **Year:** 2017-2025
- **Key papers:** 
  - Shazeer et al. "Outrageously Large Neural Networks" (2017)
  - Mixtral (Mistral AI, 2024)
  - DeepSeek-V2/V3 (2024-2025)
- **ArXiv Survey:** [2507.11181](https://arxiv.org/abs/2507.11181)
- **Why it matters:** Sparse activation — only a subset of parameters active per token. Enables scaling total parameters without proportional compute cost. DeepSeek-V3 uses MoE extensively. This is how the industry now scales beyond dense transformers.

---

## 9. Latest Developments (2025-2026)

### 9.1 Next Generation LLM Architecture
- **Source:** [Stanford - Jingke Sun](https://web.stanford.edu/~jksun/blog/llm-architecture.html)
- **Why it matters:** Analysis of what's next — questioning whether standard components (softmax router, pre-norm, attention patterns) are optimal.

### 9.2 Multi-head Latent Attention (MLA) + MoE
- **Source:** [arXiv 2507.15465](https://arxiv.org/abs/2507.15465)
- **Why it matters:** DeepSeek's MLA + MoE combination challenges the need for specialized attention hardware. Represents the current frontier.

---

## Papers to Download (Priority Order)

1. ✅ Attention Is All You Need (1706.03762)
2. ✅ GPT-3: Language Models are Few-Shot Learners (2005.14165)
3. ✅ LoRA (2106.09685)
4. ✅ QLoRA (2305.14314)
5. ✅ InstructGPT (2203.02155)
6. ✅ DPO (2305.18290)
7. ✅ Scaling Laws (2001.08361)
8. ✅ Chinchilla (2203.15556)
9. ✅ RAG Original (2005.11401)
10. ✅ ReAct (2210.03629)
11. ✅ Mamba (2312.00752)
12. ✅ RLHF Limitations (2307.15217)
