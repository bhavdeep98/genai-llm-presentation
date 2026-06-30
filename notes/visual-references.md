# Visual References — Images & Diagrams to Include

This document lists all visuals needed for the presentation with their sources and suggested image search terms.

---

## Priority 1: Diagrams from Original Papers (Cite & Reference)

| # | Visual | Source | Paper Figure | Usage |
|---|--------|--------|-------------|-------|
| 1 | Transformer architecture (encoder-decoder) | Vaswani et al. 2017 | Figure 1 | Section 1, Slide 6 |
| 2 | Scaled dot-product attention & multi-head | Vaswani et al. 2017 | Figure 2 | Section 1, Slide 4-5 |
| 3 | Scaling law curves (loss vs. compute) | Kaplan et al. 2020 | Figure 1 | Section 2, Slide 10 |
| 4 | InstructGPT RLHF 3-stage pipeline | Ouyang et al. 2022 | Figure 2 | Section 3, Slide 13 |
| 5 | LoRA mechanism (frozen W + low-rank path) | Hu et al. 2021 | Figure 1 | Section 4, Slide 18 |
| 6 | RAG architecture diagram | Lewis et al. 2020 | Figure 1 | Section 5, Slide 23 |

---

## Priority 2: Diagrams to Create

| # | Visual | Description | Slide |
|---|--------|-------------|-------|
| 1 | GPT parameter growth | Log-scale chart: 117M → 1.5B → 175B → ~1.8T | Slide 9 |
| 2 | Chinchilla vs. Gopher | Bar chart comparing performance (same compute) | Slide 11 |
| 3 | RLHF vs. DPO pipeline | Side-by-side: 3-stage vs. 2-stage | Slide 14 |
| 4 | Memory comparison | Bar chart: Full FT / LoRA / QLoRA memory usage | Slide 20 |
| 5 | LoRA family tree | Tree diagram showing variants | Slide 21 |
| 6 | API integration patterns | 4 flow diagrams (Direct/RAG/Agent/Chain) | Slide 22 |
| 7 | Agent loop | ReAct cycle: Think → Act → Observe | Slide 24 |
| 8 | Error compounding chart | Line graph: success rate vs. steps | Slide 25 |
| 9 | Architecture landscape | Map of current architectures (2026) | Slide 26 |
| 10 | MoE routing diagram | Tokens → Router → Expert selection | Slide 27 |
| 11 | O(n²) vs O(n) scaling | Line graph comparing Transformer vs. SSM | Slide 28 |
| 12 | Hype vs. Reality table | 3-column visual summary | Slide 30 |
| 13 | Paper vs. Social media | Side-by-side showing distortion | Slide 31 |
| 14 | Innovation chain | The full "chain of ideas" timeline | Slide 32 |

---

## Priority 3: Reference Images (Search & Include with Attribution)

### Suggested Image Search Terms

| Concept | Search Terms | Notes |
|---------|-------------|-------|
| Attention heatmap | "transformer attention visualization heatmap" | Shows which words attend to which |
| RNN vs Transformer | "RNN sequential vs transformer parallel processing" | Before/after comparison |
| GPT model size comparison | "GPT-1 GPT-2 GPT-3 GPT-4 parameters comparison chart" | Scale visualization |
| Token prediction | "next token prediction language model illustration" | Core concept visual |
| Vector embedding space | "word embedding vector space visualization" | For RAG section |
| FAISS/vector search | "vector database similarity search diagram" | RAG retrieval visual |
| MoE expert routing | "mixture of experts routing diagram transformer" | Architecture section |
| Mamba SSM | "mamba state space model architecture diagram" | Alternative arch. |

---

## Statistics & Data to Visualize

### Performance Comparisons (from papers)
- Transformer BLEU scores vs. RNN baseline (Vaswani 2017)
- InstructGPT 1.3B preference rate vs. GPT-3 175B (Ouyang 2022)
- LoRA: 10,000x parameter reduction with equivalent performance (Hu 2021)
- QLoRA: 65B model on 48GB GPU (Dettmers 2023)
- Chinchilla 70B vs. Gopher 280B benchmark scores (Hoffmann 2022)
- Mamba-3B vs. Transformer-3B and Transformer-6B (Gu & Dao 2023)

### Cost/Efficiency Data
- Training cost estimates (GPT-3 ~$4.6M in 2020)
- LoRA adapter sizes (MBs vs. GBs for full model)
- API pricing comparison (approximate, verify current)
- Memory requirements per model size

---

## Tool Suggestions for Creating Visuals

1. **Architecture diagrams:** draw.io, Excalidraw, or TikZ (for paper-quality)
2. **Charts & graphs:** Python matplotlib/seaborn, or Google Sheets
3. **Comparison tables:** PowerPoint/Keynote built-in
4. **Timelines:** Mermaid.js, or manual in presentation tool
5. **Code snippets:** Use monospace font, syntax highlighting in slides

---

## Attribution Template

For every external visual used:
```
[SOURCE: {url}]
[Paper: {title}, {authors}, {year}]
[Figure {number}: {caption from paper}]
[Used under fair use for educational purposes]
```
