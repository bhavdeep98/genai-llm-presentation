# Generative AI & LLMs — Research-Grounded Presentation Project

## Purpose
Build a comprehensive, visually-driven presentation on LLM architectures, fine-tuning, LoRA, and API integrations — grounded in original research papers, not hype.

## Project Structure

```
demo prep/
├── .kiro/steering/          ← Project guidelines & standards
│   ├── project-goals.md     ← Overall objectives and flow
│   ├── presentation-style.md ← Visual & content style guide
│   └── research-standards.md ← Source verification standards
│
├── research-papers/          ← Paper catalog & summaries
│   ├── paper-catalog.md      ← Master list of all papers with links
│   ├── download-links.md     ← Direct PDF download URLs
│   └── summaries/            ← One-page summaries per topic
│       ├── 01-attention-is-all-you-need.md
│       ├── 02-gpt-evolution.md
│       ├── 03-scaling-laws.md
│       ├── 04-alignment-rlhf-dpo.md
│       ├── 05-lora-and-variants.md
│       ├── 06-rag-and-apis.md
│       ├── 07-agents-and-tool-use.md
│       ├── 08-alternative-architectures.md
│       └── 09-hype-vs-reality.md
│
├── presentation/             ← Final presentation content
│   └── presentation-outline.md ← 34-slide outline with timing
│
├── notes/                    ← Working notes for learning
│   ├── learning-path.md      ← Step-by-step conceptual understanding
│   └── visual-references.md  ← All visuals needed with sources
│
├── references/               ← Verified external sources
│   └── verified-sources.md   ← All citations, tiered by reliability
│
└── README.md                 ← This file
```

## How to Use This Project

### Step 1: Learn (Start Here)
1. Read `notes/learning-path.md` — understand the chain of ideas
2. Read summaries in order (01 → 09)
3. Self-test with the questions at the end of learning-path.md

### Step 2: Deep Dive (Download Papers)
1. Use `research-papers/download-links.md` to get PDFs
2. Read papers in the recommended order from learning-path.md
3. Focus on the sections highlighted in each summary

### Step 3: Build the Presentation
1. Follow `presentation/presentation-outline.md`
2. Use `notes/visual-references.md` for all diagrams/images
3. Cite from `references/verified-sources.md`

### Step 4: Filter for the Audience
- Every slide follows: Problem → Insight → Paper → How → Results → Limits
- Use the Hype vs. Reality framework for critical thinking
- End with inspiration, not overwhelm

## Key Papers (Quick Reference)
| Paper | What It Introduced | Year |
|-------|-------------------|------|
| Attention Is All You Need | Transformer architecture | 2017 |
| GPT-1/2/3 | Pre-train → Scale → In-context learning | 2018-2020 |
| Scaling Laws / Chinchilla | Principled scaling decisions | 2020/2022 |
| InstructGPT | RLHF alignment pipeline | 2022 |
| DPO | Simpler alternative to RLHF | 2023 |
| LoRA | Parameter-efficient fine-tuning | 2021 |
| QLoRA | Fine-tuning on consumer hardware | 2023 |
| RAG | Retrieval-augmented generation | 2020 |
| ReAct | LLM agents with tool use | 2022 |
| Mamba | Linear-time alternative to transformers | 2023 |

## Status
- [x] Steering documents created
- [x] Research papers cataloged (20 papers, verified sources)
- [x] Paper summaries written (9 topic areas)
- [x] Presentation outline created (34 slides, ~87 min)
- [x] Visual references documented
- [x] Learning path for self-study
- [x] Hype vs. Reality analysis
- [ ] Download all PDFs to local folder
- [ ] Create actual slide visuals
- [ ] Final presentation file (PPTX/Keynote/PDF)
