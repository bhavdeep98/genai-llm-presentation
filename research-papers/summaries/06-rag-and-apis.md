# RAG & API Integrations — Summary

---

## Paper 1: RAG — Retrieval-Augmented Generation (Lewis et al., 2020)

**Paper:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"  
**ArXiv:** [2005.11401](https://arxiv.org/abs/2005.11401)  
**Authors:** Lewis, Perez, Piktus, Petroni, Karpukhin, et al. (Meta AI)  

### The Problem
- LLMs store knowledge in parameters → fixed at training time
- Can't update knowledge without retraining
- Hallucinate when asked about things not in training data
- Struggle with rare or specialized facts

### The Key Insight
Don't try to memorize everything — **retrieve relevant information at inference time** and condition generation on it.

Combine two pre-trained components:
1. **Retriever:** Dense Passage Retrieval (DPR) — finds relevant documents
2. **Generator:** BART/T5 — generates answers conditioned on retrieved docs

### How RAG Works

```
Query → Retriever → Top-k Documents → [Query + Documents] → Generator → Answer
```

Two variants:
- **RAG-Token:** Can retrieve different documents for each output token
- **RAG-Sequence:** Retrieves once, conditions entire generation on retrieved docs

### Architecture Details
1. Query encoding: Encode input question with BERT-based encoder
2. Document retrieval: Maximum Inner Product Search (MIPS) against document index
3. Document encoding: Pre-encoded passages stored in FAISS index
4. Generation: Generator sees [question; retrieved_passage_1; ...; retrieved_passage_k]
5. Marginalization: Probability of output marginalizes over retrieved documents

### Results
- SOTA on open-domain QA (Natural Questions, TriviaQA, WebQuestions)
- Generates more factual and specific text than pure parametric models
- Can update knowledge by updating the document index (no retraining!)
- More interpretable: you can see what was retrieved

### Limitations
- Retrieval quality caps generation quality
- Added latency from retrieval step
- Document index must be maintained and kept current
- Doesn't help with reasoning tasks (only factual recall)
- "Garbage in, garbage out" — bad retrieval leads to bad generation

---

## Modern RAG Architecture (2024-2026)

### Beyond the Original Paper

The field has evolved significantly:

**Advanced Retrieval:**
- Hybrid retrieval (dense + sparse/BM25)
- Reranking (cross-encoder after initial retrieval)
- Query decomposition (break complex queries into sub-queries)
- HyDE: Generate hypothetical answers to retrieve against

**Advanced Generation:**
- Iterative retrieval (retrieve → generate → retrieve again)
- Self-RAG: Model decides when to retrieve
- Corrective RAG (CRAG): Verify retrieval quality before using

**Chunking Strategies:**
- Fixed-size chunks with overlap
- Semantic chunking (by topic/section)
- Recursive splitting (hierarchical)
- Document-level summaries + chunk-level detail

---

## API Integration Patterns

### Pattern 1: Direct API Call
```
User → LLM API → Response
```
Simplest. Use for: chatbots, content generation, classification.

### Pattern 2: RAG Pipeline
```
User → Embedding API → Vector DB → Top-k Docs → LLM API (with context) → Response
```
Use for: Q&A over private data, customer support, documentation bots.

### Pattern 3: Agent with Tools
```
User → LLM (reasoning) → Tool Call → Execute → Observe → LLM (next step) → Response
```
Use for: Complex tasks requiring external systems, multi-step workflows.

### Pattern 4: Chain of Calls
```
User → LLM (plan) → LLM (execute step 1) → LLM (execute step 2) → Final Output
```
Use for: Complex reasoning, multi-step generation, self-verification.

---

## Practical API Integration Considerations

### Cost Structure
| Provider | Input (per 1M tokens) | Output (per 1M tokens) | Context Window |
|----------|----------------------|------------------------|----------------|
| OpenAI GPT-4o | ~$2.50 | ~$10 | 128K |
| Anthropic Claude 3.5 | ~$3 | ~$15 | 200K |
| Google Gemini 1.5 | ~$1.25 | ~$5 | 1M |
| Open models (self-hosted) | Compute cost only | - | Varies |

*Note: Prices are approximate and change frequently. Verify current pricing.*

### Latency Considerations
- Direct API: 1-30s depending on output length
- RAG pipeline: +200-500ms for retrieval
- Agent loops: Multiply by number of steps
- Streaming: First token in ~200ms, then continuous

### Rate Limits & Reliability
- All APIs have rate limits (tokens/min, requests/min)
- Need retry logic with exponential backoff
- Consider fallback providers
- Cache frequent queries

---

## Hype vs. Reality

### What's Real
- RAG genuinely reduces hallucination for factual queries
- API-based development is accessible and practical
- Tool-use agents can accomplish complex multi-step tasks
- Embedding + vector search is a proven retrieval pattern

### What's Overhyped
- "RAG solves hallucination" → It reduces it for retrieval-backed facts, not for reasoning
- "Agents can do anything" → They fail on long chains, compound errors multiply
- "Just add RAG" → Retrieval quality determines 80% of final quality
- "Vector search is magic" → It's approximate nearest neighbor search; embeddings have blind spots
- "100K context windows replace RAG" → Long context loses precision; retrieval is still better for targeted facts

### What Actually Works in Production
1. RAG for document Q&A (with good chunking + reranking)
2. Structured output from APIs (JSON mode, function calling)
3. Simple tool-use (1-2 tools, well-defined actions)
4. Classification and extraction tasks
5. Code generation with human review

### What Doesn't Work Well Yet
1. Fully autonomous agents on complex tasks (too many failure modes)
2. RAG over highly technical/mathematical content (embedding models struggle)
3. Multi-hop reasoning chains without human oversight
4. Real-time systems requiring <100ms latency

---

## Visual Suggestions
- [DIAGRAM] RAG architecture: query → retriever → generator flow
- [COMPARISON] Pure LLM vs. RAG for factual questions (accuracy chart)
- [DIAGRAM] API integration patterns (4 patterns as flow diagrams)
- [TABLE] Cost comparison across providers
- [DIAGRAM] Agent loop: Reason → Act → Observe cycle
