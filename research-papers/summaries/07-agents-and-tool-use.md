# Agents & Tool Use — Summary

---

## Paper 1: ReAct (Yao et al., 2022)

**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models"  
**ArXiv:** [2210.03629](https://arxiv.org/abs/2210.03629)  
**Authors:** Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao (Princeton / Google)  

### The Problem
- LLMs can reason (Chain-of-Thought) but can't take actions
- LLMs can use tools but without explicit reasoning about why
- Combining reasoning + acting was done separately, not interleaved

### The Key Insight
Interleave **reasoning traces** (thinking) with **actions** (doing) in a single generation:

```
Question: What is the elevation of the birthplace of the inventor of the telephone?

Thought 1: I need to find who invented the telephone.
Action 1: Search[inventor of telephone]
Observation 1: Alexander Graham Bell invented the telephone.
Thought 2: Now I need to find where Bell was born.
Action 2: Search[Alexander Graham Bell birthplace]  
Observation 2: Bell was born in Edinburgh, Scotland.
Thought 3: Now I need the elevation of Edinburgh.
Action 3: Search[elevation Edinburgh Scotland]
Observation 3: Edinburgh has an elevation of 47 meters.
Thought 4: I have the answer.
Action 4: Finish[47 meters]
```

### Why This Works
1. **Reasoning → better action selection** (knows WHAT to search for)
2. **Actions → grounded reasoning** (observations correct hallucinations)
3. **Interpretable** — you can see why the model did what it did
4. **Self-correcting** — can adjust strategy based on observations

### Results
- HotpotQA: +6% accuracy over Chain-of-Thought alone
- FEVER fact verification: significant improvement
- Interactive decision-making (ALFWorld, WebShop): competitive with RL agents
- Reduced hallucination compared to CoT-only approaches

### Limitations (from recent analysis)
- Sensitive to prompt formatting (Fragile Foundations paper, 2024)
- Doesn't actually "reason" — it pattern-matches the interleaving format
- Performance degrades with complex multi-step tasks
- Error propagation: one bad observation can derail the chain

---

## The Agent Architecture Stack (2024-2026)

### Core Components of an LLM Agent

```
┌─────────────────────────────────┐
│         LLM (Brain)             │
│   Planning, reasoning, deciding │
├─────────────────────────────────┤
│      Memory                     │
│   Short-term: conversation      │
│   Long-term: vector store       │
├─────────────────────────────────┤
│      Tools                      │
│   Search, code exec, APIs,      │
│   databases, file systems       │
├─────────────────────────────────┤
│      Environment                │
│   Web, OS, applications         │
└─────────────────────────────────┘
```

### Agent Patterns

**1. Single-Agent (ReAct-style)**
- One LLM, loop of think-act-observe
- Best for: Simple tasks with clear tool boundaries
- Weakness: Gets confused on long tasks

**2. Multi-Agent (Debate/Collaboration)**
- Multiple LLMs with different roles (planner, executor, critic)
- Best for: Complex tasks needing diverse perspectives
- Weakness: Coordination overhead, increased cost

**3. Hierarchical Agent**
- Manager agent breaks task into subtasks
- Worker agents handle individual subtasks
- Best for: Large decomposable tasks
- Weakness: Manager must decompose correctly

---

## Tool Use Evolution (2022-2026)

### Phase 1: Function Calling (2023)
- OpenAI introduced structured function calling
- Model outputs JSON matching a schema
- Developer executes the function, returns result
- Limited to pre-defined function signatures

### Phase 2: Code Interpreter (2023-2024)
- Model writes and executes code directly
- Sandboxed execution environment
- Can iterate on code based on output
- More flexible than pre-defined tools

### Phase 3: Computer Use (2024-2025)
- Models can interact with GUIs (screenshots → actions)
- Anthropic's computer use, OpenAI's operator
- Can use any software a human can use
- Still brittle and slow compared to API-based tools

### Phase 4: Agentic RL (2025-2026)
- Models trained specifically for tool use via reinforcement learning
- Learn when and how to use tools through practice
- More robust than prompt-based approaches
- [arXiv:2505.01441](https://arxiv.org/abs/2505.01441) — "Agentic Reasoning and Tool Integration via RL"

---

## Hype vs. Reality

### What's Real
- Simple tool use (search, calculation, API calls) works reliably
- ReAct-style prompting improves factual accuracy over pure LLM
- Structured output / function calling is production-ready
- Agents can handle 2-3 step tasks with high success rate

### What's Overhyped
- "Autonomous agents that complete complex tasks end-to-end" → Error rates compound; 5-step tasks with 90% per-step accuracy = 59% end-to-end
- "Agents replace human workers" → They augment, not replace; need oversight
- "Just add more tools and the agent handles it" → More tools = more confusion about which to use
- "Planning capabilities" → LLMs don't truly plan; they generate plausible-looking plans
- "Self-correcting" → Models often can't identify their own errors reliably

### What Actually Works in Production
1. Search-augmented generation (RAG + web search)
2. Code generation + execution for data analysis
3. Form filling and structured data extraction
4. Simple workflow automation (2-3 steps)
5. Human-in-the-loop agent systems

### What Doesn't Work Yet
1. Long-horizon planning (10+ steps)
2. Novel problem-solving without human guidance
3. Reliable self-evaluation and error detection
4. Agents operating in adversarial environments
5. Tasks requiring real-world state tracking

---

## Visual Suggestions
- [DIAGRAM] ReAct loop: Thought → Action → Observation cycle
- [DIAGRAM] Agent architecture stack (layered diagram)
- [CHART] Success rate vs. number of steps (exponential decay)
- [TIMELINE] Tool use evolution: 2022 → 2026
- [COMPARISON] What works vs. what doesn't in production agents
