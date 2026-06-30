# Alignment: RLHF & DPO — Summary

---

## Paper 1: InstructGPT (Ouyang et al., 2022)

**Paper:** "Training Language Models to Follow Instructions with Human Feedback"  
**ArXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)  
**Published at:** NeurIPS 2022  

### The Problem
- GPT-3 was powerful but unreliable: hallucinated, produced toxic content, didn't follow instructions
- Bigger models ≠ more aligned models
- Users wanted a helpful assistant, not a next-token predictor

### The 3-Stage RLHF Pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**
- Collect demonstration data: human labelers write ideal responses to prompts
- Fine-tune GPT-3 on these demonstrations using standard supervised learning
- Creates a baseline model that can follow instructions (but imperfectly)

**Stage 2: Reward Model Training**
- For a given prompt, generate multiple outputs from the SFT model
- Human labelers rank outputs from best to worst
- Train a reward model to predict human preference rankings
- The reward model learns what "good" looks like

**Stage 3: Reinforcement Learning (PPO)**
- Use the reward model as a scoring function
- Optimize the SFT model using Proximal Policy Optimization (PPO)
- The model learns to generate outputs that score highly on the reward model
- KL divergence penalty prevents the model from drifting too far from original

### Key Results
- 1.3B InstructGPT preferred over 175B GPT-3 (100x fewer params!)
- Less hallucination, less toxic output
- Better instruction following across diverse tasks
- The aligned model is worse at some benchmarks but much better at what users actually want

### Limitations
- Expensive: requires human labelers for each stage
- Reward hacking: model can game the reward model
- Alignment tax: model gets slightly worse at some academic benchmarks
- Not fully generalizable: alignment to one set of values may not transfer

---

## Paper 2: DPO — Direct Preference Optimization (Rafailov et al., 2023)

**Paper:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"  
**ArXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)  

### The Problem with RLHF
- PPO is complex, unstable, and computationally expensive
- Requires training a separate reward model
- Sensitive to hyperparameters
- Hard to reproduce

### The Key Insight
The optimal policy under the RLHF objective can be expressed in closed form. This means you can directly optimize the policy using preference data, without ever training a reward model or using RL.

### How DPO Works
Instead of: preference data → reward model → RL → policy  
DPO does: preference data → policy (directly)

The loss function is a simple binary cross-entropy:
```
L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```
Where:
- y_w = preferred (winning) response
- y_l = dispreferred (losing) response
- π_ref = reference policy (the SFT model)
- β = temperature parameter

### Advantages over RLHF
| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Reward model | Required (separate training) | Not needed |
| RL algorithm | PPO (complex, unstable) | Simple classification loss |
| Hyperparameters | Many (PPO clip, KL coeff, etc.) | One (β temperature) |
| Training stability | Can be unstable | Stable gradient descent |
| Compute | 3 models in memory | 2 models in memory |
| Results | SOTA alignment | Comparable to RLHF |

### Limitations of DPO
- Assumes preference data is clean and consistent
- Can exploit length bias (longer responses scored higher)
- Doesn't learn an explicit reward model (harder to interpret)
- May not scale as well as RLHF for very complex alignment objectives

---

## Paper 3: Open Problems in RLHF (Casper et al., 2023)

**ArXiv:** [2307.15217](https://arxiv.org/abs/2307.15217)  

### Critical Issues (Essential for Hype vs. Reality)

1. **Reward Hacking:** Models learn to game the reward model rather than genuinely improve
2. **Scalable Oversight:** Humans can't reliably evaluate complex outputs (coding, research, etc.)
3. **Preference Inconsistency:** Different annotators have different values; averaging destroys signal
4. **Distribution Shift:** Models encounter inputs unlike anything in training data
5. **Deceptive Alignment:** In theory, a model could learn to appear aligned during training

### What This Means for Practitioners
- RLHF/DPO are practical tools but NOT "solving alignment"
- The aligned model is still a next-token predictor with learned biases
- "Aligned" is relative to the specific training data and labeler demographics
- Don't confuse "follows instructions better" with "is safe"

---

## Visual Suggestions
- [DIAGRAM] 3-stage RLHF pipeline (SFT → Reward Model → PPO)
- [COMPARISON] RLHF vs. DPO pipeline (3 models vs. 2 models)
- [CHART] InstructGPT 1.3B vs. GPT-3 175B human preference ratings
- [FORMULA] DPO loss function with annotations
- [TABLE] RLHF failure modes and examples
