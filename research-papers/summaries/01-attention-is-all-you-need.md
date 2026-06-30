# Attention Is All You Need — Summary

**Paper:** "Attention Is All You Need"  
**Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google Brain / Google Research)  
**Year:** 2017  
**ArXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)  
**Published at:** NeurIPS 2017  

---

## The Problem Before This Paper
- RNNs (LSTMs, GRUs) processed sequences one token at a time → couldn't parallelize
- Long-range dependencies were hard to learn (gradient vanishing/exploding)
- Training was slow because of sequential nature
- ConvNets for sequences (like ByteNet) had limited receptive fields

## The Key Insight
You don't need recurrence or convolutions for sequence modeling. **Self-attention** can directly relate any two positions in a sequence in O(1) operations, regardless of distance. This makes training massively parallelizable.

## How It Works

### Architecture Components:
1. **Encoder-Decoder structure** (6 layers each)
2. **Multi-Head Self-Attention:** Instead of one attention function, run multiple attention "heads" in parallel, each learning different relationships
3. **Positional Encoding:** Since no recurrence, inject position info via sine/cosine functions
4. **Feed-Forward Networks:** Two linear layers with ReLU between, applied identically to each position
5. **Residual Connections + Layer Normalization:** Around each sub-layer

### The Attention Formula:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
- Q (Query): What am I looking for?
- K (Key): What do I contain?
- V (Value): What information do I provide?
- √d_k: Scaling factor to prevent softmax saturation

### Multi-Head Attention:
- Split Q, K, V into h heads (paper uses h=8)
- Each head has its own learned linear projections
- Concatenate outputs, project again
- Different heads can attend to different types of relationships

## Real Results
- BLEU 28.4 on English-German translation (beating all previous models)
- BLEU 41.8 on English-French (new SOTA at 1/4 training cost of previous best)
- Training time: 3.5 days on 8 P100 GPUs (vs. weeks for RNN models)

## Limitations
- Self-attention is O(n²) in sequence length — problematic for very long sequences
- Positional encoding is somewhat ad-hoc (learned vs. sinusoidal debate continues)
- No inherent sequential bias — must learn order entirely from positional encodings
- Encoder-decoder architecture is over-specified for many tasks (later models use decoder-only)

## What This Led To
- BERT (encoder-only, 2018)
- GPT series (decoder-only, 2018-present)
- All modern LLMs are transformers
- Spawned research into efficient attention (linear attention, sparse attention, etc.)

## Visual Suggestions
- [DIAGRAM] Multi-head attention mechanism showing Q, K, V flow
- [DIAGRAM] Full encoder-decoder architecture
- [COMPARISON] RNN vs. Transformer training parallelization
- [FORMULA] Attention equation with visual annotation

## Source for Visuals
- Original paper Figure 1 (architecture) and Figure 2 (attention mechanisms)
- [SOURCE: https://arxiv.org/abs/1706.03762] — Paper figures are the canonical reference
