---
url: https://arxiv.org/abs/2005.14165
title: "Language Models are Few-Shot Learners (GPT-3)"
type: arxiv_paper
authors: Tom Brown et al.
year: 2020
quality: 5
relevance: core
---

# GPT-3 Architecture

Autoregressive decoder-only transformer:
- **175B parameters** (10x larger than any previous non-sparse model)
- 96 transformer decoder layers
- 96 attention heads per layer, each 128 dimensions
- Hidden dimension: 12,288
- FFN inner dimension: 49,152 (4x hidden)
- Context window: 2,048 tokens
- Vocabulary: 50,257 (BPE)
- Learned positional embeddings
- Pre-norm architecture (LayerNorm before attention/FFN)

## Training Data (~300B tokens)
- Filtered Common Crawl (410B → 45TB filtered)
- WebText2
- Books1, Books2
- Wikipedia

## Key Contribution
Demonstrated emergent in-context learning: zero/few-shot task performance without gradient updates. Established that scale enables generalization through prompting rather than fine-tuning.
