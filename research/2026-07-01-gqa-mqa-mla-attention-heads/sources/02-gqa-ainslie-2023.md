---
url: https://arxiv.org/abs/2305.13245
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
type: arxiv_paper
authors: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai
year: 2023
accessed: 2026-07-01
quality: 5
relevance: core
---

## Abstract

Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up
decoder inference. However, MQA can lead to quality degradation, and moreover it may not be
desirable to train a separate model just for faster inference. We (1) propose a recipe for
uptraining existing multi-head language model checkpoints into models with MQA using 5% of
original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization
of multi-query attention which uses an intermediate (more than one, less than number of query
heads) number of key-value heads. We show that uptrained GQA achieves quality close to
multi-head attention with comparable speed to MQA.

## Full-text extracted content (via arxiv PDF)

### Method

Grouped-query attention divides query heads into G groups, each of which shares a single key
head and value head. GQA-G refers to grouped-query with G groups.
- GQA-1 (single group, single K/V head) is equivalent to MQA.
- GQA-H (groups = number of heads) is equivalent to MHA.
- Intermediate G interpolates: higher quality than MQA, faster than MHA.

Rationale: as models scale, the number of heads generally scales too, so MQA represents an
increasingly aggressive cut in both memory bandwidth AND representational capacity as models
grow. GQA lets the KV cache shrink proportionally to model size instead of collapsing to a
single head regardless of scale. Also: larger models suffer relatively less from memory
bandwidth overhead since KV cache scales with model dimension while FLOPs/params scale with the
square of model dimension — so GQA is expected to be a particularly good trade-off for larger
models. GQA is NOT applied to encoder self-attention (encoder is parallel, not bandwidth-bound).

Uptraining recipe: convert an existing MHA checkpoint by MEAN-POOLING the K/V projection
matrices within each group (works better than "select first head" or "random init" — ablation:
Mean > First > Random). Then continue pretraining for a small proportion α of original steps
(5% recommended; diminishing returns beyond 10%).

### Key experimental results (T5 Large/XXL, uptrained with α=0.05)

- Uptrained MQA-XXL: favorable trade-off vs MHA-Large — higher quality AND faster inference.
- Uptrained GQA-8-XXL: significant additional quality gains over MQA, achieving performance
  close to MHA-XXL while retaining speed close to MQA.
- GQA already gets reasonable performance right after checkpoint conversion (before uptraining);
  MQA requires uptraining to be usable at all.
- Ablation on number of groups: for larger models, increasing groups from MQA causes only modest
  slowdown initially, increasing cost as approaching MHA. 8 groups selected as favorable middle
  ground for T5-XXL.

### Relation to later work

This paper explicitly frames the KV cache size formula: MHA needs O(H) key/value tensors per
token, MQA needs O(1), GQA needs O(G) where 1 ≤ G ≤ H.
