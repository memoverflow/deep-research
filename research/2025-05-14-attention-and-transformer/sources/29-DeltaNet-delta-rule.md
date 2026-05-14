---
url: https://arxiv.org/abs/2406.06484
title: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
type: arxiv_paper
authors: Songlin Yang et al.
year: 2024
quality: 4
relevance: high
---

# DeltaNet: Delta Rule for Linear Attention

## Key Innovation
Applies error-correction (delta rule) to linear attention state updates.

## Formulation
S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k_t^T

Where:
- (v_t − S_{t-1}k_t) = prediction error
- β_t = learning rate
- Equivalent to one step of online gradient descent on associative recall

## Key Insight
Write-and-OVERWRITE mechanism (vs just additive accumulation in standard linear attention).
Can correct outdated associations.

## Performance
- Significantly improves upon vanilla linear attention on recall tasks
- Parallelized via chunkwise algorithm for large-scale training
- Approaches softmax attention quality
