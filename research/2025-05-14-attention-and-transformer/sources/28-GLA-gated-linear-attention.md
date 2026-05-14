---
url: https://arxiv.org/abs/2312.06635
title: "Gated Linear Attention Transformers with Hardware-Efficient Training"
type: arxiv_paper
authors: Songlin Yang et al.
year: 2024
quality: 4
relevance: high
---

# Gated Linear Attention (GLA)

## Key Innovation
Data-dependent gating for linear attention with hardware-efficient chunkwise training.

## Formulation
S_t = G_t ⊙ S_{t-1} + k_t v_t^T
o_t = q_t^T S_t

Where G_t is data-dependent gating matrix.

## Chunkwise Training
- Divide sequence into chunks
- Intra-chunk: parallel attention-like computation
- Inter-chunk: recurrent state passing
- Efficient GPU utilization via FlashLinearAttention

## Unifying Framework
GLA subsumes:
- Mamba (with specific gating structure)
- RWKV (with specific decay structure)
- Linear attention (G=I)

## Performance
Competitive with Mamba and RetNet while being conceptually simpler.
