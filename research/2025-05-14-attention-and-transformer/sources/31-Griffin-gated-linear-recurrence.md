---
url: https://arxiv.org/abs/2402.19427
title: "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models"
type: arxiv_paper
authors: Soham De, Samuel Smith et al. (Google DeepMind)
year: 2024
quality: 5
relevance: core
---

# Griffin (RecurrentGemma)

## Architecture
Mixes Real-Gated Linear Recurrent Unit (RG-LRU) with local sliding-window attention.

## RG-LRU
h_t = α_t ⊙ h_{t-1} + √(1-α_t²) ⊙ (B x_t)
Where α_t = σ(a_t), a_t is input-dependent

## Hybrid Design
- Interleaves recurrent blocks with local MQA attention
- Ratio: 2 recurrent : 1 local attention
- Local window size: 128-1024

## Key Properties
- Recurrence for global context
- Local attention for precise short-range
- Strong length extrapolation beyond training context

## Productionized as RecurrentGemma
Practical Transformer replacement by Google.
