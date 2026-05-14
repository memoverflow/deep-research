---
url: https://arxiv.org/abs/2407.04620
title: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
type: arxiv_paper
authors: Yu Sun et al.
year: 2024
quality: 5
relevance: core
---

# Test-Time Training (TTT) Layers

## Key Innovation
Hidden state IS a model that learns via gradient descent during both training and inference.

## Formulation
W_t = W_{t-1} - η ∇ℓ(W_{t-1}; x_t)

Where ℓ is self-supervised reconstruction loss.

## Two Variants
- **TTT-Linear**: W is linear projection → equivalent to linear attention with online learning
- **TTT-MLP**: W is two-layer MLP → greater expressiveness

## Key Insight
Hidden state expressiveness grows with model capacity (unlike fixed-size RNN states).

## Performance
- TTT-Linear matches Mamba at 1.3B
- TTT-MLP outperforms Mamba on long-context tasks (>8K tokens)
- Advantage grows with context length

## Limitation
Gradient computation adds overhead vs simple recurrences.
