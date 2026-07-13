---
url: https://arxiv.org/abs/2405.18781
title: "On the Role of Attention Masks and LayerNorm in Transformers"
type: arxiv_paper
authors: Xinyi Wu et al.
year: 2024
venue: NeurIPS 2024
accessed: 2026-07-13
quality: 5
relevance: core
---

## Key content
Extends Dong et al.'s rank collapse analysis by considering attention masks (e.g. causal masks) and LayerNorm together, which prior literature (Dong et al. 2021) had mostly overlooked in combination.

Key finding: attention masks and LayerNorm, considered jointly (not in isolation as Dong et al. did for LayerNorm alone), can counteract token homogeneity and boost expressivity — contrary to the flat "LayerNorm plays no role" conclusion when analyzed without masks. With a causal mask, LayerNorm's role becomes more nuanced: it can help avoid collapse to a single point in some regimes, though the mechanism differs from naive "it re-spreads distances" intuition.

Uses Perron-Frobenius theorem and ergodicity arguments: since attention matrices are strictly positive right-stochastic matrices (softmax guarantees positivity + row-sum-1), their products are analyzed via Perron-Frobenius / ergodic theory to characterize convergence behavior.

## Notes for article
Used to nuance the "LayerNorm does nothing" claim — shows later work complicates this when causal masks are considered jointly.
