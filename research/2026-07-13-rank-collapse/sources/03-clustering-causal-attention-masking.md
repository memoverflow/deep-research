---
url: https://arxiv.org/html/2411.04990v2
title: "Clustering in Causal Attention Masking"
type: arxiv_paper
authors: Nikita Karagodin, Yury Polyanskiy (MIT)
year: 2024
accessed: 2026-07-13
quality: 5
relevance: core
---

## Key content
Studies rank-collapse-like dynamics specifically under causal attention masking (the autoregressive/decoder-only setting used by GPT-style models), as opposed to Dong et al.'s bidirectional/BERT-style analysis.

Theorem 4.1: with V = Identity and Q,K arbitrary, for almost any starting configuration on the sphere, the causal transformer dynamics converge — but instead of collapsing to a single point (rank-1) as in the unmasked case, tokens under causal masking can converge to TWO separate clusters (not one), depending on λ_max sign conditions. This is a softer form of degeneracy than full rank-1 collapse: tokens still lose most of their individual distinctiveness but partition into a small number of clusters rather than a single point.

## Notes for article
Used to explain why decoder-only/GPT-style causal models may be somewhat more resistant to full collapse than bidirectional models — the causal mask itself restricts how much "averaging" each token undergoes (earlier tokens see fewer other tokens to average with).
