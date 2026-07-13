---
url: https://arxiv.org/pdf/2203.00555
title: "DeepNet: Scaling Transformers to 1,000 Layers"
type: arxiv_paper
authors: Microsoft Research
year: 2022
accessed: 2026-07-13
quality: 5
relevance: supporting
---

## Key content
Introduces DeepNorm, a normalization + residual scaling scheme (replacing Post-LN) that stabilizes training of extremely deep Transformers — successfully trained models up to 1000 layers. A 200-layer 3.2B parameter model outperformed a 48-layer 12B parameter model by 5 BLEU points on multilingual NMT benchmark, showing depth (properly stabilized) can be more parameter-efficient than width.

DeepNorm works by scaling the residual connection by a constant factor before adding it back, tuned to keep gradient/activation scale bounded across very deep stacks — engineering response to the same underlying tension: attention layers want to homogenize representations, and depth amplifies this, so residual "strength" needs careful tuning as depth grows.

## Notes for article
Used as the "engineering answer" to the rank collapse theory — shows how tuning residual connection strength enables scaling to unprecedented depths, echoing the theoretical finding that skip connections are the primary defense against collapse.
