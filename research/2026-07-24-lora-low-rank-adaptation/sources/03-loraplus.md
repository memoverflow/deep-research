---
url: https://arxiv.org/abs/2402.12354
title: "LoRA+: Efficient Low Rank Adaptation of Large Models"
type: arxiv_paper
authors: Soufiane Hayou, Nikhil Ghosh, Bin Yu
year: 2024
accessed: 2026-07-24
quality: 5
relevance: important extension
---

## Abstract
Shows that LoRA (Hu et al. 2021) leads to suboptimal finetuning of models with large width (embedding dimension), because adapter matrices A and B are updated with the SAME learning rate. Using scaling arguments for large width networks, demonstrate that using the same LR for A and B doesn't allow efficient feature learning. Correcting this by setting different learning rates for A and B with a well-chosen fixed ratio → LoRA+. In experiments: 1-2% performance improvement, up to ~2x speedup, at same computational cost as LoRA.

## Key Content
- Root cause: In ΔW = BA, B starts at zero and A starts random. Because of how gradients flow through this bilinear product, in the infinite-width limit, the gradient with respect to B is much smaller in scale than gradient with respect to A (or vice versa depending on convention) — meaning with one shared learning rate, one matrix effectively "learns" much slower than the theoretically optimal rate.
- Fix: set learning rate ratio λ = η_B / η_A >> 1 (asymmetric learning rates). This is grounded in "feature learning" scaling theory (related to μP - maximal update parameterization) — ensures that as model width grows, both matrices contribute meaningfully to feature learning instead of one dominating/vanishing.
- Practical takeaway for practitioners: naive LoRA with equal LR for A,B is provably suboptimal for large models; a simple fix (different LR for the two matrices) recovers most of the lost performance with no extra compute cost.
