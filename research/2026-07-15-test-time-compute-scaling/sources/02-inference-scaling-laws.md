---
url: https://arxiv.org/abs/2408.00724
title: "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving"
type: arxiv_paper
year: 2024
accessed: 2026-07-15
quality: 5
relevance: core
---

Abstract: While training scaling laws are extensively studied, optimal inference configurations remain underexplored. Studies inference scaling laws / compute-optimal inference: tradeoffs between model sizes and generating additional tokens with different inference strategies (greedy search, majority voting, best-of-n, weighted voting, tree search). Findings: scaling inference compute with inference strategies can be more computationally efficient than scaling model parameters. Smaller models + advanced inference algorithms offer Pareto-optimal cost/performance tradeoffs — e.g. Llemma-7B + novel tree search consistently outperforms Llemma-34B on MATH benchmark.

Key takeaway: independent confirmation from a different group that "smaller model + smart inference strategy" can Pareto-dominate "bigger model, naive decoding".
