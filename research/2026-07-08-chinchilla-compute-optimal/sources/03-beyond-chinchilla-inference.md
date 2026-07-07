---
url: https://arxiv.org/abs/2401.00448
title: "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws"
type: arxiv_paper
authors: Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle (MosaicML/Databricks)
year: 2024 (ICML 2024)
accessed: 2026-07-08
quality: 5
relevance: core
---

## Abstract
Large language model (LLM) scaling laws are empirical formulas that estimate changes in model quality as a result of increasing parameter count and training data. However, these formulas, including the popular DeepMind Chinchilla scaling laws, neglect to include the cost of inference. We modify the Chinchilla scaling laws to calculate the optimal LLM parameter count and pre-training data size to train AND deploy a model of a given quality and inference demand. We conduct our analysis both in terms of compute budget and real-world dollar costs, finding that LLM researchers expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal. We train 47 models of varying sizes/token counts to validate our formula, finding that model quality continues improving as tokens-per-parameter is scaled to extreme ranges (up to 10,000!). We also ablate the coefficient-fitting procedure, finding that fitting scaling laws only from data at typical token/parameter ratios overestimates the impact of additional tokens at extreme ranges.

## Key content
- Chinchilla only minimizes *training* compute. It ignores that a deployed model gets queried potentially billions of times — inference compute dominates lifetime cost for popular models.
- Because inference cost scales with model size (not tokens seen during training), if you expect huge deployment volume you should deliberately train a SMALLER model for LONGER (more tokens per parameter) than Chinchilla prescribes — "overtraining." The extra training FLOPs pay for themselves via cheaper-per-token inference over the model's lifetime.
- This directly explains why LLaMA (Meta) and other production models trained way past the Chinchilla ratio: LLaMA 2: 2T tokens; LLaMA 3: 15T tokens — vastly more tokens per parameter (LLaMA-3-8B alone implies ~1875 tokens/param, far beyond the ~20:1 Chinchilla ratio).
- Quote: "the extra training compute required to train the 7B model beyond its Chinchilla-optimal point to match the 13B's quality is made up for during inference."
- Found model quality keeps improving even at token/parameter ratios up to 10,000:1 — no wall, diminishing returns but still positive.
- Practical conclusion: Chinchilla answers "how to train the best model per training FLOP" — NOT "how to train the best model per total dollar (train+serve) I'll ever spend." Different optimization target → different answer.
