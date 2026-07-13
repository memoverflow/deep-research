---
url: https://arxiv.org/abs/2305.20050
title: "Let's Verify Step by Step"
type: arxiv_paper
authors: Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe (OpenAI)
year: 2023
accessed: 2026-07-15
quality: 5
relevance: core
---

Abstract: Compares outcome supervision (feedback on final result) vs process supervision (feedback for each intermediate step) for training reliable multi-step reasoning models. Process supervision significantly outperforms outcome supervision on the MATH dataset — process-supervised model solves 78% of problems from a representative test subset. Active learning further improves process supervision efficacy. Released PRM800K: 800,000 step-level human feedback labels, the dataset used to train the best reward model.

Key takeaway: this is the canonical PRM (Process Reward Model) paper. PRM800K became the standard dataset referenced by nearly all later process-reward / test-time search work (used in the "search + PRM" scaling mechanism described in Snell et al. 2024).
