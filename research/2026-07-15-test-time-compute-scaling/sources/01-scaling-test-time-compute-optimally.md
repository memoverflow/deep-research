---
url: https://arxiv.org/abs/2408.03314
title: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"
type: arxiv_paper
authors: Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar (Google DeepMind)
year: 2024
accessed: 2026-07-15
quality: 5
relevance: core
---

Abstract: Enabling LLMs to improve their outputs by using more test-time computation is a critical step towards building generally self-improving agents. We study the scaling of inference-time computation in LLMs, focusing on: if an LLM is allowed to use a fixed but non-trivial amount of inference-time compute, how much can it improve its performance on a challenging prompt? We analyze two primary mechanisms to scale test-time computation: (1) searching against dense, process-based verifier reward models (PRMs); and (2) updating the model's distribution over a response adaptively given the prompt at test time (revision models). Effectiveness of different approaches critically varies depending on the difficulty of the prompt. This motivates a "compute-optimal" scaling strategy that adaptively allocates test-time compute per prompt. Using this strategy, efficiency of test-time compute scaling improves by more than 4x compared to a best-of-N baseline. In a FLOPs-matched evaluation, test-time compute can be used to outperform a 14x larger model on problems where a smaller base model attains non-trivial success rates.

Key takeaways for article:
- Two scaling mechanisms: search w/ PRM verifiers, and revision (iterative self-correction)
- Difficulty-adaptive allocation is the key insight — "compute-optimal" beats uniform Best-of-N by 4x
- FLOPs-matched: small model + test-time compute > 14x larger model (on appropriate problems)
