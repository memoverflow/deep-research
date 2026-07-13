---
url: https://arxiv.org/abs/2411.19477
title: "A Simple and Provable Scaling Law for the Test-Time Compute of Large Language Models"
type: arxiv_paper
year: 2024
accessed: 2026-07-15
quality: 4
relevance: supporting
---

Abstract: Two algorithms with provable scaling laws for test-time compute: (1) knockout-style tournament — generate candidates, aggregate via knockout tournament; failure probability decays exponentially/power-law with test-time compute. (2) league-style — each candidate evaluated by average win rate vs multiple opponents; failure probability also decays exponentially. Both require only a black-box LLM, no external verifier/reward model.

Key takeaway: theoretical grounding that repeated sampling + pairwise comparison (even without external verifiers) can provably reduce failure probability as compute scales — a minimalist alternative to PRM-based search.
