---
url: https://arxiv.org/abs/2310.02743
title: "Reward Model Ensembles Help Mitigate Overoptimization"
type: arxiv_paper
authors: Thomas Coste, Usman Anwar, Robert Kirk, David Krueger
year: 2023
accessed: 2026-07-10
quality: 5
relevance: core
---

Abstract: RLHF is a standard approach for fine-tuning LLMs to follow instructions.
Learned reward models used to approximate human preferences are, as imperfect
representations of the "true" reward, susceptible to overoptimization. Building
on Gao et al. (2023)'s synthetic gold-reward-model setup, this paper studies
whether ensemble-based conservative optimization can counter overoptimization.
Two objectives tested: worst-case optimization (WCO, take the minimum score
across the ensemble) and uncertainty-weighted optimization (UWO, penalize by
ensemble variance), on two optimization methods: best-of-n (BoN) and PPO.

Key findings:
- With and without added 25% label noise (to mirror realistic noisy human
  feedback), conservative optimization practically eliminates overoptimization
  and improves final performance by up to 70% for BoN sampling.
- For PPO, ensemble-based conservative optimization always reduces
  overoptimization and beats single-reward-model optimization.
- Combining ensemble conservative optimization with a SMALL KL penalty
  successfully prevents overoptimization at essentially no performance cost —
  contrasting with Gao et al.'s finding that KL penalty alone (without ensembling)
  wasn't a clean fix.
- Practical takeaway: the fix for reward hacking isn't "add more regularization"
  in isolation, it's "quantify how much you can trust the reward signal in a
  given state" and discount optimization pressure accordingly — closer in spirit
  to ensemble uncertainty methods from Bayesian RL / offline RL pessimism.
