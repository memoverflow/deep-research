---
url: https://arxiv.org/abs/2305.16264
title: "Scaling Data-Constrained Language Models"
type: arxiv_paper
authors: Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al. (Hugging Face et al.)
year: 2023 (NeurIPS 2023)
accessed: 2026-07-08
quality: 5
relevance: supporting
---

## Abstract
The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. We run experiments varying the extent of data repetition and compute budget, up to 900B training tokens and 9B parameter models (400 total training runs). We find that with constrained data, training with up to 4 epochs of repeated data yields negligible changes to loss compared to unique data. But with more repetition, the value of adding compute eventually decays to zero. We propose and validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters.

## Key content
- Extends Chinchilla's premise (assumes effectively infinite unique data) to the real-world case where you might run out of fresh text and have to repeat data.
- Repeating data isn't free: each additional epoch on the same tokens is worth less than a fresh token, and this "decay" needs to be built into the N_opt/D_opt formula.
- Relevant to the article as a "yes but" footnote: Chinchilla's clean 20:1 ratio assumes unlimited unique data; in practice (and increasingly, as the internet's text supply gets exhausted for frontier labs) this assumption breaks and the optimal allocation shifts again.
