---
url: https://arxiv.org/abs/2203.15556
title: "Training Compute-Optimal Large Language Models"
type: arxiv_paper
authors: Jordan Hoffmann, Sebastian Borgeaud, et al. (DeepMind)
year: 2022
quality: 5
relevance: core
---

# Chinchilla Scaling Laws

## Core Finding
Optimal ratio: tokens ≈ 20× parameters

## Mathematical Relationship
- N_optimal ≈ C^0.49 (parameters scale with sqrt of compute)
- D_optimal ≈ C^0.51 (data scales with sqrt of compute)
- Scale BOTH roughly equally (correcting Kaplan's model-size bias)

## Validation
- Chinchilla: 70B params, 1.4T tokens
- Matches Gopher (280B params, 300B tokens) with 4× fewer parameters
- Same compute budget, much cheaper inference

## Implication
Most existing models (GPT-3, Gopher, PaLM) were SIGNIFICANTLY undertrained relative to their size.
