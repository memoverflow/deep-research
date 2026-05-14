---
url: https://arxiv.org/abs/2401.00448
title: "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws"
type: arxiv_paper
year: 2024
quality: 4
relevance: high
---

# Inference-Aware Scaling

## Key Insight
Chinchilla optimality minimizes TRAINING compute, but real deployment must account for inference.

## When Inference Dominates
- High deployment demand → optimal strategy shifts to smaller models trained longer
- "Overtraining" justified by inference efficiency

## LLaMA Justification
- LLaMA 7B trained on 1T tokens (far exceeding 20:1 Chinchilla ratio)
- Much cheaper to deploy than Chinchilla-optimal 30B at equivalent loss
- Total cost (training + inference) minimized by overtraining

## Formalization
Total_cost = C_train + N_queries × C_inference
Optimal model size decreases as N_queries increases.
