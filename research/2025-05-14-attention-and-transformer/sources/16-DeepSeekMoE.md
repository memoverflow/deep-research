---
url: https://arxiv.org/abs/2401.06066
title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
type: arxiv_paper
authors: Damai Dai et al.
year: 2024
quality: 5
relevance: core
---

# DeepSeekMoE

## Two Key Strategies

### 1. Fine-Grained Expert Segmentation
- Instead of N experts each size d → use mN experts each size d/m
- Activate mK of them (instead of K)
- Example: 16 experts choose 2 → 64 experts choose 8
- More flexible combinations, better specialization

### 2. Shared Expert Isolation
- Dedicate Ks experts as permanently active shared experts
- Capture common knowledge preventing redundancy in routed experts
- Remaining experts specialize in diverse knowledge

## Results
- DeepSeekMoE 16B: 16.4B total, 2.8B active parameters
- Approaches LLaMA2 7B performance (dense 6.7B)
- ~2x parameter efficiency over conventional MoE
