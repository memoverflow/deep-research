---
url: https://arxiv.org/abs/2402.03300
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning (introduces GRPO)"
type: arxiv_paper
authors: Zhihong Shao et al.
year: 2024
quality: 5
relevance: core
---

# GRPO: Group Relative Policy Optimization

## Key Innovation
Eliminates critic/value model from PPO.

## Algorithm
1. For each prompt, sample group of K outputs
2. Compute rewards for all K outputs
3. Normalize within group as baseline:
   advantage_i = (r_i - mean(r_group)) / std(r_group)
4. Use clipped surrogate objective (like PPO) but without value function

## Advantages
- No critic model → less memory
- Group statistics provide natural baseline
- Better for reasoning tasks (math, coding)
- Used in DeepSeek-R1 for reasoning without supervised data

## Relationship to PPO
- Same clipped objective
- Same KL penalty from reference
- But: group reward normalization replaces learned value function
