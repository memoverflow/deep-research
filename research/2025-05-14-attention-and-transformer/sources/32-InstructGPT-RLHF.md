---
url: https://arxiv.org/abs/2203.02155
title: "Training language models to follow instructions with human feedback"
type: arxiv_paper
authors: Long Ouyang et al. (OpenAI)
year: 2022
quality: 5
relevance: core
---

# InstructGPT & RLHF Pipeline

## Three-Stage Alignment
1. **SFT**: Supervised fine-tuning on human demonstrations
2. **Reward Model**: Train on human preference comparisons (A vs B)
3. **PPO**: Optimize policy against reward model with KL penalty

## Key Result
1.3B InstructGPT preferred over 175B GPT-3 by human evaluators.
→ RLHF more effective than 100× model size increase for helpfulness.

## PPO Details
- Clipped surrogate objective
- Value function baseline
- KL penalty from SFT model (prevents reward hacking)
- Reward model: 6B parameters

## Legacy
Established the paradigm for ALL subsequent alignment work (ChatGPT, Claude, etc.)
