---
url: https://arxiv.org/abs/2305.18290
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
type: arxiv_paper
authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, et al.
year: 2023
quality: 5
relevance: core
---

# Direct Preference Optimization (DPO)

## Key Innovation
Eliminates explicit reward model and RL loop entirely.

## Mathematical Formulation
Derives closed-form showing optimal RLHF policy has direct relationship to reward:

L_DPO = -log σ(β[log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)])

Where:
- y_w = preferred response, y_l = dispreferred
- π_θ = policy model, π_ref = reference model
- β = temperature controlling deviation from reference

## Advantages
- No reward model training
- No RL instability (PPO clipping, value function)
- Simple binary cross-entropy loss
- Computationally lightweight
- Widely adopted (Llama 2, Zephyr, etc.)
