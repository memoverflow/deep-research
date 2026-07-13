---
url: https://arxiv.org/abs/2501.12948
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
type: arxiv_paper
authors: DeepSeek-AI
year: 2025
accessed: 2026-07-15
quality: 5
relevance: core
---

Abstract: Reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), without human-labeled reasoning trajectories. The RL framework facilitates emergent development of advanced reasoning patterns: self-reflection, verification, dynamic strategy adaptation. Trained model achieves superior performance on math, coding, STEM vs. conventional SFT on human demonstrations. Emergent reasoning patterns from large models can be distilled to guide smaller models.

Key takeaways:
- R1-Zero: pure RL, no SFT bootstrap, correctness-only + format reward → emergent long CoT, self-verification behaviors
- R1 (full): adds cold-start SFT data before RL to fix readability/language-mixing issues in R1-Zero
- Distillation: reasoning patterns transferable to smaller dense models
