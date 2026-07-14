---
url: https://arxiv.org/abs/2306.08543
title: "MiniLLM: Knowledge Distillation of Large Language Models"
type: arxiv_paper
authors: Yuxian Gu, Li Dong, Furu Wei, Minlie Huang
year: 2023
accessed: 2026-07-14
quality: 5
relevance: core
---

Abstract: Knowledge Distillation (KD) is a promising technique for reducing the high
computational demand of large language models (LLMs). However, previous KD methods are
primarily applied to white-box classification models or training small models to imitate
black-box model APIs like ChatGPT. How to effectively distill the knowledge of white-box
LLMs into small models is still under-explored, which becomes more important with the
prosperity of open-source LLMs. In this work, we propose a KD approach that distills LLMs
into smaller language models. We first replace the forward Kullback-Leibler divergence
(KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for
KD on generative language models, to prevent the student model from overestimating the
low-probability regions of the teacher distribution. Then, we derive an effective
on-policy optimization approach to learn this objective. The student models are named
MiniLLM. Extensive experiments in the instruction-following setting show that MiniLLM
generates more precise responses with higher overall quality, lower exposure bias, better
calibration, and higher long-text generation performance than the baselines. Our method is
scalable for different model families with 120M to 13B parameters.

Key ideas:
- Forward KL (student fits teacher, mean-seeking under many textbook framings) can push a
  low-capacity student to spread probability mass over regions the teacher considers
  low-probability/nonsensical, causing hallucination-like overestimation.
- Reverse KL (minimize KL(student || teacher)) is framed as more "mode-seeking" for a
  weak student — student prioritizes matching the teacher's high-probability/major modes
  rather than trying to cover its entire (huge, open-ended) output distribution.
- Because reverse KL over a generative distribution can't be computed via a fixed dataset
  (no closed form / requires sampling from the student itself), they derive a policy
  -gradient-style on-policy optimization procedure — connecting KD to RL / inverse RL from
  the teacher's feedback.
- Works from 120M to 13B parameter students.
