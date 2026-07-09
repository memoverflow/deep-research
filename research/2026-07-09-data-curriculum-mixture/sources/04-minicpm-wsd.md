---
url: https://arxiv.org/abs/2404.06395
title: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies"
type: arxiv_paper
authors: Shengding Hu, et al. (OpenBMB / Tsinghua NLP)
year: 2024
accessed: 2026-07-09
quality: 5
relevance: core
---

## Abstract

The burgeoning interest in developing Large Language Models (LLMs) with up to trillion parameters has been met with concerns regarding resource efficiency and practical expense, particularly given the immense cost of experimentation. This scenario underscores the importance of exploring the potential of Small Language Models (SLMs) as a resource-efficient alternative. In this context, we introduce MiniCPM, specifically the 1.2B and 2.4B non-embedding parameter variants, not only excel in their respective categories but also demonstrate capabilities on par with 7B-13B LLMs. While focusing on SLMs, our approach exhibits scalability in both model and data dimensions for future LLM research. Regarding model scaling, we employ extensive model wind tunnel experiments for stable and optimal scaling. For data scaling, we introduce a Warmup-Stable-Decay (WSD) learning rate scheduler (LRS), conducive to continuous training and domain adaptation. We present an in-depth analysis of the intriguing training dynamics that occurred in the WSD LRS. With WSD LRS, we are now able to efficiently study data-model scaling law without extensive retraining experiments on both axes of model and data, from which we derive the much higher compute optimal data-model ratio than Chinchilla Optimal.

## Key Points
- WSD (Warmup-Stable-Decay) 调度器把学习率分成三段：warmup（升）→ stable（恒定，主训练阶段）→ decay（快速降到接近0）。
- 关键工程洞察：**decay 阶段正是切入高质量数据的最佳窗口**。在 stable 阶段用海量、质量参差的数据训练主干能力；当学习率开始快速衰减时，把训练数据换成高质量/领域相关的数据，可以让模型在训练末期"精修"，效果远好于从头到尾都用同一批数据。
- 因为 stable 阶段的学习率不依赖预先设定好的总步数，WSD 允许在任意时刻"分支"出一个衰减分支来产出一个可用的 checkpoint，不需要提前锁定训练预算——这让"训练时长与数据课程解耦"，可以灵活地做多阶段数据课程实验。
- 通过 WSD，团队可以低成本地研究 data-model scaling law（不需要每次都重新训练），发现了比 Chinchilla Optimal 更高的"计算最优数据/模型比例"（即在同样计算量下，应该给更多的数据、相对更小的模型）。
