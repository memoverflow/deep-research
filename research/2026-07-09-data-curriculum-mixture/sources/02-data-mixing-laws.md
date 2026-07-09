---
url: https://arxiv.org/abs/2403.16952
title: "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance"
type: arxiv_paper
authors: Jiasheng Ye, Peiju Liu, Tianxiang Sun, et al. (Fudan/Shanghai AI Lab)
year: 2024
accessed: 2026-07-09
quality: 5
relevance: core
---

## Abstract

Pretraining data of large language models composes multiple domains (e.g., web texts, academic papers, codes), whose mixture proportions crucially impact the competence of outcome models. While existing endeavors rely on heuristics or qualitative strategies to tune the proportions, we discover the quantitative predictability of model performance regarding the mixture proportions in function forms, which we refer to as the data mixing laws. Fitting such functions on sample mixtures unveils model performance on unseen mixtures before actual runs, thus guiding the selection of an ideal data mixture. Furthermore, we propose nested use of the scaling laws of training steps, model sizes, and our data mixing law to enable predicting the performance of large models trained on massive data under various mixtures with only small-scale training. Moreover, experimental results verify that our method effectively optimizes the training mixture of a 1B model trained for 100B tokens in RedPajama, reaching a performance comparable to the one trained for 48% more steps on the default mixture. Extending the application of data mixing laws to continual training accurately predicts the critical mixture proportion that avoids catastrophic forgetting and outlooks the potential for dynamic data schedules.

## Key Points
- 核心发现："配比 → 性能"的关系可以用一个函数形式拟合（类似 scaling law 的形式，但自变量是混合比例而不是模型规模/token 数）。
- 三层嵌套 scaling laws: 训练步数的 scaling law × 模型规模的 scaling law × 数据配比的 mixing law，组合起来可以用小规模实验（小模型、少数据、少数配比组合）预测大规模训练在任意配比下的表现，而不需要真的跑大规模实验。
- 实际验证：1B 模型 100B token 在 RedPajama 上，用预测出的最优配比训练，效果等价于用默认配比多训练 48% 的步数——即"配对了，等于省了将近一半算力"。
- 应用到持续训练（continual training）：可以预测在混入新领域数据时，需要保留多少比例的旧数据才能避免"灾难性遗忘"（模型在旧任务上性能骤降），为动态数据调度提供理论支撑。
