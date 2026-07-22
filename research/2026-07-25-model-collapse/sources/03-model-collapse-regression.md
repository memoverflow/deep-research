---
url: https://arxiv.org/html/2402.07712v1
title: "Model Collapse Demystified: The Case of Regression"
type: arxiv_paper
authors: Elvis Dohmatob, Yunzhen Feng, Julia Kempe (NYU) — NeurIPS 2024
year: 2024
accessed: 2026-07-25
quality: 5
relevance: core
---

## 核心贡献
在核回归（kernel regression）的简化设定下给出 model collapse 的理论刻画，得到修正后的 scaling law，揭示"从可以应对合成数据"到"性能完全崩溃"之间存在明确的 **crossover（相变点）**。

## Scaling Law 修正
在多项式衰减谱（polynomial decaying spectral）和 source condition 假设下：
- 正常（无合成数据污染）情况下，test error 随训练数据量 n 呈幂律衰减：error ∝ n^(-β)
- 引入合成数据后，衰减速率发生 crossover：从 fast rate 变为 slow rate，即模型不再能像正常情况那样随着更多"数据"变好——因为这些新增数据里混入的合成部分实际上在拉低信息含量
- 存在一个由合成数据比例决定的相变阈值：低于阈值时代际训练仍可控，超过阈值后误差迅速失控

## 缓解策略：Adaptive Regularization
论文提出一种基于自适应正则化的简单策略来缓解 model collapse——本质上是让模型对"看起来像自己生成的"数据施加更强的正则化约束，防止模型过度自信地强化自己的偏差。

## 意义
这篇论文把 model collapse 从"经验现象"提升为"可以用统计学习理论精确刻画的相变"，为后续研究（如判断"多少合成数据比例是安全的"）提供了数学工具。
