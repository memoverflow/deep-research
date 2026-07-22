---
url: https://arxiv.org/abs/2305.17493
title: "The Curse of Recursion: Training on Generated Data Makes Models Forget"
type: arxiv_paper
authors: Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, Ross Anderson
year: 2023 (v3 2024)
accessed: 2026-07-25
quality: 5
relevance: core
---

原创提出"model collapse"概念的论文。核心内容：

## 定义
Model Collapse: 生成模型代际训练中的退化过程，生成数据污染下一代训练集，模型逐渐"误解现实"。分两阶段：
- Early model collapse: 模型开始丢失分布尾部信息
- Late model collapse: 模型混淆不同 mode，收敛到与原分布几乎无关、方差极小的分布

## 两种误差来源
1. **Statistical approximation error（主因）**：有限样本采样导致的误差，样本数趋于无穷时消失。每次重采样都有非零概率丢失信息。例：10^7 个样本估计标准正态分布均值，仍有 ~1.9e-4 的偏差。
2. **Functional approximation error（次因）**：函数近似器表达力不足或过强导致的误差，例如用单个高斯拟合两个高斯的混合。若无统计误差，此误差只在第一代出现。

## 数学模型（Learning with Generational Data）
第 i 代数据集 D_i 由分布 p_i 生成，用函数近似 F_θ: p_i → p_θ(i+1)，
下一代采样分布 p_{i+1} = α_i p_θ(i+1) + β_i p_i + γ_i p_0

单维高斯情形下推导出：随代数增加，方差呈随机游走式增长（Var → 0 或发散），均值随机漂移，最终分布退化。

## LLM 实验：OPT-125m 在 wikitext2 上递归微调
经典"jackrabbit"退化案例：
- Input: 关于中世纪建筑历史的一段文本
- Gen 0: 输出还算连贯（讲 Perpendicular Revival architecture）
- Gen 7: 开始跑题（变成关于一个人接受采访）
- Gen 9: 完全退化为无意义重复列表："...home to some of the world's largest populations of black-tailed jackrabbits, white-tailed jackrabbits, blue-tailed jackrabbits, red-tailed jackrabbits, yellow-..."

这个例子直观展示了 late model collapse：模型收敛到极低方差、重复模式的输出。

## Key takeaway
模型坍缩在 VAE、GMM、LLM 中都被观察到，是所有递归训练在自身生成数据上的生成模型的普遍现象（"universal among generative models"）。
