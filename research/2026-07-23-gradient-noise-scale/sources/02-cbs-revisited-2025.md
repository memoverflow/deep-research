---
url: https://arxiv.org/abs/2505.23971
title: "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training"
type: arxiv_paper
authors: William Merrill, Shane Arora, Dirk Groeneveld, Hannaneh Hajishirzi (Allen Institute for AI)
year: 2025
accessed: 2026-07-23
quality: 5
relevance: core
---

## 核心内容摘要

对 McCandlish 2018 的梯度噪声尺度方法提出理论质疑，并提出一种更直接的"局部分支训练"(local branched training) 方法来测量临界批量大小 (CBS)，应用于 OLMo 1B/7B 模型的预训练。

## 对原方法的批评

1. **假设1：SGD 优化器** — McCandlish 的推导假设 SGD，但实践中 LLM 训练用 Adam。Malladi et al. (SDE 理论分析) 指出：Adam 应该用**平方根缩放法则** (learning rate ∝ √batch_size)，而非线性缩放法则
2. **假设2：良态优化 (Hessian ≈ 单位矩阵倍数)** — 这是把 B_noise 简化为 B_simple 的关键假设，但在实践中很少成立，导致无法确定 B_simple 到 B_crit 的准确换算系数

## 新方法：局部分支训练 (Local Branched Training)

从某个训练 checkpoint 出发，用不同的批量大小 k·B（和相应缩放的学习率 f(k)·η）训练一小段 token 预算 Δ，比较各分支的 loss 恢复情况，找出不再造成 loss 明显退化的最大批量。

## 关键实证发现

1. **CBS 在训练初期接近 0，然后迅速上升，最终趋于平台**：训练刚开始时任何批量都会造成明显退化；训练中期 CBS 快速增长；后期趋于稳定
2. **CBS 几乎不依赖模型规模**：1B 与 7B 模型的 CBS 曲线走势一致，意味着小模型的 CBS 测量结果可以指导大模型训练的批量大小选择
3. **批量大小热身 (Batch Size Warmup)**：从小批量开始，随着训练进行、CBS 增长而逐步倍增批量，用此策略训练 OLMo 1B，用少 43% 的梯度步数达到（略优于）原始 loss
4. **GPT-3 技术报告确实采用了 McCandlish 的梯度噪声尺度方法**来指导批量大小选择（"we measure the gradient noise scale during training and use it to guide our choice of batch size [MKAT18]"）——但作者认为这一实践缺乏严格的理论支撑

## 结论意义

这篇论文没有否定"存在临界批量大小"这一核心直觉，而是指出：用梯度噪声尺度直接估算 CBS 在 Adam 优化器下缺乏理论保证，更可靠的做法是直接测量（分支训练）而非用代理统计量估计。这体现了理论模型（2018）与工程实践验证（2025）之间七年后的迭代和纠偏，是很好的"理论-实践"张力案例。
