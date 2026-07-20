---
url: https://arxiv.org/pdf/2005.14165
title: "Language Models are Few-Shot Learners (GPT-3)"
type: arxiv_paper
authors: Tom B. Brown et al., OpenAI
year: 2020
accessed: 2026-07-23
quality: 5
relevance: supporting
---

## 相关内容

GPT-3 技术报告 Table 2.1 列出了各规模模型的批量大小（以 token 计）与学习率超参数。论文正文明确写道："larger models can typically use a larger batch size, but require a smaller learning rate. We measure the gradient noise scale during training and use it to guide our choice of batch size [MKAT18]." （MKAT18 即指 McCandlish, Kaplan, Amodei, Team 2018 论文）

这证实了梯度噪声尺度方法并非纯理论玩具，而是被 OpenAI 在训练 GPT-3（1750亿参数）这样的真实大规模系统时实际采用的工程方法，是"理论直接指导工业级训练超参数选择"的少数公开案例之一。

## Also referenced

- Goyal et al. 2017 "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (arxiv.org/abs/1706.02677) — 提出线性学习率缩放法则 Linear Scaling Rule：批量放大 k 倍，学习率也放大 k 倍，其他超参数不变。ResNet-50 用 batch=8192 达到与 batch=256 相近精度，1小时内完成 ImageNet 训练。
- Malladi et al. 2022 "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms" (arxiv.org/abs/2205.10287) — 用随机微分方程理论证明 Adam/RMSProp 应采用平方根缩放法则而非线性缩放法则。
