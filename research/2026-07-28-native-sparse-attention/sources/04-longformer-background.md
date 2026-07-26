---
url: https://arxiv.org/abs/2004.05150
title: "Longformer: The Long-Document Transformer"
type: arxiv_paper
authors: Iz Beltagy, Matthew E. Peters, Arman Cohan
year: 2020
accessed: 2026-07-28
quality: 4
relevance: background
---

## 关键内容
Longformer 是稀疏注意力的经典早期工作之一，把标准 self-attention 替换为"局部滑动窗口 + 少量任务相关的全局 token"的组合模式，实现了 O(n) 复杂度。是本文中"静态稀疏结构"路线的代表——注意力形状（谁看谁）是预先设计好的固定规则，不随训练学习变化。

这条路线（连同 BigBird 的 local+global+random 三种模式）后来被 MoBA 论文明确指出局限：这类结构"高度任务特定"(highly task-specific)，可能损害模型的整体泛化能力。MoBA 论文进一步证明：sliding window attention 和 attention sink 都只是 MoBA 的特例（把动态门控网络替换成恒定选择规则）。

## 用途
用于文章中对比"静态规则稀疏" vs "可学习/动态稀疏"两条路线的历史脉络。
