---
url: https://arxiv.org/abs/2402.17762
title: "Massive Activations in Large Language Models"
type: arxiv_paper
authors: Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu
year: 2024
accessed: 2026-07-14
quality: 5
relevance: supporting
---

发现并命名"massive activations"现象，是理解 attention sink 底层机制的另一半拼图。

核心发现：
- 在 LLM 的隐藏状态中，极少数激活值会比其他激活值大得多——论文给出的例子是高达 100,000 倍。
- 这些"巨量激活"广泛存在于各种 LLM 中，作者定位了它们出现的具体层/维度。
- 关键性质：这些激活值的大小基本不随输入变化——即无论输入什么句子，这个位置/维度的激活值都差不多大。这意味着它们的功能更像是**模型内部隐式的偏置项（bias term）**，而不是在编码某个具体输入的信息。
- 因果链条：massive activations → 导致注意力概率集中到对应的 token 上（也就是 attention sink 现象的成因之一）→ 进一步在 self-attention 输出中产生隐式的偏置项。
- 论文同时在 Vision Transformer 中发现了类似的 massive activations，说明这不是语言模型独有的现象，而是注意力架构本身的一般性质。

意义：这篇论文提供了 attention sink 的"物理层"解释——sink token 之所以被选中，很可能是因为该位置的隐藏状态里出现了几个数值极大、几乎和输入无关的激活分量，这些分量在 QK 点积计算中天然会产生很大的注意力得分，从而把注意力"吸"过去。这与 Barbero et al. 的"信息论"解释（避免 over-mixing）是互补的两个视角：一个讲机制如何实现（massive activations），一个讲为什么这样做有用（避免过度混合）。
