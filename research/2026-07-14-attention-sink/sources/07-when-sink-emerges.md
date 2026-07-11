---
url: https://arxiv.org/abs/2410.10781
title: "When Attention Sink Emerges in Language Models: An Empirical View"
type: arxiv_paper
authors: (ICLR 2025, sail-sg)
year: 2024
accessed: 2026-07-14
quality: 4
relevance: supporting
---

研究 attention sink 在预训练过程中"何时"以及"因何"出现的实证论文。

核心发现：
- Attention sink 并非从训练一开始就存在，而是在训练进行到一定程度后才逐渐出现——实验中观察到大约在 1,000-2,000 个优化步之间开始出现，并随着预训练推进逐渐增强、变得更明显。
- Attention sink 的出现与"有效的优化动态"相关：只有在模型在足够多的数据上得到有效优化后，才会出现该现象；使用更小的学习率会延迟 sink 出现，需要更多步数才能达到相当的 loss。
- 在从头训练的 30B A3B MoE 模型上分析训练轨迹，发现该机制在训练早期就出现，并逐渐集中在最初两层——这可能可以作为追踪预训练健康状况的一个信号。
- 关键结论：如果修改训练数据分布，sink 的位置可以从"第一个 token"转移到其他位置——说明 sink 落在"第一个 token"这件事并非架构强制规定，而是训练数据分布（第一个 token 对所有后续 token 可见）自然导致的次优解。

意义：从时间维度和训练动态角度补充了"sink 是学出来的、是优化过程的产物，不是初始化或架构写死的"这一结论,与 Barbero et al. 的静态分析（LLaMA 3.1 405B 80% 头有强 sink）形成时间线上的呼应。
