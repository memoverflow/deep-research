---
url: https://arxiv.org/abs/1706.03762
title: "Attention Is All You Need"
type: arxiv_paper
authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
year: 2017
accessed: 2026-07-07
quality: 5
relevance: core
---

Multi-head attention 原始定义论文。关键论述："Multi-head attention allows the model to jointly attend to
information from different representation subspaces at different positions. With a single attention head,
averaging inhibits this." 论文用 h=8 个头，每个头 d_k=d_v=d_model/h=64，保持总计算量与单头 d_model 维度的
attention 大致相同（"we vary the number of attention heads and the attention key and value dimensions...
while keeping the amount of computation constant"）。

Table 3 (A) 行显示：单头 attention 比最优配置差 0.9 BLEU；但头数过多（如 h=16, h=32）质量同样下降，说明存在
最优头数区间，不是头越多越好。
