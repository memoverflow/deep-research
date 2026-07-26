---
url: https://arxiv.org/abs/2310.06825
title: "Mistral 7B"
type: arxiv_paper
authors: Albert Q. Jiang, et al. (Mistral AI)
year: 2023
accessed: 2026-07-28
quality: 4
relevance: background
---

## 关键内容
Mistral 7B 结合 Grouped-Query Attention (GQA) 和 Sliding Window Attention (SWA) 来平衡效率与长上下文能力，是 SWA 在生产级开源模型中的代表性应用。SWA 让每层只关注固定窗口内的最近 token，通过堆叠多层，有效接收域随层数线性增长（类似 CNN 的感受野堆叠原理）——这是"静态稀疏"路线在工业界落地的典型案例，同时也说明了这类方法为何仍然是稀疏注意力谱系里重要的一环（简单、成熟、易实现），但表达力不如后续动态门控方法。

## 用途
补充说明静态稀疏路线（SWA）在实际产品中的应用，作为 NSA/MoBA 之前技术演进的一环。
