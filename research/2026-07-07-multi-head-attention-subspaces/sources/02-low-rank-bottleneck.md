---
url: https://arxiv.org/abs/2002.07028
title: "Low-Rank Bottleneck in Multi-head Attention Models"
type: arxiv_paper
authors: Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank J. Reddi, Sanjiv Kumar
year: 2020
accessed: 2026-07-07
quality: 5
relevance: core
---

核心论点：标准多头注意力中，把 d_model 平分给 h 个头，每个头的维度 d_k = d_model/h。当头数增多时，
每个头的维度就变小。作者证明：单个 head 计算出的 attention 权重矩阵（softmax(QK^T)，大小为
sequence_length × sequence_length）的秩不能超过 d_k。当 d_k 比序列长度小很多时，这个 attention 矩阵
存在"低秩瓶颈" —— 无法表达某些需要高秩的 attention 模式，限制了模型的表达能力。

解决方案：将每个头的维度设为与输入序列长度（而非头数）无关，即不再要求 d_k = d_model/h，而是让 d_k
独立设置为足够大的值（如接近序列长度），从而让 attention 矩阵可以达到更高的秩。实验证明这样可以用更小
的 embedding 维度训练出性能更好的模型。

意义：这篇论文揭示了"头数 vs 每头维度"不是随意的工程选择，而是存在数学上的表达力权衡 —— 头数越多、
每头维度越小，attention 矩阵的秩上限越低，可能丢失某些需要"精细/高秩关注模式"的能力。
