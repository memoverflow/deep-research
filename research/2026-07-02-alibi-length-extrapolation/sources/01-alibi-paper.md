---
url: https://arxiv.org/abs/2108.12409
title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
type: arxiv_paper
authors: Ofir Press, Noah A. Smith, Mike Lewis
year: 2021
accessed: 2026-07-02
quality: 5
relevance: core
---

Abstract: 提出 ALiBi (Attention with Linear Biases)，一种不使用位置嵌入的位置方法。核心做法：在 query-key 点积之后，直接加一个与距离成正比的线性惩罚项。

关键结果：
- 1.3B 参数模型，用长度 1024 训练，在 2048 长度上测试，达到与 sinusoidal 模型（训练长度就是 2048）相同的困惑度，同时训练速度快 11%，内存少 11%。
- ALiBi 在两倍训练长度附近性能最好，在长度 10000 上仍保持较强性能。
- 在 WikiText-103 上超过多种强位置方法。

核心公式（第3节 Attention with Linear Biases）：

softmax(q_i K^T + m·[-(i-1), ..., -2, -1, 0])

其中 m 是每个 head 固定、不学习的标量斜率(slope)。

斜率设置：对于 n 个 heads，斜率集合是从 2^(-8/n) 开始、以同样比值构成的几何序列。例如 8 heads 时斜率为 1/2, 1/4, ..., 1/256。

设计动机：ALiBi 有"偏向近处"的归纳偏置(inductive bias towards recency)——惩罚随着 query-key 距离增大而线性增大，不同 head 增大速率不同。作者尝试过让斜率可学习，但效果不如固定几何序列好（还会拖慢训练速度3%）。

论文还证明：当时的 sinusoidal、rotary (RoPE)、T5 bias 三种位置方法都做不到高效外推——T5 bias 外推效果最好但速度慢、占内存多。这正是 ALiBi 存在的理由。

## Key Figures
- Figure 1: 外推能力对比图。x 轴为验证集输入长度，y 轴为困惑度。sinusoidal/rotary/T5 三种方法随长度增加困惑度急剧上升，ALiBi 保持平稳。
- Figure 3: ALiBi 计算示意图——在 q_i·K^T 矩阵右侧加上一个常数偏置矩阵，再做 softmax。
