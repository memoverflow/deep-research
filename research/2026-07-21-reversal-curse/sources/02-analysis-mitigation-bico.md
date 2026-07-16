---
url: https://arxiv.org/pdf/2311.07468
title: "An Analysis and Mitigation of the Reversal Curse"
type: arxiv_paper
authors: Ang Lv, Kaiyi Zhang, Shufang Xie, Quan Tu, Yuhan Chen, Ji-Rong Wen, Rui Yan
year: 2023
accessed: 2026-07-21
quality: 5
relevance: core
---

## Core Contribution
第一篇专门研究 Reversal Curse **成因**的论文。核心论点：Reversal Curse 源于具体的训练目标——尤其是绝大多数因果语言模型使用的 next-token prediction (NTP) 目标。

## Key Mechanism (因果 attention mask)
Berglund et al. (2023) 只测试了 Llama 和 GPT 系列。这些模型用**因果注意力掩码 (causal attention mask)**，每个 token 只能依赖前面的 token。用 NTP 预训练时，模型只在最大化 p(b|a)（entity a 在前，b 在后），对 p(a|b) 没有任何保证。

对比：像 GLM 用 autoregressive blank infilling (ABI) 目标训练——被 mask 掉的 token 可以同时attend 到前面和后面的 token。这隐含地考虑了反向条件似然 p(a|b)，因此 ABI 训练的模型可能对 Reversal Curse 更鲁棒。

## Verification Experiment
在 GLM 上用同样的 name-to-description 微调数据测试（"Joe Biden is the American president"→ 反向补全"The American president is."→ 期望"Joe Biden"）。
定义任务：
- N2D task: 用 name→description 数据训练，同顺序测试
- 反向 N2D (←N2D) task: 用 name→description 数据训练，反向顺序测试

## Proposed Fix: BICO
论文提出一种新的微调方法 **BICO** (Bidirectional Casual language modeling Optimization)，设计目标是在不改变预训练模型架构的前提下，规避引入额外的 reversal curse，同时更好地利用训练数据。

## Key Insight (for article)
这篇论文的价值在于把"为什么"落到了具体机制：**训练目标的方向性**。因果注意力掩码 + next-token prediction 的组合，从数学上就只优化了一个方向的条件概率，另一个方向的条件概率没有任何梯度信号去优化它——除非训练数据里显式包含反向排列的句子。

## Relevance
提供了 Reversal Curse 成因的第一个具体机制解释（训练目标视角），并给出了缓解方法（BICO）。是从"现象"到"机制"的关键桥梁论文。
