---
url: https://arxiv.org/abs/2306.15595
title: "Extending Context Window of Large Language Models via Positional Interpolation"
type: arxiv_paper
authors: Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian (Meta AI)
year: 2023
accessed: 2026-07-02
quality: 5
relevance: core
---

提出 Position Interpolation (PI)：将位置索引线性缩小以匹配原始上下文窗口大小，而不是外推超出训练时所见的范围（外推被证明会导致灾难性的高注意力分数，破坏模型）。PI 只需要在 Pile 数据集上微调约 1000 步，就能将 RoPE-based 预训练 LLM（如 LLaMA）的上下文窗口扩展到 32768，微调成本相对预训练可忽略不计。

论文的理论分析指出：直接外推（不缩放，直接让位置索引超出训练范围）会导致某些注意力头的分数出现远超训练时分布的异常值，这是外推失败的根本原因；而插值将位置索引保持在训练时见过的范围内，从根本上避免了这个问题。这也是为什么 PI 优于"直接外推 RoPE"的朴素做法。
