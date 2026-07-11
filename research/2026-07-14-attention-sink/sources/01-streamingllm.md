---
url: https://arxiv.org/abs/2309.17453
title: "Efficient Streaming Language Models with Attention Sinks"
type: arxiv_paper
authors: Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
year: 2023
accessed: 2026-07-14
quality: 5
relevance: core
---

首次系统命名并利用"attention sink"现象的论文（MIT Han Lab, ICLR 2024）。

核心发现：
- Window attention 在超出窗口后 KV cache 逐出最早的 token，会导致 perplexity 灾难性爆炸。
- 原因：模型对序列最初的几个 token（不管其语义内容）分配了远超其"重要性"的注意力分数，这些 token 起到"注意力汇"的作用。
- KV cache 可分为两部分：(1) attention sinks（最初 4 个 token，稳定注意力计算）；(2) rolling KV cache（保留最近 token，用于语言建模内容）。
- 解决方案 StreamingLLM：始终保留最初 4 个 token 的 KV + 滑动窗口最近 token 的 KV，可以让 Llama-2、MPT、Falcon、Pythia 稳定处理 400 万 token 以上的流式输入。
- 位置编码技巧：给 cache 中的 token 分配"cache 内位置"而不是"原文位置"（例如 cache 里是 [0,1,2,3,6,7,8]，分配位置 [0,1,2,3,4,5,6,7]），这一点对 RoPE/ALiBi 都关键。
- 关键实验（Table 3）：Vanilla 模型只加回 0 个 sink token 时 PPL=27.87（1024 上下文），加回 1 个 sink 后骤降到 18.49；Zero Sink（加一个全零 token）帮助有限（PPL 29214→19.90 需要更多token）；训练阶段专门加入一个 learnable Sink Token 后，仅需这一个 token 就能稳定 PPL=18.01。
- 提出原因假说：softmax 要求所有注意力权重之和为 1，即使没有语义相关的 key，模型也必须把"多余"的概率质量倾倒到某个地方；由于自回归掩码，最初的 token 对所有后续位置都可见，天然是这个"垫底"位置的最佳候选。
- 提及备选方案：SoftMax-off-by-one（分母加 1，允许权重不必和为 1）可以缓解此现象。

价值：确立了 attention sink 的工程意义（长文本流式推理），是后续几十篇后续研究的起点。
