---
url: https://arxiv.org/abs/2309.00071
title: "YaRN: Efficient Context Window Extension of Large Language Models"
type: arxiv_paper
authors: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
year: 2023
accessed: 2026-07-02
quality: 5
relevance: core
---

摘要核心信息：RoPE 是当前主流大模型的位置编码方式（LLaMA, PaLM 等），但其原始设计不能很好地泛化到训练时未见过的序列长度。YaRN (Yet another RoPE extensioN method) 是一种计算高效的方法，可以扩展这类模型的上下文窗口，所需的 token 量比之前方法少 10 倍，训练步数少 2.5 倍。YaRN 让 LLaMA 系列模型有效利用并外推到远超原始预训练允许范围的上下文长度，同时在上下文窗口扩展方面达到 SOTA，即使在训练数据有限的场景下也适用。

论文还提到 Dynamic YaRN 推理时技巧，可以让扩展上下文窗口的能力用于其他部署场景，且无需额外微调。

关键定量结果：YaRN 微调 LLaMA 2 7B/13B 到 64k/128k 上下文，在长文档语言建模任务上取得极低困惑度增长，且短序列性能几乎不掉。
