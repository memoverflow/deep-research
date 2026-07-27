---
url: https://arxiv.org/abs/2310.01889
title: "Ring Attention with Blockwise Transformers for Near-Infinite Context"
type: arxiv_paper
authors: Hao Liu, Matei Zaharia, Pieter Abbeel
year: 2023
accessed: 2026-07-29
quality: 5
relevance: core
---

Abstract & key sections extracted via arxiv HTML (arxiv.org/html/2310.01889):

Transformers 的自注意力内存开销随序列长度平方增长,导致长序列训练/推理受限于单卡内存。作者提出 Ring Attention with Blockwise Transformers,利用 blockwise 自注意力和 FFN 计算,把长序列分布到多个设备,并让 KV 块的通信与 blockwise 注意力计算完全重叠,从而不引入额外的通信/计算开销。

核心机制:
- 设备组成一个逻辑环(ring)。每个设备持有一个 query 块,并在内循环中让 KV 块沿环传递。
- 每个设备计算完当前 KV 块的 blockwise attention 后,立即把该 KV 块发给下一个设备,同时从上一个设备接收下一块 KV——通信与计算并发。
- 只要"块计算耗时 ≥ 块传输耗时",通信就被完全隐藏,不增加wall-clock开销。
- 利用 FlashAttention 式在线 softmax 的可重排序性质(排列不变性),保证多块累积结果与整体计算完全一致(精确注意力,非近似)。

内存对比表(每层激活,字节,bfloat16):
- Vanilla Transformer: 2bns² (自注意力) + 8bsh (FFN)
- Memory efficient attention (Rabe & Staats 2021): 2bsh+4bch + 8bsh
- Memory efficient attention+FFN (BPT, Liu & Abbeel 2023b): 2bsh + 2bsh
- Ring Attention: 6bch (自注意力) + 2bch (FFN) —— 与总序列长度 s 脱钩,只跟块大小 c 相关

Arithmetic intensity 条件(决定通信能否被计算完全掩盖):
4dc²/F ≥ 4cd/B ⟹ c ≥ F/B
其中 F=单设备FLOPS, B=设备间带宽, c=块大小, d=隐藏维度。

实验结果:在 TPUv4-1024 上,Ring Attention 可训练比之前最优内存高效方案长 500 倍以上的序列,支持超过 1 亿 token 的序列长度训练,且不做近似、无额外通信计算开销。

代码: https://github.com/lhao499/llm_large_context
