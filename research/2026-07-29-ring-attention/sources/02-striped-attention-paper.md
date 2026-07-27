---
url: https://arxiv.org/pdf/2311.09431
title: "Striped Attention: Faster Ring Attention for Causal Transformers"
type: arxiv_paper
authors: William Brandon, Aniruddha Nrusimha, Kevin Qian, Zachary Ankner, Tian Jin, Zhiye Song, Jonathan Ragan-Kelley
year: 2023
accessed: 2026-07-29
quality: 5
relevance: core
---

从 PDF 全文提取(前12页)。

核心发现:Ring Attention 在因果自注意力(causal self-attention,即生成式语言模型用的注意力)场景下存在工作负载不均衡问题。原因:
- 因果掩码使得约一半的 Query-Key 交互天然无效(被 mask 掉,softmax 输出为0)。
- 单卡上 FlashAttention 等实现可以利用这一点跳过约一半计算量。
- 但 Ring Attention 按连续 token 段切分给各设备:持有序列早期 token 的设备,每轮几乎都要做"完全必要"的满量计算;持有序列晚期 token 的设备,很多轮的计算结果会被因果掩码完全丢弃。
- Ring Attention 每轮迭代耗时取决于最慢的设备,因此整体表现等同于"未利用因果稀疏性的全量计算",没能省下理论上的一半算力。

解决方案 Striped Attention:
- 不按连续段切分,而是把序列打乱重排(permute),让每个设备持有分散在整条序列中、间隔均匀的 token 子集(如 4 卡 16 token 场景下,GPU0 持有 token {0,4,8,12} 而非 {0,1,2,3})。
- 利用注意力计算对 token 排列的等变性(permutation equivariance):只要 Q/K/V 按同一方式重排,结果经逆重排后与原序完全一致——因此仍是精确注意力,非近似。
- 重排后每个设备与任意 KV 块交互时,约有一半会被因果掩码阻断、一半有效——负载天然均衡。

实验结果:
- 8×A100 80GB,序列长度 25.6万 token,10亿参数级因果语言模型训练:最高 1.45× 端到端吞吐提升(vs 原始 Ring Attention)。
- 16×TPUv4,序列长度超过 50万(786k)token:最高 1.65× 加速。

代码开源: https://github.com/exists-forall/striped_attention/
