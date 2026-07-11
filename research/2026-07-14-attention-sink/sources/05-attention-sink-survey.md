---
url: https://arxiv.org/abs/2604.10098
title: "Attention Sink in Transformers: A Survey on Utilization, Interpretation, and Mitigation"
type: arxiv_paper
authors: (survey, 2026)
year: 2026
accessed: 2026-07-14
quality: 4
relevance: core
---

首篇针对 attention sink 的综述，系统回顾了超过 180 篇相关研究，将文献组织为三个维度：Utilization（如何利用）、Interpretation（机制解释）、Mitigation（缓解策略）。

关键内容摘录：

**解释维度**：一种被广泛讨论的解释将该现象与 softmax 的归一化行为联系起来——即使没有强相关的 key，softmax 也强制要求注意力质量被分配出去（相关引用：Bondarenko 2023 quantizable transformers、softpick、attention register 等工作）。

**利用维度**：
- Sink token preservation 是 LLM 推理中广泛采用的策略，尤其在 token pruning、KV cache 压缩、稀疏注意力机制中（引用 StreamingLLM、H2O、MInference、DuoAttention 等工作）。
- Sink Registers 被专门引入到 Diffusion Transformer (DiT) 架构中，用于在迭代去噪过程中吸收冗余的注意力质量——说明 sink 不仅是语言模型现象，在扩散模型架构中同样被主动设计利用。

**缓解维度**（量化场景）：
- 量化感知保护（Quantization-Aware Protection）：量化方法认识到 sink token 表现出极端的激活值，对数值精度损失特别敏感，因此保护这些 token 对维持模型精度至关重要（如 pivot token preservation 策略）。

意义：这篇综述确认了 attention sink 已经从一个"奇怪现象"演变为一个有专门研究方向、跨越语言模型/视觉模型/扩散模型的通用架构话题，且已经被工程界主动利用（不仅是要缓解的 bug）。
