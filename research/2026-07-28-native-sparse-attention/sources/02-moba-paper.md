---
url: https://arxiv.org/abs/2502.13189
title: "MoBA: Mixture of Block Attention for Long-Context LLMs"
type: arxiv_paper
authors: Enzhe Lu, Zhejun Jiang, Jingyuan Liu, et al. (Moonshot AI / Kimi)
year: 2025
accessed: 2026-07-28
quality: 5
relevance: core
---

## Abstract
MoBA 把 Mixture-of-Experts (MoE) 的思想应用到 attention 机制本身：不再让每个 query 关注全部上下文，而是把上下文切成 block，让一个"无参数"的门控网络为每个 query 动态选出最相关的 top-k 个 block。核心设计原则是"少结构"(less structure)：不像 sliding window / attention sink 那样预先规定注意力的形状，而是让模型自己决定该看哪里。MoBA 已经部署在 Kimi 的长上下文请求中，其关键优势是可以在全注意力和稀疏注意力之间无缝切换。

## 关键内容摘录

### 问题背景 (Section 1)
- 现有方法要么"结构太强"（sink/window attention，任务特定，泛化性差），要么"改动太大"（线性注意力如 Mamba/RWKV/RetNet，转换成本高，复杂推理任务上效果存疑）
- MoBA 的定位问题：能不能设计一个**保留原始 Transformer 框架**、同时坚持"少结构"原则、让模型自主决定注意力位置的架构？

### 核心方法 (Section 2)
- 数学形式：MoBA(q,K,V) = Softmax(qK[I]^T) V[I]，I 是被选中的 key/value 索引集合
- 把长度 N 的上下文切成 n 个 block，block i 覆盖范围 [(i-1)*B+1, i*B]
- **无参数门控 (parameter-less gating)**：block i 的相关性分数 s_i 就是 query 和该 block 内 key 的 mean pooling 的内积 —— 不引入任何新的可学习权重（这点和 MoE 的 router 不同，MoE router 通常有专门的线性层）
- 对每个 query，用 top-k 门控选出 k 个最相关 block（Topk({s_j}, k)），未被选中的 block gate=0
- **因果性保证的两个设计**：
  1. 禁止路由到未来 block：对于 pos(q) < i×B 的 block，直接设 s_i = -∞, g_i = 0
  2. "当前块"必须强制路由 + 因果 mask：因为 mean pooling 整个 block 会泄漏未来信息，所以当前块（包含 query 自身的 block）必须强制选中(g_i=1)，并在块内部再加因果 mask。作者类比说这个"当前块必选"就像 MoE 里的共享专家(shared expert)机制
- **MoBA 是 sliding window / attention sink 的推广**：作者证明这两种经典静态稀疏模式都可以看作 MoBA 的特例——sliding window 相当于门控网络恒定选择"最近的block"，attention sink 相当于门控网络恒定选择"最初的block + 最近的block"。因此 MoBA 具有比它们更强的表达能力
- **平滑过渡全注意力 ↔ MoBA**：因为 MoBA 参数量与全注意力完全相同（不增不减），可以在训练中动态切换某一层用全注意力还是 MoBA，这为渐进式训练策略提供了灵活性

### 实现 (Section 2.3)
- 借助 FlashAttention 和 MoE 的优化技术，实现五个步骤：
  1. 根据门控网络和因果 mask 确定 query-to-KV-block 的分配
  2. 按分配的 KV block 重新排列 query token 顺序
  3. 用支持可变长度的 FlashAttention 计算每个 block 内的 attention 输出
  4. （后续步骤见原文 Algorithm 1，涉及输出的重新聚合与归一化）
- 已经在生产环境中部署支持 Kimi 的长上下文请求，1M 上下文长度的 needle-in-haystack 评测中验证了检索能力

## Key Figures
- Figure 1: MoBA 架构图 —— (a) 两个 query、四个 KV block 的路由示例：query1 被路由到 block 1&2，query2 被路由到 block 3&4；(b) MoBA 与 FlashAttention 的整合方式
