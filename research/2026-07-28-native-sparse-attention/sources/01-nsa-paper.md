---
url: https://arxiv.org/abs/2502.11089
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
type: arxiv_paper
authors: Jingyang Yuan, Huazuo Gao, Damai Dai, et al. (DeepSeek-AI)
year: 2025
accessed: 2026-07-28
quality: 5
relevance: core
---

## Abstract
NSA (Native Sparse Attention) 是 DeepSeek 提出的一种"原生可训练"稀疏注意力机制，将硬件对齐的算子优化和算法创新结合起来，在保持模型能力的同时实现长上下文建模的效率提升。NSA 采用动态分层稀疏策略：粗粒度的 token 压缩 + 细粒度的 token 选择，同时保留全局上下文感知和局部精度。两个核心创新：(1) 通过算术强度均衡的算法设计实现大幅加速；(2) 支持端到端训练，降低预训练计算量而不损失模型性能。27B 参数模型 pretraining 实验显示 NSA 在通用 benchmark、长上下文任务和推理任务上达到或超过全注意力（Full Attention）模型，同时在 64k 长度序列上实现解码、前向、反向传播全阶段的显著加速。

## 关键内容摘录

### 问题动机 (Section 1, 2)
- 长上下文建模已成为下一代 LLM 关键能力，但 vanilla attention 的高复杂度是关键延迟瓶颈
- 理论估计：在 64k 长度上下文解码时，attention 计算占总延迟的 **70-80%**
- 现有稀疏方法两大失败模式：
  1. **推理效率的幻觉 (The Illusion of Efficient Inference)**：
     - Phase-Restricted Sparsity: 像 H2O 只在解码阶段稀疏但预填充阶段仍需密集计算；MInference 只在预填充阶段稀疏。没有一个方法能在所有推理阶段都加速。
     - 与先进架构（GQA/MQA）不兼容：像 Quest 这种方法每个 attention head 独立选择 KV 子集，在 MHA 下有效，但在 GQA 下，同一组内所有 query head 的选择取并集，导致 KV cache 内存访问量仍然很高——理论上算力省了，内存访问却没有真正减少
  2. **可训练稀疏性的迷思 (The Myth of Trainable Sparsity)**：
     - 事后（post-hoc）施加稀疏性会让模型偏离预训练时的优化轨迹
     - **关键数据点**：Top 20% 的 attention 只能覆盖 70% 的总 attention scores，这意味着像"检索头"(retrieval heads) 这样的结构在推理时剪枝会很脆弱
     - 非连续离散操作（如 ClusterKV 的 k-means 聚类、MagicPIG 的 SimHash）导致计算图不连续，梯度无法通过 token 选择过程传播
     - Token 粒度的选择（如 HashAttention）导致需要从 KV cache 中加载大量不连续的单个 token，这种非连续内存访问阻碍了 FlashAttention 等技术的高效适配

### 核心方法 (Section 3)
- 三条并行注意力分支，每个 query 都同时经过：
  1. **压缩注意力 (Compression)**：将连续 block 的 key/value 聚合成 block-level 的压缩表示，捕捉粗粒度、高层语义信息。用带内 block 位置编码的可学习 MLP 映射
  2. **选择注意力 (Selection)**：从压缩阶段获得的 attention score 复用作为 block 重要性分数（无需额外计算开销），选出 top-n 个重要 block，保留其细粒度的原始 key/value
  3. **滑动窗口注意力 (Sliding Window)**：维护最近 w 个 token 的独立分支，专门处理局部上下文——因为局部模式学习速度快，若不隔离会"抢跑"掉压缩和选择分支的学习信号
- 三分支输出通过一个 MLP+sigmoid 学出的门控分数 g_c 加权求和
- **GQA 感知的重要性分数共享**：block 重要性分数在同一 GQA 组内所有 query head 间求和共享，确保同组内所有 head 选择一致的 block，从而最小化 KV cache 加载量
- **超参数**（27B 模型实验）：压缩 block 长度 l=32，滑动步长 d=16，选择 block 大小 l'=64，选择 block 数 n=16（含固定的 1 个初始 block + 2 个局部 block），滑动窗口 w=512

### Kernel 设计 (Section 3.4)
- Group-Centric Data Loading: 把同一 GQA 组内所有 query head 的 query 一起加载到 SRAM（因为它们共享同样的稀疏 KV block）
- Shared KV Fetching: 按 block index 顺序加载连续 KV block 到 SRAM，最小化内存加载
- Outer Loop on Grid: 由于不同 query block 的 inner-loop 长度几乎相同（都是选中的 block 数量 n），把 query/output 循环放到 Triton 的 grid scheduler 里
- 目标：实现接近最优的算术强度（arithmetic intensity），通过组内共享消除冗余 KV 传输，并平衡 GPU SM 间的计算负载

### 实验结果
- 27B 参数、3B 激活参数（结合 GQA + MoE, DeepSeekMoE 结构），270B token 预训练
- 通用 benchmark：NSA 平均分 0.456 vs Full Attention 0.443（NSA 略优）
- LongBench：NSA 平均分 0.503（部分子任务超过 Full Attention 的 0.512），远超 H2O(0.428)、InfLLM(0.474)、Quest(0.495)、Exact-Top(0.502)
- 64k 长度序列上：decode 11.6x、forward 9.0x、backward 6.0x 加速比

## Key Figures
- Figure 2: NSA 架构总览 —— 三分支（压缩/选择/滑动窗口）并行处理，最终门控融合
  - Source: NSA paper Figure 2
  - Content: 展示了每种分支产生的稀疏 attention mask 模式：绿色为需要计算的区域，白色为可以跳过的区域
- Figure 3: Kernel 设计 —— Grid Loop（按 GQA 组循环 query）+ Inner Loop（加载对应稀疏 KV block）+ SRAM 上计算
