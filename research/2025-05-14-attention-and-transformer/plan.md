# Research Plan: Attention 机制与 Transformer 架构穷尽调研

## 精确研究问题
Attention 机制和 Transformer 架构的完整技术演进：从 2014 年 Bahdanau attention 到 2025 年最新变体。涵盖数学原理、关键论文原文、架构图解、性能数据、工程优化、以及前沿替代方案。

## Level: 4 (Exhaustive)
- 目标：50-100+ searches, 30-50 full extractions, 10-20 arxiv papers, 10-20 key images

## Sub-questions (8 个 MECE)

1. **起源 (2014-2016)** — Seq2seq → Bahdanau → Luong → Self-attention 前身
2. **Transformer 核心 (2017)** — 原始论文完整架构、数学、训练细节
3. **位置编码 (2017-2025)** — Sinusoidal → Learned → Relative → RoPE → ALiBi → YaRN
4. **高效 Attention (2019-2025)** — Flash, Sparse, Linear, MQA/GQA/MLA, PagedAttention
5. **架构变体 (2018-2025)** — BERT/GPT/T5/MoE/SSM/Hybrid
6. **Scaling & Training (2020-2025)** — Scaling laws, 并行策略, 后训练
7. **2024-2025 前沿** — Mamba/RWKV/RetNet/Griffin/xLSTM/TTT
8. **开放问题** — 理论局限、未来方向、Attention 是否被取代

## 执行计划
- Pass 1: 子问题 1, 2, 3 并行（基础，其他依赖）
- Pass 2: 子问题 4, 5, 6 并行
- Pass 3: 子问题 7, 8 并行 + 图片收集补充
- Pass 4: Browser 补充（Google Scholar, 关键架构图下载）
- Pass 5: 综合写报告 + 归档 + push
