---
url: https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md
title: "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"
type: technical_blog
authors: Microsoft DeepSpeed Team
year: 2023
accessed: 2026-07-29
quality: 4
relevance: comparison
---

DeepSpeed Ulysses 是另一条上下文并行(序列并行)路线,与 Ring Attention 形成对比:

- 核心机制:按注意力头切分(而非按 token 切分)。先将序列均分给各设备(每设备拿连续 token 段),计算注意力前做一次 all-to-all 通信,把布局重组为"每设备持有完整序列长度,但只负责部分注意力头";算完注意力后再做一次 all-to-all 换回按 token 切分的布局,继续算 FFN。
- 优势:可以直接复用单卡上未经改造的 FlashAttention 实现,工程改动小;支持不同类型的 attention(标准/稀疏)。
- 局限:并行度受注意力头数量上限制约(比如只有 32 个头,最多能切 32 份),且与张量并行(同样切分头维度)存在资源冲突,限制了可扩展性上限。
- 与 Ring Attention 对比(来自 HuggingFace ulysses-sp 博客及多篇工程对比文章):Ring Attention 无头数上限,可无限扩展序列长度(随设备数线性增长),但需要改造注意力内部循环,工程复杂度更高;Ulysses 实现简单但存在头数瓶颈。后续出现 Unified Sequence Parallelism (USP) 等方案将两者结合。
