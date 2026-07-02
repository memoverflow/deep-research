---
url: https://mbrenndoerfer.com/writing/ntk-aware-scaling-context-extension
title: "NTK-aware Scaling: Extending Context Length in LLMs"
type: technical_blog
year: 2025
accessed: 2026-07-02
quality: 3
relevance: supporting
---

关键事实：NTK-aware scaling 并非来自正式论文，而是 2023 年 5 月一位独立研究者在 Reddit (r/LocalLLaMA) 上发布的帖子，目的是无需微调就能扩展 LLaMA 的上下文窗口，同时把困惑度退化降到最低。这是开源社区自发驱动技术创新、后来才被官方论文/博客（如 EleutherAI 的 YaRN 论文）系统化、理论化的典型案例。

技术要点复述：NTK-aware scaling 通过调整 RoPE 的 base frequency（而不是像 PI 一样均匀压缩位置索引）来实现扩展；在高频维度（模型学习细粒度局部关系的维度）保留接近原始的旋转速度，在低频维度（编码长距离关系的维度）进行更大压缩。这利用了 Neural Tangent Kernel 理论：低维输入若缺少高频分量，深度网络就学不好高频细节。

Dynamic NTK scaling: 在推理时依据当前实际序列长度实时计算缩放因子，而不是提前固定一个静态值，这样短序列时不会有性能损失，长序列时也不会突然崩掉。
