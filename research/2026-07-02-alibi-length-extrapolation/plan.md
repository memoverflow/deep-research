# 研究计划：ALiBi 和 YaRN 长度外推原理

## 研究问题
为什么 Transformer 模型在处理超出训练长度的序列时会性能崩溃？ALiBi 和基于 RoPE 的 NTK-aware/YaRN 方案分别如何解决这个问题？

## 子问题
1. 长度外推问题的本质是什么？sinusoidal/RoPE 为何做不到？
2. ALiBi 的设计原理、数学公式、实验效果
3. Position Interpolation 的原理与局限
4. NTK-aware scaling 的直觉（高频外推、低频内插）
5. NTK-by-parts 与 YaRN 的精细化处理 + 温度缩放
6. Dynamic Scaling 推理时技巧
7. ALiBi vs RoPE系方案的产业选择对比

## Level
L3 (Deep) — 单一核心主题但涉及多篇论文与技术演化脉络，需要交叉验证多个来源。

## 产出
- 1篇教学文章 (~5500字)，收录进"LLM 原理深度解析"系列 (series_order 22)
- 4 张内嵌 SVG 图表
- 6 个 sources/ 归档文件（ALiBi论文, YaRN论文, YaRN博客, PI论文, NTK-aware来源, 综述论文）
