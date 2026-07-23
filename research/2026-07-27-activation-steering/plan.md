# 研究计划：激活转向 / 表征工程 (Activation Steering / Representation Engineering)

## 研究问题
大模型内部如何用"方向"编码高层行为概念（拒绝、诚实、乐观等）？如何从对比样本对中提取这些方向，
并通过运行时激活干预或永久权重编辑来控制/移除这些行为？这项技术对模型安全意味着什么？

## Level: 3 (Deep)

## 子问题分解
1. 线性表征假说——概念在残差流中作为方向存在的证据（Golden Gate Claude 案例）
2. 差值均值法 (difference-in-means)：如何从对比提示词对中提取"拒绝方向"
3. 激活加法 vs 方向消除：两种运行时干预方式的区别与效果
4. 权重正交化：如何把运行时干预变成永久的权重手术（abliteration 的数学基础）
5. CAA (Contrastive Activation Addition)：作为通用、连续可调的行为拨盘
6. 批判视角：abliteration 是否真的"外科手术式"？副作用研究

## 核心来源
- Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717)
- Panickssery et al. 2023, "Steering Llama 2 via Contrastive Activation Addition" (arXiv:2312.06681)
- Zou et al. 2023, "Representation Engineering: A Top-Down Approach to AI Transparency" (arXiv:2310.01405)
- Anthropic 2024, "Scaling Monosemanticity" (Golden Gate Claude)
- 2026, "Abliteration Is Not a Scalpel" (arXiv:2607.17427)

## 与已发布话题的关系
与已发布的"叠加态假说/多义性/稀疏自编码器"文章互补——那篇讲的是概念如何在神经元层面叠加存在，
这篇讲的是概念如何在方向层面被找到并操纵。两者共享"线性表征假说"的背景，但聚焦完全不同：
一个是被动的"读懂"，一个是主动的"改写"。
