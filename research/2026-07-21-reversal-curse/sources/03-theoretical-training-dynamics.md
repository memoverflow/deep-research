---
url: https://arxiv.org/abs/2405.04669
title: "Towards a Theoretical Understanding of the 'Reversal Curse' via Training Dynamics"
type: arxiv_paper
authors: Hanlin Zhu, Baihe Huang, Shaolun Zhang, Michael Jordan, Jiantao Jiao, Yuandong Tian, Stuart Russell
year: 2024
venue: NeurIPS 2024
accessed: 2026-07-21
quality: 5
relevance: core
---

## Core Contribution
用（随机）梯度下降的训练动力学，对两种自回归模型做严格理论分析：
1. Bilinear model —— 可以看作是一层 transformer 的简化版
2. One-layer transformer —— 用 Tian et al. (2023a) 的框架分析

## Key Theoretical Finding: Weight Asymmetry
核心结论：两种自回归模型的（有效）权重都表现出**不对称性 (asymmetry)**——从 token A 到 token B 的权重在训练中增加，并不必然导致从 B 到 A 的权重也增加。

这解释了为什么"A is B"训练之后，"B is A"的能力不会自动出现：模型内部学到的其实是一个**方向性的关联强度**，而不是一个对称的"事实"表示。梯度下降本身就没有对称化这个权重矩阵的机制——除非损失函数明确要求。

## Extension to Chain-of-Thought
这个分析框架可以自然推广到其他逻辑推理任务，比如 Chain-of-Thought (CoT)。论文证明：一个训练过 "A→B" 和 "B→C" 的一层 transformer，如果不显式生成中间推理步骤 (CoT)，无法直接推出 "A→C"（也是 Allen-Zhu and Li 2023 观察到的经验现象）。

这和之前 Feng et al. (2024) 从 expressivity (表达能力) 角度分析 CoT 必要性不同——这篇论文是从**训练动力学**角度证明 CoT 的必要性，提供了新的理论视角。

## Existing Mitigations Have Trade-offs
论文指出：现有缓解方法（比如反转训练数据集、使用替代训练目标）通常会负面影响模型在其他任务上的表现，或者需要架构改动——说明这不是一个"打个补丁"就能完全解决的问题，而是根植在自回归+梯度下降这套训练范式里的结构性限制。

## Relevance
提供了 Reversal Curse 最严格的数学解释：权重矩阵的不对称性。把现象从"经验观察"提升到了"可证明的训练动力学结论"，并把它和 CoT 的必要性联系起来——这是文章"技术细节"部分的核心素材。
