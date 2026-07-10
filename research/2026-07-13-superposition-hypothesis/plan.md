# 研究计划：叠加态假说 (Superposition Hypothesis)

## 研究问题
神经网络中单个神经元经常对多个不相关概念产生激活反应（多义性 polysemanticity）。这篇文章要讲清楚：
1. 这个现象是什么、为什么重要
2. Anthropic 2022 年 "Toy Models of Superposition" 论文如何用数学工具解释它
3. 叠加背后的几何/数学基础（高维空间中"几乎正交"向量数量的指数增长）
4. 相变现象与几何结构（多胞体几何）
5. 稀疏自编码器 (SAE) 如何反向拆解叠加编码，Scaling Monosemanticity 论文在 Claude 3 Sonnet 上的成果
6. Golden Gate Claude 案例展示的"可操控性"
7. 当前局限性

## Level
Level 3（单篇深度教学文章，聚焦机制可解释性这一新颢题）

## 素材来源
- arxiv 2209.10652 (Toy Models of Superposition) — 核心论文，全文提取
- transformer-circuits.pub Scaling Monosemanticity (2024) — 核心论文，全文提取
- Wikipedia Polysemanticity 条目 — 背景与定义
- 数学参考：几乎正交向量的高维几何 (MathOverflow, Princeton lecture notes)
- Adam Karvonen SAE 直觉性博客 — 补充技术细节和局限性讨论

## 结构
单篇文章，故事线：多义性现象 → 叠加假说的直觉与数学基础 → 相变与几何结构 → SAE 如何拆解叠加 → Golden Gate Claude 操控案例 → 意义与局限
