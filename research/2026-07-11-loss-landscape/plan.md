# 研究计划：损失景观（Loss Landscape）与平坦/尖锐最小值

## 研究问题
损失景观的几何形状（平坦 vs 尖锐最小值）如何影响神经网络的泛化能力？大批量训练、SAM、
edge of stability 等现象背后的几何直觉是什么？

## Level
L3 — 15+ web_search, 7 篇 arxiv 论文全文/摘要提取

## 子问题
1. 损失景观的基本概念：平坦最小值假说
2. 大批量训练为什么容易导致泛化差距（sharp minima）
3. SAM：主动寻找平坦最小值的优化算法
4. 2025 年最新反直觉发现：Sharp Minima Can Generalize（体积假说）
5. Edge of Stability：梯度下降的动态稀疏性悬崖行走现象

## 搜索执行
- 15+ web_search queries (loss landscape, flat/sharp minima, SAM, batch size generalization,
  Hessian eigenvalue spectrum, edge of stability, saddle points)
- 7 篇 arxiv 论文 abstract 通过 curl 直接提取（web_extract 因 DDGS-only backend 不支持，
  改用 terminal curl 绕过限制）
