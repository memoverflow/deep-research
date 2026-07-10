---
url: https://mathoverflow.net/questions/24864/almost-orthogonal-vectors; https://www.cs.princeton.edu/courses/archive/fall14/cos521/lecnotes/lec11.pdf
title: "Almost Orthogonal Vectors in High Dimensions"
type: math reference
accessed: 2026-07-13
quality: 4
relevance: supporting (数学基础)
---

关键数学事实（叠加假说的几何基础）：

1. **严格正交的限制**：在 n 维实空间中，如果要求向量两两内积 ≤ 0（严格正交或钝角），最多只能放下 2n 个单位向量（一组正交基加上它的负方向）。这个数量是 n 的线性函数——非常受限。

2. **允许"几乎正交"后的爆炸性增长**：如果把条件放宽为"内积绝对值 ≤ ε"（即允许夹角在 90° 附近有一个小的容差，比如 88°-92°），那么在 n 维空间中可以放下的向量数量是**指数级**的，大约是 exp(c·ε²·n) 量级（Johnson–Lindenstrauss 型论证 / 球面填充论证的推论）。也就是说，只要允许一点点的"不完美正交"，高维空间能塞进的近似独立方向数量会远远超过维度本身。

3. 直觉来源：高维球面上随机选取的两个单位向量，它们的内积集中在 0 附近，且随维度增加集中程度越强（大数定律/集中不等式的效果）。这就是为什么"随机选出的高维向量几乎总是几乎正交的"。

这个数学事实正是 superposition hypothesis 的几何基础：一个 d 维的激活空间理论上只能放下 d 个严格正交的方向（对应 d 个"无干扰"特征），但如果允许特征方向之间存在微小的非零内积（即"几乎正交"），这个空间可以编码远超过 d 个特征，只是代价是这些特征之间会有微弱的"泄漏"或"干扰"——这正是模型付出的表示容量代价，需要靠稀疏性（大部分特征同时不激活）和非线性去噪（ReLU）来控制干扰不至于淹没信号。
