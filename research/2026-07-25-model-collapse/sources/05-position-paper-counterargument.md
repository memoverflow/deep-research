---
url: https://arxiv.org/abs/2503.03150
title: "Position: Model Collapse Does Not Mean What You Think"
type: arxiv_paper
authors: Rylan Schaeffer, Joshua Kazdan, Alvan Caleb Arulandu, Sanmi Koyejo
year: 2025
accessed: 2026-07-25
quality: 5
relevance: critical/counterpoint
---

## 核心论点
批评"model collapse"叙事被过度简化和夸大。指出文献中实际存在 **8 种不同、有时互相矛盾的 model collapse 定义**，术语不统一导致领域内难以形成共识理解。

## 关键批评点
1. 很多"证明 model collapse 不可避免"的研究依赖脱离现实的假设（例如：完全替换真实数据、假设合成数据没有任何过滤/校验、假设无限代数递归）
2. 现实世界中：
   - 数据是累积而非替换的（见 Gerstgrasser et al. 2024）
   - 合成数据往往经过人工/自动筛选（RLHF、reward model 过滤、拒绝采样等），不是"盲目"使用
   - 模型提供商有意保留高质量人类数据源
3. 作者按照研究方法与真实世界条件的匹配程度对文献重新加权评估，得出结论：**若干被广泛引用的坍缩场景在现实条件下是可以被规避的**

## 意义
这是这个话题里最重要的"反方声音"，教学文章里必须呈现——否则读者会以为 model collapse 是确定无疑、不可逆的世界末日预言。真实情况更微妙：坍缩的数学机制是真实存在且可证明的，但其现实严重程度依赖于"数据是否被替换 vs 累积""合成数据是否被过滤"等工程决策。
