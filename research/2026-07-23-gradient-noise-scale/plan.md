# 研究计划：梯度噪声尺度与临界批量大小

## 话题
从 topics-published.md 去重检查确认未发布过。选定话题：Gradient Noise Scale / Critical Batch Size —— 训练原理池中"批量大小理论"这一空白点。

## 核心问题
1. 为什么加大 batch size 不能无限提速训练？
2. 梯度噪声尺度 (Gradient Noise Scale) 的数学定义与直觉
3. 临界批量大小 (Critical Batch Size) 与时间/计算效率权衡
4. 噪声尺度随训练/任务/模型规模如何变化
5. GPT-3 等真实系统的应用案例
6. 2025年的理论修正 (Adam vs SGD scaling rule, 分支训练法)

## 材料来源
1. McCandlish et al. 2018 "An Empirical Model of Large-Batch Training" (arxiv 1812.06162) - 核心论文，完整提取
2. Merrill et al. 2025 "Critical Batch Size Revisited" (arxiv 2505.23971) - 后续修正论文，完整提取
3. GPT-3 论文 (arxiv 2005.14165) 中的相关引用确认

## 输出
单篇 L3 教学文章，约7000字，2张内嵌SVG图（转折点曲线图 + 双曲线权衡图）
