---
url: https://en.wikipedia.org/wiki/Polysemanticity
title: "Polysemanticity"
type: encyclopedia
year: 2026 (last edited)
accessed: 2026-07-13
quality: 3
relevance: supporting
---

定义：多义性 (polysemanticity) 是神经网络中的一种现象，单个神经元对多个不相关概念产生响应，而不是对应单一明确定义的概念。例子：一个多义神经元可能同时对法律文本、DNA序列和希伯来文字激活。

背景：机制可解释性 (mechanistic interpretability) 研究通常希望模型内部组件能对应到人类可读的概念。但实践中，训练出的模型里许多神经元会跨越语义无关的输入同时激活，导致难以单独分析——这是可解释性研究的核心障碍。

Superposition hypothesis 是对多义性的主流解释：真实世界数据包含的独特特征数量远超网络拥有的神经元数量，因此网络会把这些特征编码成跨越许多神经元的、互相重叠的线性组合，以此表示比它维度更多的特征。代价是单个神经元最终对多个不相关但共享表示方向的特征都产生响应。Elhage 等人论证这种权衡在损失最小化的意义上是"划算的"——用可解释性换取了表示容量。

Sparse autoencoders 是从多义表示中恢复可解释结构的一种方法：在被研究模型的激活上训练一个稀疏自编码器，学习出一组更大的方向集合，理想情况下每个方向对应一个单一概念。2023 年 Anthropic 的论文报告 dictionary learning 能把一个 512 神经元的 transformer 层分解成 4000 多个这样的特征。2024 年的后续工作把这个技术用在 Claude 3 Sonnet 上，发现了许多可解释的特征，其中一些具有安全相关性。

值得注意：MIT Technology Review 将机制可解释性列为 2026 年十大突破性技术之一。
