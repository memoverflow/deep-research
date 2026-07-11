---
url: https://arxiv.org/abs/2306.12929
title: "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing"
type: arxiv_paper
authors: Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort (Qualcomm AI Research)
year: 2023
accessed: 2026-07-14
quality: 5
relevance: core
---

比 StreamingLLM 更早（2023 NeurIPS）从"量化"角度触及了这个现象，并提出了修改架构本身来解决问题的思路——标题直译就是"通过帮助注意力头什么都不做来消除outlier"，这与 Barbero 后来提出的"no-op"解释高度一致，说明这是独立发现的收敛结论。

核心发现：
- Transformer（论文以 BERT-base 为例，12 头 × 64 维）在训练后会出现极端的激活值 outlier，这使得量化（把浮点数压缩成低比特整数）非常困难——因为量化需要给定一个数值范围，而这些 outlier 会把范围撑得极大，压垮其余正常数值的精度。
- 通过分析这些 outlier 对应的注意力头的注意力模式，发现它们本质上在执行"什么都不做"的操作——即让某个 token（往往是无语义的分隔符/首 token）吸收几乎全部注意力，从而使该 head 对当前 token 的表示不产生实质性更新。
- 提出两个独立的架构修改：
  1. **Clipped softmax**：让 softmax 的输出范围可以低于原始下界（允许"负的"注意力权重被裁剪到 0 附近），从而不需要通过产生巨大激活值来实现"不更新"效果。
  2. **Gated attention**：为每个注意力头引入一个可学习的门控，允许头直接输出"不更新"的信号，而不需要依赖某个 token 吸收全部注意力这种迂回方式。
- 实验结果：使用这两种方法预训练的模型学到的 outlier 显著更小，同时保持甚至提升了浮点精度下的任务表现，量化后的效果也更好。

意义：这篇论文从工程侧（量化落地）证明了"attention sink 是模型为了实现'某些 head 不更新表示'这一诉求，被迫借助数值 outlier 达成的一种笨拙但有效的手段"。这与后续的理论解释（over-mixing、no-op）互相印证，也说明这不是一个孤立观察，而是横跨"量化"、"流式推理"、"表示学习理论"三个不同社区反复独立发现的同一现象。
