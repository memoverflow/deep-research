---
url: https://arxiv.org/abs/1905.10650
title: "Are Sixteen Heads Really Better than One?"
type: arxiv_paper
authors: Paul Michel, Omer Levy, Graham Neubig
year: 2019
accessed: 2026-07-07
quality: 5
relevance: core
---

NeurIPS 2019。核心发现：即使模型用多头训练，测试时可以裁掉大部分头而不显著影响性能，有些层甚至可以
裁到只剩 1 个头也不影响。这说明训练出来的多头注意力存在大量冗余（redundancy）——不是所有头都在做
"独特的、不可替代的"工作。

方法：用贪心算法逐个裁剪头，衡量裁剪后每个头对最终 loss/准确率的重要性得分（Head Importance Score）。
发现头的重要性分布很不均匀：少数头承担了绝大部分有效计算，其余头可以移除。

与"多头有效"的关系：这不是否定多头注意力的价值，而是说明"训练时多头有用"（帮助优化/表示多样性）和
"推理时多头必要"（每个头都不可或缺）是两个不同的问题。论文提出训练动态（training dynamics）可能是
多头带来收益的重要原因之一，而不仅仅是最终推理时的表达能力。

CMU ML Blog 补充解读要点：对每个头做移除后测试集分数变化的消融实验，发现在机器翻译任务中，某些层
（尤其是最后几层）对多头的依赖显著高于其他层，且不同任务对头数的需求不同（翻译任务比分类任务更依赖
多头）。
