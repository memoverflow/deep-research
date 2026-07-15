---
url: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
title: "In-context Learning and Induction Heads"
type: blog_post_technical
authors: Catherine Olsson, Nelson Elhage, Neel Nanda, et al. (Anthropic)
year: 2022
accessed: 2026-07-15
quality: 5
relevance: core
---

Anthropic 团队关于归纳头(induction heads)与上下文学习(ICL)关系的原始论文/技术博客(Transformer Circuits Thread)。

核心内容摘录：

- 归纳头定义：一种在 [A][B] ... [A] → [B] 模式下，通过"匹配-复制"操作补全序列的 attention head。机制上由两个不同层的 head 组成电路：第一个是"上一个 token 头"(previous token head)，负责把前一个 token 的信息复制到当前 residual stream；第二个是真正的归纳头，利用第一个 head 写入的信息，寻找"当前 token 之前出现过的相同 token"，并复制那个位置后面跟着的 token 作为预测。

- 关键假说：归纳头可能是大型 Transformer 模型中绝大部分 ICL 能力的机制来源。特别是它的"模糊/近邻"版本：[A*][B*] ... [A] → [B]，其中 A* ≈ A，B* ≈ B，能泛化到语义相近而非完全一致的模式匹配（如翻译任务）。

- ICL 的量化定义：比较模型在序列中第 500 个 token 位置的 loss 与第 50 个 token 位置的 loss 之差，作为 ICL 强度的度量指标。

- 核心发现：训练过程中存在一个"相变"(phase change)，几乎所有层数大于 1 的 Transformer 都会出现，表现为训练损失曲线上的一个"鼓包"(bump)。这个相变发生的时刻，恰好与归纳头在模型内部形成的时刻精确重合。

- 六条互补证据线：
  1. Argument 1 (Macroscopic co-occurrence)：归纳头形成时刻与损失鼓包时刻的宏观共现，跨模型规模一致。
  2. Argument 2 (Macroscopic co-perturbation)：扰动训练设置使鼓包位置移动，归纳头形成时刻同步移动，说明因果关联而非巧合。
  3. Argument 3 (Direct ablation)：直接消融被识别为归纳头的组件，ICL 表现显著下降；保留归纳头 attention pattern 而消融其他部分，ICL 大部分保留。这是最强的因果证据。
  4. Argument 4 (Examples of generality)：归纳头能处理字面复制(literal sequence copying)、翻译(translation)、模式匹配(pattern matching)等多种泛化行为，不只是死板复制。
  5. Argument 5 (Mechanistic plausibility)：归纳头所需电路结构是 Transformer 架构下"最省力"就能实现的路径之一。
  6. Argument 6 (Continuity)：从小型 attention-only 模型到带 MLP 的中型模型到全规模语言模型，归纳头相关特征连续变化，无明显断层。

- 证据强度的自我克制：对于小型 attention-only 模型，提供的是强因果证据(direct causal evidence)；对于带 MLP 的更大模型，由于电路复杂难以精确拆解，提供的更多是相关性证据(correlational evidence)。论文措辞为"preliminary and indirect evidence"。

- 术语补充：K-composition — 归纳头的 query 直接来自当前 token 的 embedding，但其匹配的 key 并非来自 token 原始表征，而是来自前一层"上一个 token 头"写入 residual stream 的信息。这种跨层信息借用是归纳头能实现两步推理的关键机制。

- Residual stream 概念：Elhage et al. (2021) 《A Mathematical Framework for Transformer Circuits》引入的术语，指每层之后的 embedding，后续层从中读取信息、写回信息，类比"流"强调残差跳连的性质。

摘要(abstract 原文)：
"Induction heads" are attention heads that implement a simple algorithm to complete token sequences like [A][B] ... [A] → [B]. In this work, we present preliminary and indirect evidence for a hypothesis that induction heads might constitute the mechanism for the majority of all "in-context learning" in large transformer models (i.e. decreasing loss at increasing token indices). We find that induction heads develop at precisely the same point as a sudden sharp increase in in-context learning ability, visible as a bump in the training loss. We present six complementary lines of evidence, arguing that induction heads may be the mechanistic source of general in-context learning in transformer models of any size. For small attention-only models, we present strong, causal evidence; for larger models with MLPs, we present correlational evidence.
