---
url: https://arxiv.org/abs/2404.07129
title: "What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation"
type: arxiv_paper
authors: Aaditya K. Singh, Ted Moskovitz, Felix Hill, Stephanie C. Y. Chan, Andrew M. Saxe
year: 2024
accessed: 2026-07-15
quality: 5
relevance: core
---

后续论文，深入分析归纳头形成的动态过程，是对 Olsson et al. 2022 假说的重要延伸和细化。

核心内容摘录：

- 研究动机：尽管归纳头(IH)存在的证据很强，且与训练中的相变现象高度相关，但关于 IH 的多样性和形成动态知之甚少。为什么会存在多个 IH？它们之间如何相互依赖？为什么 IH 看起来"忽然"出现，背后有哪些子电路使其能够形成？

- 方法：受神经科学"光遗传学"(optogenetics)启发，开发了一种"因果穿越训练"(causal-through-training)的"钳制"(clamping)框架，能够在训练过程的任意时刻精确操纵模型内部激活值，观察对归纳电路最终形成的因果影响。在 JAX/Equinox 实现，支持 jax.jit 加速。

- 实验设置：在 Omniglot 数据集上训练简化的少样本学习（FSL）任务，使用因果的 2 层 attention-only transformer（延续 Olsson et al. 2022 的模型设置）。序列由多个"样例-标签"对构成，标签在每个序列中随机打乱，强迫模型必须使用 ICL 才能最小化训练损失。

- 关键发现一：多个归纳头会同时形成，贡献是加和式的(additive)，且存在功能冗余(redundancy)——即使没有应用 dropout 等正则化手段，去掉部分归纳头，其他归纳头仍能部分补偿，这与大规模语言模型中观察到的多头冗余现象(Michel et al. 2019; Voita et al. 2019)相呼应。

- 关键发现二：归纳头和上一个token头之间的连接是多对多(many-to-many)而非一对一(one-to-one)。

- 关键发现三（核心结论）：通过 clamping 分析，识别出三个平滑演化的底层子电路(subcircuits)，它们相互作用产生了看似离散的"相变"表象。也就是说，那个损失曲线上的鼓包，底层实际上是几个连续变化的子过程汇合越过某个门限的结果，而非单一机制的突变。这三个子电路中，一个此前被认为"容易实现"的"复制"(copy)操作，被证明其实是关键限制性因素（此前研究更关注"匹配"match 操作）。

- 关键发现四：数据属性(data properties)会影响相变发生的时间点（timing），这种时移可以通过分别理解各子电路的数据依赖形成动态来解释。

- 意义：这篇论文没有否定 Olsson et al. 2022 的核心假说，而是"提高了分辨率"——将一个看起来单一的"顿悟"现象，分解为多个可独立操纵、连续演化的子机制的叠加结果。同时开源了代码库 icl-dynamics 供社区使用。

Abstract 原文摘要：
In-context learning is a powerful emergent ability in transformer models. Prior work in mechanistic interpretability has identified a circuit element that may be critical for in-context learning -- the induction head (IH), which performs a match-and-copy operation. During training of large transformers on natural language data, IHs emerge around the same time as a notable phase change in the loss. Despite the robust evidence for IHs and this interesting coincidence with the phase change, relatively little is known about the diversity and emergence dynamics of IHs. Why is there more than one IH, and how are they dependent on each other? Why do IHs appear all of a sudden, and what are the subcircuits that enable them to emerge? We answer these questions by studying IH emergence dynamics in a controlled setting by training on synthetic data. In doing so, we develop and share a novel optogenetics-inspired causal framework for modifying activations throughout training. Using this framework, we delineate the diverse and additive nature of IHs. By clamping subsets of activations throughout training, we then identify three underlying subcircuits that interact to drive IH formation, yielding the phase change. Furthermore, these subcircuits shed light on data-dependent properties of formation, such as phase change timing, already showing the promise of this more in-depth understanding of subcircuits that need to "go right" for an induction head.
