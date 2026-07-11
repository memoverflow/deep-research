---
url: https://arxiv.org/abs/2504.02732
title: "Why do LLMs attend to the first token?"
type: arxiv_paper
authors: Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu (Oxford / NUS / Google DeepMind)
year: 2025
accessed: 2026-07-14
quality: 5
relevance: core
---

这是解释 attention sink "为什么会被学出来"最有理论深度的论文。

核心论点：attention sink 是 Transformer 用来**避免 over-mixing（过度混合）**的一种机制，而不是缺陷或副产品。

理论背景：
- Rank collapse（秩坍塌）：纯注意力堆叠多层后，所有 token 的表示会收敛到彼此非常相似（低秩），这是比"representational collapse"更强的一种退化条件。
- Over-squashing / over-mixing：信息在图/序列上传播时，远距离 token 之间的交互会被过度压缩/混合，导致模型丢失对局部、精确信息的保留能力（这是图神经网络文献中的成熟概念，作者将其类比到 Transformer）。
- 关键洞察：如果每个 token 都把注意力"均匀地"分给上下文里所有其他 token，多层叠加后模型会越来越难区分"这是当前位置该重点关注的信息"和"背景噪音"——表示会被过度平均、过度混合。

Sink 的作用："近似 no-op"：
- 让大量 attention 集中到第一个 token（通常是 BOS），意味着该 head 在这一层几乎不从其它 token 汲取新信息，实际上执行的是一次"什么都不做"的操作(no-op)，从而保留了上一层已经算好的表示不被稀释。
- 这类似于残差连接的作用：允许某些 head/层选择"不更新"当前 token 的表示，为模型提供了"按需计算"的自由度。

实验验证：
- 在 Gemma 7B 上验证了该假说，展示 sink head 具体如何构造出近似 no-op。
- 更深的模型、更长上下文训练的模型会形成更强的 sink（over-squashing 分析预测，并在 LLaMA 3.1 系列上得到验证）：LLaMA 3.1 405B 中高达 80% 的注意力头形成了强 sink。
- BOS 本身并不"特殊"：不管训练时 BOS 如何插入，sink 都会自然形成在序列的第一个位置；但如果预训练时把 BOS 固定在第一个位置，会影响 sink 具体的构造方式。
- Context length、depth、data packing 都会影响 sink 行为的强弱。

引用的下游影响：quantisation difficulties（Liu et al. 2024）、improved KV-caching（Ge et al. 2024）、streaming attention（Xiao et al. 2024）、security vulnerabilities（Yona et al. 2025）。

价值：把 sink 从"观察到的怪现象"提升为"可以从信息传播理论推导出的必然结果"，是理解"为什么"而非仅仅"是什么"的关键论文。
