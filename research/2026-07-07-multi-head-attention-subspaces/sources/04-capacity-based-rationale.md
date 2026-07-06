---
url: https://arxiv.org/abs/2509.22840
title: "A Capacity-Based Rationale for Multi-Head Attention"
type: arxiv_paper
authors: Micah Adler
year: 2025
accessed: 2026-07-07
quality: 5
relevance: core
---

提出一个新的、信息论视角的多头注意力合理性论证。定义"Relational Graph Recognition"任务：key-query
channel 需要编码一个有向图（谁和谁相关），并能从给定的上下文子集中恢复每个顶点的邻居。

核心结果：在固定 key 维度预算 D_K = h·d_k 的情况下，证明恢复 m' 个关系所需的 D_K 需要随
m'/d_model 增长（信息论下界+构造性上界均证明）。关键发现是：即使在最简单的场景（"permutation
graph"，每个 query 只对应一个 target），把固定的 D_K 预算拆分成多个头也能提升容量——因为这样做
减少了"embedding superposition"造成的干扰（多个不同语义关系挤在同一个向量空间里互相干扰）。

直觉翻译：如果只用一个大头，所有类型的"相关性判断"都要在同一个坐标系里完成，不同类型的关系信号会
相互干扰（类似多个电台挤在同一频段互相串音）；拆成多个头相当于给每种关系判断分配独立的"频段"，
减少干扰，在相同总参数量下能表达更多、更精细的关系。论文通过受控实验验证了这一理论预测的相变点。
