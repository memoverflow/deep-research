---
url: https://arxiv.org/abs/2510.06662
title: "The Effect of Attention Head Count on Transformer Approximation"
type: arxiv_paper
authors: Penghao Yu, Haotian Jiang, Zeyu Bao, Ruoxi Yu, Qianxiao Li
year: 2025
accessed: 2026-07-07
quality: 5
relevance: supporting
---

ICLR 2026 接收。研究 attention 头数对 Transformer 逼近能力（approximation power）的影响。提出
"D-retrieval task" 作为衡量框架，并证明：头数足够多时，Transformer 能高效逼近目标函数（参数量随
误差 ε 呈多项式增长）；但头数太少时，所需参数量必须随 1/ε 的指数级增长（首次给出这种非线性、
实际相关场景下的严格下界）。

同时研究单头极端情况：证明单头 attention 若 embedding 维度设为 O(序列长度)，可以完全"记住"输入，
此时近似能力完全由 feed-forward 层承担，attention 本身退化为纯粹的检索/复制机制而非灵活的关系建模。
这从理论上印证了"多头存在的意义之一是避免退化为纯记忆型系统"。
