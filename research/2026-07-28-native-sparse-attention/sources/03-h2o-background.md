---
url: https://arxiv.org/abs/2306.14048
title: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
type: arxiv_paper
authors: Zhenyu Zhang, et al.
year: 2023
accessed: 2026-07-28
quality: 4
relevance: background
---

## 关键内容
H2O 是较早的、推理阶段（inference-only）的 KV cache 淘汰（eviction）方案，属于 NSA 论文中批评的"事后稀疏"（post-hoc sparsity）代表方法之一。核心观察：并不是所有历史 token 对未来生成都同等重要，一小部分"重磅"（Heavy Hitter）token 累积了大部分的 attention mass。H2O 提出动态保留"最近 token + 重磅 token"的平衡策略，只需保留约 20% 的 KV cache 就能维持性能。

但正是这类方法暴露的问题（只在解码阶段稀疏、需要在预填充阶段先算出完整的 attention map 来判断谁是"重磅"、且是事后施加在预训练好的全注意力模型上）促使 NSA 团队提出"原生"可训练稀疏注意力的必要性——如果稀疏模式是通过训练学出来的、而不是靠启发式规则在推理时强加的，模型可以更好地适应稀疏结构而不损失能力。

## 用途
用作 NSA/MoBA 文章的背景对照：说明"先密后疏"（先用全注意力训练、推理时再剪枝）这条路线的局限。
