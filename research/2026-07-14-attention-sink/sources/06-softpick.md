---
url: https://arxiv.org/abs/2504.20966
title: "Softpick: No Attention Sink, No Massive Activations with Rectified Softmax"
type: arxiv_paper
authors: (2025)
year: 2025
accessed: 2026-07-14
quality: 4
relevance: supporting
---

从"如果我们改掉 softmax，会怎样"的角度反向验证了 attention sink 的成因——softmax 强制"和为 1"的归一化性质是罪魁祸首之一。

核心内容：
- Softpick 是一种整流（rectified）、不要求"和为 1"的 softmax 替代函数，可直接替换 Transformer 注意力中的 softmax。
- 实验结果（340M 参数模型）：
  - Softmax 基线：约 63% 的注意力头/层出现明显的 sink 现象。
  - Softpick 模型：sink 出现率降到 0%。
  - Massive activations 显著减少：隐藏状态的 kurtosis（尖峰程度，用来衡量"是否存在极端离群值"）从 33,510 降到 340，降低了约 100 倍。
- 该函数是 drop-in replacement，同时提供了 FlashAttention-2 kernel 的修改版实现（说明工程上可落地，不只是理论玩具）。

意义：这是最直接的"因果实验"——如果去掉 softmax 强制归一化到 1 的约束，attention sink 和 massive activations 会同时几乎消失。这有力支持了"attention sink 源于 softmax 的数学约束，而不是数据或任务本身固有的需求"这一假说，同时呼应了 StreamingLLM 论文里提到的 SoftMax-off-by-one 思路。
