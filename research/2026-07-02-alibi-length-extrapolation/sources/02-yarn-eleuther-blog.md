---
url: https://blog.eleuther.ai/yarn/
title: "Extending the RoPE"
type: technical_blog
authors: Honglu Fan, Bowen Peng, Jeffrey Quesnelle (EleutherAI)
year: 2023
accessed: 2026-07-02
quality: 5
relevance: core
---

这篇 EleutherAI 官方博客系统梳理了 RoPE 上下文扩展方法的演化脉络：Position Interpolation → NTK-aware → NTK-by-parts → YaRN → Dynamic Scaling。

## RoPE 基础
RoPE 让 f_q, f_k 满足 f_q(x_m,m)^T f_k(x_n,n) = g(x_m,x_n,m-n)，即注意力分数只依赖相对距离 m-n。具体实现是把 query/key 向量按不同频率 θ_d = b^(-2d/|D|) 旋转，b 通常取 10000。

## Position Interpolation (PI, kaiokendev + Chen et al. 2023)
不重新训练更长序列，而是把新的位置索引压缩进原窗口：g(m) = m/s，s = L'/L（新长度/旧长度）。
- 优点：finetune 成本远低于重新预训练。
- 缺点：仍需要约 1-10B tokens 的微调；微调后短序列上的困惑度会略微上升；均匀压缩所有频率，没有针对高低频区别处理。

## "NTK-aware" Interpolation
基于 Neural Tangent Kernel 理论：深度网络在输入维度低、缺乏高频分量时难以学习高频信息。PI 均匀压缩所有频率会破坏高频信息的学习能力。NTK-aware 的想法是：改变旋转的 base b 而不是均匀缩放位置索引，让高频维度保持接近原始颗粒度（外推），低频维度被压缩（插值）。这就是"高频外推、低频内插"的核心直觉。该方法最初来自 Reddit 社区帖子（2023年5月），而非正式论文。

## "NTK-by-parts" Interpolation
NTK-aware 简单改 base 仍不够精细，NTK-by-parts 把 RoPE 各维度的旋转频率按"波长"分组：波长远小于原始训练长度的维度（高频，转得快）完全不缩放（外推）；波长远大于训练长度的维度（低频，转得慢）完全按比例缩放（插值）；中间维度用一个斜坡函数(ramp function)平滑过渡，由超参数 α, β 决定过渡区间起止点。

## YaRN
YaRN = NTK-by-parts 插值 + 一个额外的"温度缩放"(temperature scaling) 技巧。
观察：在 softmax(q^Tk / (t·√|D|)) 中引入温度 t 可以进一步降低扩展后的困惑度。EleutherAI 通过实验发现对 LLaMA 2 系列（7B/13B/70B）都适用的经验公式：
  √(1/t) = 0.1·ln(s) + 1
其中 s 是缩放因子（新长度/旧长度）。这个公式对不同模型规模都近似适用，说明它捕捉到某种通用规律，而不是过拟合到单一模型。对于其他模型（如 YaRN-Mistral-7B-128k），拟合出的系数是 a=0.07, b=1.0，形式相同但参数不同——说明温度缩放需要针对模型家族微调超参，但函数形式(对数关系)是稳定的。

## Dynamic Scaling
真实推理场景中序列长度是逐 token 递增的（自回归生成）。如果把缩放因子 s 固定死，短序列时性能会有一个"平白无故"的下降，长度刚超过训练长度时又会有断崖式退化。解决方法（同样源自 Reddit 帖子）：让 s 随当前实际序列长度 l' 动态调整：
  s = l'/L 如果 l'/L > 1，否则 s = 1
这个技巧可以套用在 PI、NTK-by-parts、YaRN 上，分别得到 dynamic PI / dynamic NTK / dynamic YaRN。作者特别指出："dynamic NTK" 在完全不微调的情况下（L'=L）效果出奇地好。

## 意义
这条技术脉络说明 RoPE 的上下文扩展本质上是一个"频率工程"问题：模型在预训练阶段学到的旋转频率分布是针对某个长度校准的，扩展长度就是要在不破坏这个校准的前提下，让位置编码继续"看起来正常"。YaRN 是目前该技术线路的集大成者，被广泛用于 Llama 2/3、Mistral 等开源模型的长上下文微调。
