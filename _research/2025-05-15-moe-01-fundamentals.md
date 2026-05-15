---
title: "MoE 底层原理：为什么大模型可以又大又快"
date: 2025-05-15
level: 3
series: "理解 Mixture of Experts"
series_order: 1
series_total: 3
tags: [MoE, router, expert, 稀疏激活]
summary: "671B 参数的 DeepSeek-V3 推理速度接近 37B 的小模型——因为每个 token 只用 5.5% 的参数。MoE 是怎么做到的？从 router 的一个线性层说起。"
---

> DeepSeek-V3 有 6710 亿参数，但推理时每个 token 只激活 370 亿。Mixtral 8×7B 有 470 亿参数，跑起来像一个 130 亿的模型。秘密就在 Mixture of Experts——让大模型学会"该用哪部分脑子"。

## 一个直觉：图书馆 vs 百科全书

想象两种查资料的方式：

**百科全书模式（Dense 模型）**：你有一本无敌厚的百科全书（比如 700 亿页）。每回答一个问题，不管问题多简单，你都要把整本书从头翻到尾。书越厚，翻一遍就越慢。

**图书馆模式（MoE 模型）**：你有一个巨大的图书馆，有很多书架（比如 8 个），每个书架放着一部分知识。前台有一个聪明的图书管理员。你问一个化学问题，管理员只帮你从化学书架取两本相关的书。图书馆很大（总知识多），但你每次只看两本书（每次计算量小）。

这就是 MoE 的核心经济学：**用"存储成本"换"计算成本"**。模型的知识容量可以非常大，但每次推理只使用一小部分参数。

## 交互式动画：看 Token 如何被分配到 Expert

<iframe src="/assets/moe-animation.html" width="100%" height="560" style="border:1px solid #23232e; border-radius:12px; background:#0a0a0f;" loading="lazy"></iframe>

<p style="color:#6b6b78; font-size:0.85em; text-align:center; margin-top:8px;">↑ 点击画面重新播放 | 观察：token 经过 Router 被分配到不同 Expert，容量满时会被 drop</p>

## Expert 到底是什么？——没你想的那么神秘

一个 Transformer block 里有两个核心组件：Self-Attention 和 FFN（前馈网络）。Attention 负责"词和词之间互相看"，FFN 负责"每个词独立做非线性变换"。

在 MoE 模型中，**Attention 部分完全不变**——所有 token 共享同一个 attention。改变的只有 FFN：原来的一个大 FFN，被替换成**多个小 FFN**（叫做 experts）。

每个 expert 的结构和普通 FFN 完全一样：

```
Expert_i(x) = W2_i · 激活函数(W1_i · x)
```

区别只是**每个 expert 有自己独立的参数**。就像一家连锁店的不同分店——装修一样，但员工不同，服务的客户群也不同。

## Router：整个系统的大脑只有一层

8 个 expert 摆在那里，每个 token 该去哪个？这个决策由 **Router**（路由器，也叫 gating network）做出。

Router 的架构简单到令人惊讶——**就是一个线性层 + softmax**：

```python
router_scores = token_hidden_state × W_gate   # W_gate: [hidden_dim, num_experts]
probabilities = softmax(router_scores)         # 每个 expert 的概率
top_k_experts = top_k(probabilities, k=2)      # 选概率最高的 2 个
```

三行代码。没有隐藏层，没有复杂的神经网络。就是一个矩阵乘法把 token 的表示映射到 N 个分数，softmax 变成概率分布，取 top-k。

为什么这么简单就够用？因为 token 经过前面多层 attention 后，它的 hidden state 已经包含了丰富的语义信息。一个线性投影就能判断"这个 token 需要什么类型的处理"。

## Token 的旅程

一个 token 通过 MoE 层的完整流程：

<svg viewBox="0 0 650 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:20px auto;display:block;">
  <defs><marker id="amoe" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker></defs>
  <rect x="10" y="40" width="70" height="40" rx="6" fill="#1e1e2a" stroke="#60a5fa" stroke-width="1.5"/>
  <text x="45" y="64" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Token x</text>
  <line x1="80" y1="60" x2="115" y2="60" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#amoe)"/>
  <rect x="120" y="35" width="90" height="50" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="165" y="55" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">Router</text>
  <text x="165" y="70" text-anchor="middle" fill="#6b6b78" font-size="7" font-family="system-ui">Wx → softmax</text>
  <line x1="210" y1="48" x2="280" y2="30" stroke="#22d3ee" stroke-width="1.2" marker-end="url(#amoe)"/>
  <line x1="210" y1="72" x2="280" y2="90" stroke="#34d399" stroke-width="1.2" marker-end="url(#amoe)"/>
  <text x="250" y="37" fill="#22d3ee" font-size="7" font-family="system-ui">0.6</text>
  <text x="250" y="97" fill="#34d399" font-size="7" font-family="system-ui">0.3</text>
  <rect x="285" y="15" width="90" height="35" rx="5" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="330" y="36" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">Expert 1</text>
  <rect x="285" y="75" width="90" height="35" rx="5" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="330" y="96" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">Expert 3</text>
  <line x1="375" y1="32" x2="430" y2="55" stroke="#22d3ee" stroke-width="1" marker-end="url(#amoe)"/>
  <line x1="375" y1="92" x2="430" y2="65" stroke="#34d399" stroke-width="1" marker-end="url(#amoe)"/>
  <rect x="435" y="40" width="100" height="40" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="485" y="58" text-anchor="middle" fill="#fbbf24" font-size="8" font-family="system-ui">加权求和</text>
  <text x="485" y="72" text-anchor="middle" fill="#6b6b78" font-size="7" font-family="system-ui">0.6·E1 + 0.3·E3</text>
  <line x1="535" y1="60" x2="575" y2="60" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#amoe)"/>
  <rect x="580" y="40" width="60" height="40" rx="6" fill="#1e1e2a" stroke="#60a5fa" stroke-width="1.5"/>
  <text x="610" y="64" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">输出</text>
</svg>

1. Token 的 hidden state 进入 Router
2. Router 输出每个 expert 的概率（如 [0.6, 0.05, 0.3, 0.05]）
3. 选 top-2：Expert 1（概率 0.6）和 Expert 3（概率 0.3）
4. Token 分别送入这两个 expert 处理
5. 两个 expert 的输出按概率加权求和：`output = 0.6 × E1(x) + 0.3 × E3(x)`

就这样。从外面看，MoE 层的输入输出接口和普通 FFN 完全一样——一个向量进去，一个向量出来。区别只在于内部走了哪条路。

## 实际的数字

| 模型 | 总参数 | 每 token 激活 | Expert 数 | Top-K | 效率比 |
|------|--------|--------------|-----------|-------|--------|
| Mixtral 8×7B | 47B | 13B | 8 | 2 | 3.6× |
| DeepSeek-V2 | 236B | 21B | 160+2 shared | 6+2 | 11× |
| DeepSeek-V3 | 671B | 37B | 256+1 shared | 8+1 | 18× |

DeepSeek-V3 的效率比是惊人的 18×——拥有 6710 亿参数的知识容量，但计算量只相当于一个 370 亿参数的 dense 模型。

下一篇我们来看让这套系统稳定运行的关键难题：load balancing（负载均衡）、token dropping（容量溢出）和训练不稳定性——以及各种精妙的解决方案。
