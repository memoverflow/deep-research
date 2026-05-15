---
title: "为什么生成每个字都这么贵：KV Cache 的前世今生"
date: 2025-05-15
level: 3
series: "理解 KV Cache 与 Prompt Caching"
series_order: 1
series_total: 3
tags: [KV-cache, inference, transformer, 推理优化]
summary: "大模型生成文本时，为什么一个字一个字地蹦？为什么长对话越来越慢？KV Cache 是让推理可行的核心机制——但它也带来了新的内存困境。"
---

> 你有没有注意过，ChatGPT 回答问题时是一个字一个字往外"蹦"的？这不是为了装酷——而是因为 Transformer 在生成时，每次只能确定一个字。而为了确定这一个字，它需要"看"前面所有的字。KV Cache，就是让这件事不至于慢到无法使用的关键机制。

## 一个字一个字：自回归生成

大语言模型（LLM）生成文本的方式叫**自回归生成**：每次只预测"下一个 token"（大致等于一个字或半个词），然后把预测结果加入上下文，再预测下一个……循环往复，直到生成结束。

生成"今天天气真好"这 6 个字，模型实际做了 6 步：
1. 看完 prompt → 预测"今"
2. 看完 prompt + "今" → 预测"天"
3. 看完 prompt + "今天" → 预测"天"
4. ... 一直到"好"

每一步都需要对**前面所有 token** 做 attention 计算。这就引出了一个荒谬的问题。

## 没有缓存的世界：O(n²) 的噩梦

想象你在写一本书。每写一个新字，你都要**从第一页开始把整本书重新读一遍**。写第 100 个字时读 100 遍前文，写第 1000 个字时读 1000 遍。

这就是没有 KV Cache 时 Transformer 在做的事。

Attention 机制需要每个 token 的 Key 和 Value 向量来参与计算。如果不存储之前算过的结果，那生成第 $t$ 个 token 时就必须重新计算前面 $t-1$ 个 token 的全部 K 和 V——而这些值从上一步到这一步**根本没有变过**。

总工作量 = 1 + 2 + 3 + ... + N = **N²/2**。

对于 4096 个 token 的生成，这意味着约 800 万次重复计算。这不只是慢——是慢到**不可用**。

## KV Cache：算一次，存起来

解决方案极其直觉：**前面 token 的 K 和 V 不会变，那就算一次存起来，以后直接查表。**

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Prefill phase -->
  <rect x="20" y="20" width="300" height="110" rx="10" fill="#1a1a24" stroke="#3a3a4a" stroke-width="1"/>
  <text x="170" y="16" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui" font-weight="bold">Prefill 阶段 (并行)</text>
  <rect x="40" y="45" width="260" height="35" rx="6" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="170" y="67" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">输入 Prompt: [T₁, T₂, T₃, ... Tₙ]</text>
  <rect x="40" y="90" width="260" height="30" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="170" y="109" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">一次性计算所有 K, V → 存入 Cache</text>
  <!-- Arrow -->
  <line x1="320" y1="75" x2="370" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- KV Cache storage -->
  <rect x="380" y="30" width="140" height="100" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="450" y="55" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui" font-weight="bold">KV Cache</text>
  <text x="450" y="75" text-anchor="middle" fill="#9494a0" font-size="10" font-family="system-ui">K₁,V₁ | K₂,V₂ | ... | Kₙ,Vₙ</text>
  <text x="450" y="95" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">× 每层 × 每 KV 头</text>
  <text x="450" y="115" text-anchor="middle" fill="#fb7185" font-size="9" font-family="system-ui">⚠️ 内存线性增长</text>
  <!-- Decode phase -->
  <rect x="20" y="155" width="660" height="110" rx="10" fill="#1a1a24" stroke="#3a3a4a" stroke-width="1"/>
  <text x="350" y="150" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui" font-weight="bold">Decode 阶段 (逐 token 循环)</text>
  <rect x="40" y="180" width="100" height="35" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="90" y="201" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">新 token</text>
  <line x1="140" y1="197" x2="175" y2="197" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="180" y="175" width="140" height="45" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="250" y="195" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">算新 token 的 Q,K,V</text>
  <text x="250" y="210" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">K,V 追加到 Cache</text>
  <line x1="320" y1="197" x2="355" y2="197" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="360" y="175" width="160" height="45" rx="6" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="440" y="195" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">Q_new × 全部 cached K</text>
  <text x="440" y="210" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">→ Attention → 输出</text>
  <line x1="520" y1="197" x2="555" y2="197" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="560" y="180" width="100" height="35" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="610" y="201" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">输出 token</text>
  <!-- Loop arrow -->
  <path d="M 610 215 L 610 250 L 90 250 L 90 215" fill="none" stroke="#6b6b78" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="350" y="245" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">循环直到生成结束</text>
</svg>

有了 KV Cache：
- **Prefill 阶段**：把整个 prompt 一次性过模型（可并行），生成并缓存所有 token 的 K、V
- **Decode 阶段**：每步只算新 token 的 Q、K、V，新的 K、V 追加到 cache，用 Q 去查 cache 中所有 K 做 attention

计算量从 O(N²) 降到 O(N)。代价是——内存。

## 代价：内存爆炸

KV Cache 存储的是每一层、每一个 KV 头、每一个已处理 token 的 Key 和 Value 向量。

公式：`KV Cache = 2 × 层数 × KV头数 × head维度 × 序列长度 × 数据精度`

让我们用实际数字感受一下：

<svg viewBox="0 0 650 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <!-- Background bars -->
  <text x="20" y="25" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">KV Cache 内存 vs 模型权重 (FP16)</text>
  <!-- Model weights reference -->
  <rect x="130" y="45" width="56" height="25" rx="4" fill="#3a3a4a"/>
  <text x="158" y="62" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">模型 14GB</text>
  <!-- LLaMA 7B bars -->
  <text x="20" y="62" fill="#9494a0" font-size="10" font-family="system-ui">7B @ 4K</text>
  <rect x="130" y="48" width="28" height="18" rx="3" fill="#22d3ee" opacity="0.7"/>
  <text x="165" y="60" fill="#22d3ee" font-size="9" font-family="system-ui">2 GB</text>
  
  <text x="20" y="95" fill="#9494a0" font-size="10" font-family="system-ui">7B @ 32K</text>
  <rect x="130" y="82" width="160" height="18" rx="3" fill="#22d3ee" opacity="0.7"/>
  <text x="297" y="94" fill="#22d3ee" font-size="9" font-family="system-ui">16 GB</text>
  
  <text x="20" y="128" fill="#9494a0" font-size="10" font-family="system-ui">7B @ 128K</text>
  <rect x="130" y="115" width="512" height="18" rx="3" fill="#fb7185" opacity="0.7"/>
  <text x="520" y="127" fill="#fb7185" font-size="9" font-family="system-ui">64 GB !</text>
  
  <!-- LLaMA 70B GQA bars -->
  <text x="20" y="162" fill="#9494a0" font-size="10" font-family="system-ui">70B GQA@4K</text>
  <rect x="130" y="149" width="16" height="18" rx="3" fill="#34d399" opacity="0.7"/>
  <text x="152" y="161" fill="#34d399" font-size="9" font-family="system-ui">1.3GB</text>
  
  <text x="20" y="195" fill="#9494a0" font-size="10" font-family="system-ui">70B GQA@128K</text>
  <rect x="130" y="182" width="320" height="18" rx="3" fill="#fbbf24" opacity="0.7"/>
  <text x="457" y="194" fill="#fbbf24" font-size="9" font-family="system-ui">40 GB</text>
  
  <!-- Reference line -->
  <line x1="130" y1="40" x2="130" y2="210" stroke="#3a3a4a" stroke-width="0.5" stroke-dasharray="2,2"/>
</svg>

关键数字：
- **LLaMA 7B @ 128K 上下文**：KV Cache 需要 64 GB——模型权重才 14 GB，cache 是模型的 **4.5 倍**
- **LLaMA 70B (GQA) @ 128K**：约 40 GB——注意 70B 用了 GQA（只有 8 个 KV 头而非 64 个），已经压缩了 8 倍

而一张 A100 GPU 才 80 GB 总显存。装完模型后剩余空间有限，KV Cache 直接决定了：
- 能支持多长的上下文
- 能同时服务多少用户

这就是 KV Cache 的核心矛盾：**没有它，生成慢到不可用（O(n²)）；有了它，内存压力巨大。**

## 批处理的困境

一个用户的 KV Cache 还好说。但如果你要同时服务 32 个用户呢？

每个用户的上下文长度不同（有人问了一句话，有人贴了一整篇论文），每个用户的生成长度也不同。传统做法是为每个用户预分配"可能用到的最大内存"——大部分被浪费。

研究表明，传统系统中 **60-80% 的 KV Cache 内存因碎片化而浪费**。就像一个停车场画了大车位，结果来的全是小车——大量空间闲置。

这些问题催生了一整套 KV Cache 优化技术——从改架构（GQA、MLA）到改内存管理（PagedAttention）到降精度（量化）。下一篇我们逐个拆解。
