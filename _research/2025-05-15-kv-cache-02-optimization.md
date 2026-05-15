---
title: "KV Cache 瘦身术：从 GQA 到 PagedAttention"
date: 2025-05-15
level: 3
series: "理解 KV Cache 与 Prompt Caching"
series_order: 2
series_total: 3
tags: [GQA, MLA, PagedAttention, vLLM, 量化, 压缩]
summary: "KV Cache 太大装不下？这篇讲所有让它变小的方法：共享 KV 头、低秩压缩、量化到 4 位、像操作系统一样管理内存、以及聪明地选择性遗忘。"
---

> 上一篇我们知道了 KV Cache 的核心矛盾：不存不行（O(n²)太慢），全存也不行（内存爆炸）。这篇来看解决方案——从"少存一点"到"存得更聪明"。

## 策略一：共享 KV 头——MQA / GQA / MLA

### 问题：32 个头真的需要 32 套独立的 KV 吗？

标准 Multi-Head Attention（MHA）中，每个注意力头有自己独立的 Query、Key、Value 投影。32 个头 = 32 套 K 和 V 存储在 cache 中。

但仔细想想——不同头的 Key 和 Value 之间**高度相关**。它们都在表示同一段文本的信息，只是从不同"角度"看。这种冗余可以利用。

### 从 MQA 到 GQA 到 MLA：越来越精细的共享

<svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- MHA -->
  <text x="90" y="20" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui" font-weight="bold">MHA</text>
  <text x="90" y="35" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">每头独立KV</text>
  <rect x="30" y="45" width="18" height="80" rx="3" fill="#22d3ee" opacity="0.6"/><text x="39" y="90" text-anchor="middle" fill="#fff" font-size="7">K₁</text>
  <rect x="50" y="45" width="18" height="80" rx="3" fill="#22d3ee" opacity="0.6"/><text x="59" y="90" text-anchor="middle" fill="#fff" font-size="7">K₂</text>
  <rect x="70" y="45" width="18" height="80" rx="3" fill="#22d3ee" opacity="0.6"/><text x="79" y="90" text-anchor="middle" fill="#fff" font-size="7">K₃</text>
  <rect x="90" y="45" width="18" height="80" rx="3" fill="#22d3ee" opacity="0.6"/><text x="99" y="90" text-anchor="middle" fill="#fff" font-size="7">K₄</text>
  <rect x="110" y="45" width="18" height="80" rx="3" fill="#fbbf24" opacity="0.6"/><text x="119" y="90" text-anchor="middle" fill="#fff" font-size="7">V₁</text>
  <rect x="130" y="45" width="18" height="80" rx="3" fill="#fbbf24" opacity="0.6"/><text x="139" y="90" text-anchor="middle" fill="#fff" font-size="7">V₂</text>
  <text x="90" y="145" text-anchor="middle" fill="#fb7185" font-size="9" font-family="system-ui">Cache: 8 向量/token</text>
  <text x="90" y="160" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">压缩: 1×</text>
  <!-- Arrow -->
  <line x1="160" y1="85" x2="190" y2="85" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr2)"/>
  <!-- GQA -->
  <text x="270" y="20" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui" font-weight="bold">GQA</text>
  <text x="270" y="35" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">分组共享KV</text>
  <rect x="220" y="45" width="40" height="80" rx="3" fill="#22d3ee" opacity="0.7"/><text x="240" y="90" text-anchor="middle" fill="#fff" font-size="8">K组1</text>
  <rect x="265" y="45" width="40" height="80" rx="3" fill="#22d3ee" opacity="0.7"/><text x="285" y="90" text-anchor="middle" fill="#fff" font-size="8">K组2</text>
  <text x="240" y="135" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">Q₁Q₂→组1</text>
  <text x="285" y="135" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">Q₃Q₄→组2</text>
  <text x="270" y="155" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">Cache: 4 向量/token</text>
  <text x="270" y="170" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">压缩: 2×</text>
  <!-- Arrow -->
  <line x1="320" y1="85" x2="360" y2="85" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr2)"/>
  <!-- MQA -->
  <text x="430" y="20" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui" font-weight="bold">MQA</text>
  <text x="430" y="35" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">全部共享1组KV</text>
  <rect x="405" y="45" width="50" height="80" rx="3" fill="#22d3ee" opacity="0.8"/><text x="430" y="85" text-anchor="middle" fill="#fff" font-size="9">K</text>
  <text x="430" y="145" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">Cache: 2 向量/token</text>
  <text x="430" y="160" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">压缩: 4×</text>
  <!-- Arrow -->
  <line x1="470" y1="85" x2="510" y2="85" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr2)"/>
  <!-- MLA -->
  <text x="600" y="20" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui" font-weight="bold">MLA</text>
  <text x="600" y="35" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">低秩压缩潜向量</text>
  <rect x="575" y="50" width="50" height="60" rx="3" fill="#a78bfa" opacity="0.7"/><text x="600" y="80" text-anchor="middle" fill="#fff" font-size="8">c_t</text>
  <text x="600" y="120" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">d_c ≪ H×d_h</text>
  <text x="600" y="145" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">Cache: ~0.5 向量</text>
  <text x="600" y="160" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">压缩: 16-32×</text>
</svg>

**MQA (2019)**：所有 query 头共享**一组** KV。压缩最狠（32×），但质量有损——一个 KV 头承载所有信息，成了瓶颈。

**GQA (2023, LLaMA 2 采用)**：把 32 个 query 头分成 8 组，每组共享一个 KV 头。4× 压缩，质量损失几乎可忽略。这是当前最流行的方案。

**MLA (2024, DeepSeek-V2)**：最激进——不存 K 和 V 本身，而是存一个**低秩压缩潜向量** $c_t$，需要时再解压还原。压缩 16-32×，质量反而更好（因为联合优化了压缩方式）。

## 策略二：量化——用更少的位数存

KV Cache 默认用 FP16（16位）。但实验发现，attention 对 K/V 的数值精度并不那么敏感。

- **INT8 量化**：每个值从 2 字节降到 1 字节 → **2× 内存减半**，质量损失 < 0.5%
- **INT4 量化**：4 位存储 → **4× 压缩**，需要更精细的量化策略
- **KIVI (2024)**：发现 Key 和 Value 有不同的量化特性（Key 按通道有异常值，Value 按 token 有异常值），针对性地用不同策略量化到 **2 位**，质量几乎无损

量化可以和 GQA/MLA **叠加**：GQA (4×) + INT4 (4×) = **16× 总压缩**。

## 策略三：PagedAttention——像操作系统管内存

### 碎片化才是真正的敌人

传统系统给每个请求预分配一大块连续内存（准备好最大可能长度）。但大部分请求用不了那么多——就像停车场全画大车位，结果来的都是小车。

研究表明 **60-80% 的 KV Cache 内存被碎片化浪费**。

### PagedAttention 的操作系统思维

vLLM 的 PagedAttention 直接借鉴了操作系统的虚拟内存管理：

<svg viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <!-- Traditional -->
  <text x="160" y="20" text-anchor="middle" fill="#fb7185" font-size="11" font-family="system-ui" font-weight="bold">传统：连续预分配</text>
  <rect x="20" y="30" width="280" height="190" rx="6" fill="#1a1a24" stroke="#3a3a4a" stroke-width="1"/>
  <!-- Seq A -->
  <rect x="35" y="45" width="80" height="30" rx="4" fill="#22d3ee" opacity="0.5"/>
  <rect x="115" y="45" width="140" height="30" rx="4" fill="#3a3a4a" opacity="0.3"/>
  <text x="75" y="64" text-anchor="middle" fill="#fff" font-size="8">Seq A 实际</text>
  <text x="185" y="64" text-anchor="middle" fill="#6b6b78" font-size="8">浪费 (预分配)</text>
  <!-- Seq B -->
  <rect x="35" y="85" width="160" height="30" rx="4" fill="#34d399" opacity="0.5"/>
  <rect x="195" y="85" width="60" height="30" rx="4" fill="#3a3a4a" opacity="0.3"/>
  <text x="115" y="104" text-anchor="middle" fill="#fff" font-size="8">Seq B 实际</text>
  <text x="225" y="104" text-anchor="middle" fill="#6b6b78" font-size="8">浪费</text>
  <!-- Gap -->
  <rect x="35" y="125" width="220" height="20" rx="4" fill="none" stroke="#fb7185" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="145" y="139" text-anchor="middle" fill="#fb7185" font-size="8">外部碎片 (无法利用)</text>
  <!-- Seq C -->
  <rect x="35" y="155" width="40" height="30" rx="4" fill="#fbbf24" opacity="0.5"/>
  <rect x="75" y="155" width="180" height="30" rx="4" fill="#3a3a4a" opacity="0.3"/>
  <text x="55" y="174" text-anchor="middle" fill="#fff" font-size="8">C</text>
  <text x="165" y="174" text-anchor="middle" fill="#6b6b78" font-size="8">巨大浪费</text>
  <text x="160" y="205" text-anchor="middle" fill="#fb7185" font-size="10" font-family="system-ui">利用率: ~20-40%</text>
  
  <!-- PagedAttention -->
  <text x="530" y="20" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui" font-weight="bold">PagedAttention：分页按需分配</text>
  <rect x="390" y="30" width="290" height="190" rx="6" fill="#1a1a24" stroke="#3a3a4a" stroke-width="1"/>
  <!-- Grid of blocks -->
  <rect x="405" y="45" width="32" height="25" rx="3" fill="#22d3ee" opacity="0.6"/><text x="421" y="62" text-anchor="middle" fill="#fff" font-size="7">A</text>
  <rect x="441" y="45" width="32" height="25" rx="3" fill="#34d399" opacity="0.6"/><text x="457" y="62" text-anchor="middle" fill="#fff" font-size="7">B</text>
  <rect x="477" y="45" width="32" height="25" rx="3" fill="#22d3ee" opacity="0.6"/><text x="493" y="62" text-anchor="middle" fill="#fff" font-size="7">A</text>
  <rect x="513" y="45" width="32" height="25" rx="3" fill="#fbbf24" opacity="0.6"/><text x="529" y="62" text-anchor="middle" fill="#fff" font-size="7">C</text>
  <rect x="549" y="45" width="32" height="25" rx="3" fill="#34d399" opacity="0.6"/><text x="565" y="62" text-anchor="middle" fill="#fff" font-size="7">B</text>
  <rect x="585" y="45" width="32" height="25" rx="3" fill="#34d399" opacity="0.6"/><text x="601" y="62" text-anchor="middle" fill="#fff" font-size="7">B</text>
  <rect x="621" y="45" width="32" height="25" rx="3" fill="#22d3ee" opacity="0.6"/><text x="637" y="62" text-anchor="middle" fill="#fff" font-size="7">A</text>
  <rect x="405" y="75" width="32" height="25" rx="3" fill="#fbbf24" opacity="0.6"/><text x="421" y="92" text-anchor="middle" fill="#fff" font-size="7">C</text>
  <rect x="441" y="75" width="32" height="25" rx="3" fill="#34d399" opacity="0.6"/><text x="457" y="92" text-anchor="middle" fill="#fff" font-size="7">B</text>
  <rect x="477" y="75" width="32" height="25" rx="3" fill="#3a3a4a" opacity="0.3"/>
  <rect x="513" y="75" width="32" height="25" rx="3" fill="#3a3a4a" opacity="0.3"/>
  <text x="530" y="120" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">Block Table: A→[0,2,6] B→[1,4,5,8] C→[3,7]</text>
  <text x="530" y="140" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">非连续存储，按需分配，用完释放</text>
  <text x="530" y="165" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">利用率: ~96%+</text>
  <text x="530" y="185" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">吞吐提升 2-4×</text>
</svg>

核心思想：把 KV Cache 切成固定大小的**页（blocks）**，散落在 GPU 内存的任何位置。用一个"页表"记录每个序列的 blocks 在哪。按需分配，用完即释。

效果：内存浪费从 60-80% 降到 **< 4%**，同一张 GPU 可以同时服务 **2-4× 更多请求**。

## 策略四：选择性遗忘——不是每个 token 都值得记住

如果上下文实在太长，即使压缩+分页也装不下呢？一个激进但有效的思路：**只保留重要的 token，扔掉不重要的。**

**H₂O (Heavy Hitter Oracle)**：追踪每个 token 累积获得的 attention 分数。高分的（"重量级选手"）保留，低分的淘汰。只保留 20% 的 KV Cache 就能维持质量。

**StreamingLLM**：发现了一个有趣现象——序列最开头的几个 token（不管内容是什么）会获得异常高的 attention，它们是"注意力水槽"（attention sinks）。如果你把它们删了，模型就崩了。解决方案：永远保留前 4 个 token + 最近 L 个 token 的滑动窗口。可以稳定生成超过**400 万 token** 而内存不增长。

**滑动窗口 (Mistral)**：更简单粗暴——只保留最近 W 个 token 的 KV（W=4096），超过的直接覆盖。通过多层堆叠，有效感受野远大于 W。

## 这些技术可以叠加

一个生产级系统可能同时使用：
- **GQA** (架构层面) → 4× 压缩
- **INT4 量化** (精度层面) → 4× 压缩
- **PagedAttention** (内存管理) → 消除 60-80% 浪费
- **H₂O 淘汰** (容量控制) → 再减 5×

总效果：相比朴素 MHA + FP16 + 连续分配，可以实现 **50-100× 的等效内存节省**。

下一篇我们来看一个更上层的优化：如果很多请求的 prompt 开头都一样（比如系统提示、few-shot 示例），能不能只算一次前缀的 KV Cache，后面所有请求复用？这就是 Prompt Caching——一个能帮你省 90% API 账单的技术。
