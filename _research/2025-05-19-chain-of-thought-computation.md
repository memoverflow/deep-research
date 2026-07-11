---
title: "Chain-of-Thought：为什么「说出思考过程」能让模型变聪明"
date: 2025-05-19
level: 3
series: "LLM 原理深度解析"
series_order: 8
series_total: 37
tags: [chain-of-thought, computational-complexity, transformer, reasoning]
summary: "从计算复杂性理论视角解释 Chain-of-Thought 为什么有效：它本质上突破了 Transformer 固定深度的串行计算瓶颈，把一个「只能算一步」的机器变成了「能算无数步」的机器。"
---

# Chain-of-Thought：为什么「说出思考过程」能让模型变聪明

> 让 AI 写出中间步骤，准确率就能从 18% 跳到 57%。这不是玄学——背后有深刻的计算理论解释。

## 一个你一定经历过的场景

想象你参加一场心算比赛。主持人给你一道题：「23 × 47 等于多少？」

如果规则是「盯着题目看 3 秒，然后直接报答案」——大多数人会答错。

但如果允许你拿一张草稿纸，把 23 × 7 = 161、23 × 40 = 920、161 + 920 = 1081 一步步写下来——几乎所有人都能答对。

草稿纸上的那些中间步骤并没有给你「新的知识」。你本来就知道怎么做乘法。草稿纸做的事情是：给你提供了额外的「计算空间」和「记忆空间」，让你可以把一个复杂问题拆成多个简单步骤，一步步完成。

2022 年，Google 的 Jason Wei 等人发现了一个惊人的现象：大语言模型也是一样的。只要在提示词里加一句「请一步步思考（Let's think step by step）」，或者给几个带有推理过程的示例，模型在数学、逻辑、符号推理等任务上的表现就会大幅提升。

这个技术被称为 **Chain-of-Thought (CoT) Prompting**——链式思维提示。

但问题是：*为什么它有效？* 模型又不是人，它没有「工作记忆不够用」的限制……还是说，其实有？

答案出人意料：Transformer 的确有一个类似「工作记忆不够」的根本限制——而且这个限制可以用严格的数学来证明。

## Transformer 的「先天残疾」：固定深度

要理解 CoT 为什么有效，我们需要先理解 Transformer 的一个根本架构特征。

### 每次前向传播 = 固定步数的计算

一个 Transformer 模型有固定的层数——比如 GPT-4 大约有 120 层。当模型处理输入并生成下一个 token 时，信息从第 1 层流到第 120 层，经过恰好 120 步顺序计算。

这里的关键词是「恰好」。不管问题有多复杂——无论是「1+1 等于几」还是「证明费马大定理的某个引理」——模型在生成每个 token 时，都只做了相同步数的计算。

你可以把 Transformer 想象成一条固定长度的流水线。原料（输入信息）从一端进入，经过固定数量的加工站（层），从另一端出来变成产品（输出 token）。流水线可以很宽——每个加工站里有很多工人并行工作（隐藏维度大）——但流水线的**长度是固定的**。

<svg viewBox="0 0 700 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Input -->
  <rect x="20" y="60" width="90" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="65" y="92" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">输入 x</text>
  <!-- Arrow -->
  <line x1="110" y1="87" x2="150" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Layer 1 -->
  <rect x="155" y="60" width="90" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="200" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 1</text>
  <text x="200" y="102" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">(并行计算)</text>
  <!-- Arrow -->
  <line x1="245" y1="87" x2="285" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Layer 2 -->
  <rect x="290" y="60" width="90" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="335" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 2</text>
  <text x="335" y="102" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">(并行计算)</text>
  <!-- Arrow -->
  <line x1="380" y1="87" x2="420" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Dots -->
  <text x="445" y="92" text-anchor="middle" fill="#94a3b8" font-size="18" font-family="system-ui">···</text>
  <!-- Arrow -->
  <line x1="470" y1="87" x2="500" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Layer T -->
  <rect x="505" y="60" width="90" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="550" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer T</text>
  <text x="550" y="102" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">(并行计算)</text>
  <!-- Arrow -->
  <line x1="595" y1="87" x2="625" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Output -->
  <rect x="630" y="60" width="55" height="55" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="657" y="92" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">y</text>
  <!-- Labels -->
  <text x="350" y="145" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">固定 T 步串行计算（深度固定）</text>
  <text x="350" y="165" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">每一步内可以大量并行（宽度大）</text>
</svg>

### 这意味着什么？

深度（层数）决定了模型能做多少步**顺序计算**。宽度（隐藏维度）决定了每一步里能做多少**并行计算**。

很多问题是「天然需要顺序计算」的。比如计算 $f(f(f(...f(x)...)))$——把一个函数迭代 100 次，你必须先算第 1 次的结果才能算第 2 次，先有第 2 次才能算第 3 次。这种计算无法并行化，不管你给多少并行资源都没用。

如果你的模型只有 120 层，它在一次前向传播中最多只能做 120 步顺序计算。当问题需要的顺序步骤超过这个数——比如两个 100 位数相加需要 100 步进位——模型就不可能在一次前向传播中完成。

## 用电路复杂性说清楚这件事

计算机科学家有一套精确的语言来描述这种限制。这就是**电路复杂性理论（Circuit Complexity Theory）**。

### TC⁰：Transformer 一次能算的极限

2023 年，Merrill 和 Sabharwal 证明了一个重要定理：一个固定深度、有限精度、多项式大小的 Transformer，在一次前向传播中能解决的问题，恰好对应电路复杂性中的 **TC⁰** 类。

TC⁰ 是什么？直觉上，它是「常数深度电路能解决的所有问题」——你可以理解为「只需要固定步数的顺序计算就能解决的问题」。它包含很多东西——加法、固定精度乘法、排序——但它**不包含**所有多项式时间问题。

一个经典的不在 TC⁰ 中的问题是：判断一个布尔电路的输出（当电路深度不固定时）。这类问题需要的顺序步骤随输入增长而增长。

Li 等人（2024）进一步收紧了这个界：如果 Transformer 的精度是常数比特（而非多项式比特），那它一次前向传播只能解决 **AC⁰** 类问题——这甚至比 TC⁰ 还要弱，连计算两个 n 位数的乘法都做不到。

翻译成人话：**一个固定深度的 Transformer，不管多宽、注意力多强大，在「一步之内」能做到的事情是有数学上限的。**

### 但有了 CoT，情况完全不同

同一篇论文的核心定理告诉我们：如果允许 Transformer 生成 $T$ 步中间 token（即 Chain-of-Thought），那它可以解决任何大小为 $T$ 的布尔电路能解决的问题。

也就是说：
- 没有 CoT → 能力被锁定在 TC⁰（固定深度）
- 有 T 步 CoT → 能力扩展到 SIZE(T)（深度可以随 T 增长）
- T 无界 → **图灵完备**（理论上能解决任何可计算问题）

这是一个极其优美的理论结果。它告诉我们 CoT 不是什么「心理学小技巧」或「提示工程的经验法则」——它是在**突破计算复杂性的天花板**。

## CoT 到底在做什么：三个视角

### 视角一：增加串行计算深度

这是最核心的视角。

没有 CoT 时，模型对每个 token 做一次前向传播（T 层），能做 T 步顺序计算。

有 CoT 时，模型生成 K 个中间 token，每个都经过 T 层前向传播。而且关键是：生成第 k+1 个 token 时，模型可以通过注意力机制「看到」前面所有 k 个 token 的内容。

这意味着有效的串行计算深度从 T 变成了 K × T。

如果你的模型有 40 层，直接回答只有 40 步计算深度。但如果它先生成 100 个推理 token，有效深度就变成了 4000 步。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#34d399"/>
    </marker>
  </defs>
  <!-- Title: Direct -->
  <text x="350" y="20" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">直接回答 vs Chain-of-Thought</text>
  <!-- Direct prediction row -->
  <text x="20" y="65" fill="#94a3b8" font-size="11" font-family="system-ui">直接回答:</text>
  <rect x="100" y="45" width="70" height="40" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="135" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">输入 x</text>
  <line x1="170" y1="65" x2="200" y2="65" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="205" y="45" width="130" height="40" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="270" y="62" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">T 层前向传播</text>
  <text x="270" y="77" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">深度 = T</text>
  <line x1="335" y1="65" x2="365" y2="65" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="370" y="45" width="60" height="40" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="400" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">y ❌</text>
  <!-- CoT row -->
  <text x="20" y="160" fill="#94a3b8" font-size="11" font-family="system-ui">CoT:</text>
  <rect x="50" y="130" width="55" height="40" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="77" y="155" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">输入 x</text>
  <line x1="105" y1="150" x2="125" y2="150" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <!-- Step 1 -->
  <rect x="130" y="130" width="80" height="40" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="170" y="148" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">前向传播</text>
  <text x="170" y="162" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">→ z₁</text>
  <line x1="210" y1="150" x2="230" y2="150" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <!-- Step 2 -->
  <rect x="235" y="130" width="80" height="40" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="275" y="148" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">前向传播</text>
  <text x="275" y="162" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">→ z₂</text>
  <line x1="315" y1="150" x2="335" y2="150" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <!-- dots -->
  <text x="355" y="155" text-anchor="middle" fill="#94a3b8" font-size="14" font-family="system-ui">···</text>
  <line x1="370" y1="150" x2="390" y2="150" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <!-- Step K -->
  <rect x="395" y="130" width="80" height="40" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="435" y="148" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">前向传播</text>
  <text x="435" y="162" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">→ zₖ</text>
  <line x1="475" y1="150" x2="495" y2="150" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <!-- Final -->
  <rect x="500" y="130" width="80" height="40" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="540" y="148" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">前向传播</text>
  <text x="540" y="162" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="system-ui">→ y ✓</text>
  <!-- Attention arrows (curved) going back -->
  <path d="M 255 170 C 255 200 170 200 170 170" fill="none" stroke="#a78bfa" stroke-width="1" stroke-dasharray="3,3"/>
  <path d="M 435 170 C 435 210 170 210 170 170" fill="none" stroke="#a78bfa" stroke-width="1" stroke-dasharray="3,3"/>
  <path d="M 435 170 C 435 195 275 195 275 170" fill="none" stroke="#a78bfa" stroke-width="1" stroke-dasharray="3,3"/>
  <!-- Labels -->
  <text x="350" y="230" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">有效串行深度 = K × T</text>
  <text x="350" y="250" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">每步都能通过注意力「回看」所有之前的中间结果</text>
  <text x="350" y="270" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">z₁, z₂, ... 就是模型的「草稿纸」</text>
</svg>

### 视角二：外部工作记忆（草稿纸）

Transformer 在一次前向传播中的「记忆」全部存在隐藏状态里——一个固定大小的向量（比如 4096 维）。如果问题需要追踪的中间信息量超过了这个向量能编码的范围，模型就会丢失信息。

但当模型把中间结果写成 token 时——比如写下「23 × 7 = 161」——这个「161」就被显式地存储在了上下文窗口里。后续生成 token 时，模型可以通过注意力机制精确地找到这个「161」并使用它。

这就像给模型一张无限大的草稿纸：每生成一个 token，模型就在草稿纸上写下一点东西，后面可以随时「回头看」。

Nye 等人在 2021 年的「Scratchpad」论文中验证了这一点：在 8 位数加法任务中，不用草稿纸的模型准确率接近 0%，用了草稿纸后准确率超过 95%。原因很清楚：8 位加法需要 8 步进位，每步的进位结果需要传递给下一步。没有草稿纸，这些中间结果只能靠隐藏状态「记住」，而有限精度的隐藏状态装不下这么多信息。

### 视角三：将困难问题分解为简单子问题

从概率的角度看，CoT 改变了模型需要计算的条件概率结构。

直接预测要求模型计算 $P(y|x)$——直接从输入跳到输出。如果中间需要复杂推理，这个概率分布会很「模糊」（熵很高），模型难以给出准确答案。

有了 CoT，模型计算的是：

$$P(z_1|x) \cdot P(z_2|x,z_1) \cdot ... \cdot P(y|x,z_1,...,z_n)$$

每一步都是一个**相对简单**的条件概率。比如「已知 23 × 7，下一步应该写 161」对模型来说是很容易的。最后的 $P(y|x, z_1, ..., z_n)$ 也变得很简单——因为所有中间结果都已经算好了，模型只需要「读取」最终答案。

这就好比：你问一个人「把一篇英文文章翻译成中文」很难，但如果你先告诉他每个句子的意思，再问「把这些意思连起来」就容易多了。

## 数学上有多大的提升？

### 严格的表达能力分层

Li 等人（2024）的理论结果可以总结为一个清晰的层级：

| 设置 | 计算能力 | 直觉 |
|------|----------|------|
| 常数深度 Transformer，常数比特精度，无 CoT | AC⁰ | 连乘法都做不了 |
| 常数深度 Transformer，poly(n) 比特精度，无 CoT | TC⁰ | 能做加减乘除、排序，但不能做任意长的顺序推理 |
| 常数深度 Transformer + T 步 CoT | SIZE(T) | 能解决所有大小为 T 的电路能解决的问题 |
| 常数深度 Transformer + 无限 CoT | 图灵完备 | 理论上无所不能 |

Feng 等人（2023）更精确地指出：对于一个能被图灵机在 $t$ 步内解决、使用 $s$ 格磁带的问题，Transformer + CoT 可以用 $O(t)$ 个推理 token 和 $O(s)$ 的上下文长度来解决它。

换句话说，CoT 把 Transformer 从一个「固定深度电路」变成了一个「通用计算机」。

### 从 18% 到 57%：不只是技巧

Wei 等人（2022）在 GSM8K（小学数学题）上的经典结果：

- 175B 参数模型直接回答：**18%** 准确率
- 175B 参数模型 + CoT：**57%** 准确率
- CoT + 自一致性（多次采样投票）：**74%**

这个从 18% 到 57% 的跳跃不是渐进式改进——它反映的是计算能力的质变。那些小学数学题需要 3-8 步连续推理，每步的结果都依赖前一步。这恰好是「需要深度超过固定层数」的典型问题。

## 什么时候 CoT 没用？

理解了原理，我们也就知道了 CoT 的**边界**在哪里。

### 纯记忆问题

「法国的首都是哪里？」——答案直接存储在模型的权重里，一步就能取出。CoT 不仅没有帮助，反而可能引入不必要的推理 token，增加出错概率。

### 天然可并行的问题

「判断这个列表中每个数字是否为偶数」——每个判断相互独立，不需要「先算 A 才能算 B」。Transformer 一次前向传播就能并行处理所有位置。CoT 会把并行检查变成串行的，白白增加计算量。

### 模型太小

CoT 的前提是模型有能力**分解问题**（知道该怎么一步步拆）并**正确执行每一步**。如果模型太小，连「23 × 7 = ?」这种子步骤都做不对，那再多的中间步骤也没用。

这就是为什么 Wei 等人发现 CoT 的效果和模型大小高度相关：
- 10B 以下：几乎没有提升
- 10-100B：中等提升  
- 100B 以上：巨大提升

小模型缺乏的不是计算深度，而是「知道怎么分解」的能力。CoT 给的是「更多计算空间」，但如果你不知道该用这些空间算什么，空间再多也白搭。

### 错误传播的代价

CoT 的另一个风险是：中间步骤一旦出错，错误会向后传播。如果每一步有 $p$ 的概率正确，$n$ 步之后总体正确率最差情况下降到 $p^n$。

这就是为什么 **Self-Consistency**（自一致性）方法有效：让模型生成多条不同的推理链，然后投票选最常见的答案。这相当于给推理过程加上了「纠错码」。

## 从 CoT 到 o1/o3：思考越久越聪明

理解了 CoT 的计算复杂性视角，我们可以更好地理解 2024-2025 年的一个重要趋势：**test-time compute scaling**（推理时计算量扩展）。

OpenAI 的 o1、o3 系列模型，DeepSeek-R1，以及其他「推理模型」，本质上都是在做同一件事：**让模型在回答之前进行更长时间的「思考」**，生成更多的推理 token。

从复杂性理论的角度看，这等价于增加 CoT 的长度 K，从而增加有效计算深度 K × T。K 越大，模型能解决的问题越复杂。

这也是为什么这些模型在数学竞赛、编程、科学推理等需要深度顺序思考的任务上表现突出——它们用更多的推理时间（更多 token）换取了更强的计算能力。

## 总结：CoT 的本质

Chain-of-Thought 不是提示工程的小技巧，不是让模型「模仿人类思考」的心理学把戏。

它的本质是一个**计算架构层面的突破**：

1. **突破深度限制**：把固定 T 层的串行计算扩展到 K × T 步
2. **提供外部记忆**：上下文窗口充当了可寻址的草稿纸
3. **简化条件概率**：把一个难的 $P(y|x)$ 分解为多个简单的条件概率之积
4. **逼近图灵完备**：理论上，足够长的 CoT 让 Transformer 能计算任何可计算函数

下次你看到一个 AI 模型在「自言自语」地写出推理过程时，不要把它当作多余的输出——那些中间 token 的每一个，都是模型在突破自己架构限制的一次挣扎。它们是模型的「草稿纸」、「工作记忆」和「额外计算时间」。

而这个理论也指向了一个深刻的问题：既然更多的「思考时间」= 更强的能力，那能力的上限到底在哪里？当推理 token 的长度可以任意增长时，Transformer 的能力是否就没有上限了？

理论上是的——图灵完备意味着无所不能。但现实中，错误传播、注意力窗口限制、以及「学会正确的推理策略」本身的难度，都在给这个美好的理论打折扣。如何在实践中缩小理论和现实的差距，是当前 AI 推理研究最核心的问题之一。

---

**参考文献：**

- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
- Li, Liu, Zhou, Ma (2024). "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems." arXiv:2402.12875.
- Feng et al. (2023). "Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective." NeurIPS 2023.
- Merrill & Sabharwal (2023). "The Parallelism Tradeoff: Limitations of Log-Precision Transformers." TACL.
- Nye et al. (2021). "Show Your Work: Scratchpads for Intermediate Computation with Language Models."
- Merrill, Sabharwal, Smith (2024). "The Expressive Power of Transformers with Chain of Thought." ICLR 2024.
