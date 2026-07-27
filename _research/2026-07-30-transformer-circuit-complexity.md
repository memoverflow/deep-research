---
title: "Transformer 算不出奇偶校验：一个模型架构的数学天花板"
date: 2026-07-30
level: 3
series: "LLM 原理深度解析"
series_order: 56
series_total: 56
tags: [circuit-complexity, TC0, AC0, transformer-theory, chain-of-thought, parallelism]
summary: "从电路复杂性理论出发，解释为什么无论把 Transformer 训练得多大、多久，它都无法保证学会像「数一串 01 里 1 的个数是奇是偶」这样看起来极其简单的任务——除非你允许它多写几步。"
---

# Transformer 算不出奇偶校验：一个模型架构的数学天花板

> 给 GPT-4 一串 200 个 0 和 1，问它 1 的个数是奇数还是偶数——这是小学生都会做的题，但大模型的正确率经常跌到接近瞈猜。这不是训练不够、数据不够，是这个架构本身有一条数学上证明过的天花板。

## 一个奇怪的失败

如果你随便找一个大语言模型，让它做三位数乘法，它大概率会出错——这个大家已经比较熟悉了，网上有很多讨论。但更让人意外的是一个看起来简单得多的任务：给它一串很长的 0/1 字符串，问「这串里 1 的总个数是奇数还是偶数」。

这个任务叫 **PARITY（奇偶校验）**。对人类来说，只要有纸笔，从头数到尾，每数一个 1 就翻转一下「奇/偶」的标记，这件事怎么都不会出错——它甚至不需要理解数字大小、不需要进位、不需要任何复杂运算，只是一个不断翻转的开关。

但如果你去测试主流的 Transformer 模型直接一次性给出答案（不许写中间步骤），它在长序列上的表现会显著下降，往往接近随机瞎猜。这不是个别模型没训练好的偶然现象——2019 年到 2024 年间，一系列理论论文用严格的数学证明告诉我们：**这几乎是必然的**。不是「暂时做不到」，而是「这一类架构在数学上被证明做不到」。

这篇文章要讲的，就是这条数学天花板从哪里来、它到底划定了什么边界，以及为什么让模型"多写几步"——也就是 Chain-of-Thought——恰好能打破这条天花板。

## 为什么要用"电路"来理解 Transformer

要理解这个天花板，我们得先换一个看待 Transformer 的角度：不要把它想成一个"神经网络"，而是把它想成一个"电路"。

这个类比并不是牵强的比喻,而是计算机科学里一个非常正式的研究领域，叫**电路复杂性理论（circuit complexity theory）**。它研究的问题是："如果只用逻辑门（AND、OR、NOT，或者更强大的门）搭建出一个电路，深度（层数）固定、每层宽度可以任意大，这样的电路到底能计算哪些函数？"

为什么这个视角能套到 Transformer 上？因为 Transformer 做一次前向传播的过程，本质上和电路的计算过程惊人地相似：

- Transformer 有**固定的层数**（比如 96 层）——电路有**固定的深度**
- 每一层内部所有的 attention head、所有的神经元都在**同时并行计算**——电路的每一层门也是**同时并行触发**的
- 信息只能**从下往上单向流动**，同层内的不同神经元互相看不到彼此的输出——电路里同一层的门也不能互相通信

<svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="340" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Transformer 前向传播 ≈ 一个固定深度的并行电路</text>
  <!-- input row -->
  <rect x="20" y="150" width="46" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="43" y="174" text-anchor="middle" fill="#94a3b8" font-size="11">x1</text>
  <rect x="76" y="150" width="46" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="99" y="174" text-anchor="middle" fill="#94a3b8" font-size="11">x2</text>
  <rect x="132" y="150" width="46" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="155" y="174" text-anchor="middle" fill="#94a3b8" font-size="11">...</text>
  <rect x="188" y="150" width="46" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="211" y="174" text-anchor="middle" fill="#94a3b8" font-size="11">xn</text>
  <text x="130" y="205" text-anchor="middle" fill="#6e8eff" font-size="11">输入 token（宽度 = n，可以任意宽）</text>

  <!-- layer 1 -->
  <line x1="130" y1="150" x2="130" y2="110" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowA)"/>
  <rect x="40" y="70" width="180" height="40" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="130" y="94" text-anchor="middle" fill="#ededf0" font-size="12">第 1 层：所有 head 并行计算</text>

  <!-- layer 2 -->
  <line x1="130" y1="70" x2="130" y2="30" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowA)"/>
  <rect x="40" y="0" width="180" height="30" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5" opacity="0"/>

  <!-- depth label -->
  <line x1="260" y1="20" x2="260" y2="190" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="300" y="60" fill="#a78bfa" font-size="12">深度 L 是常数</text>
  <text x="300" y="80" fill="#a78bfa" font-size="12">（比如 96 层）</text>
  <text x="300" y="100" fill="#a78bfa" font-size="12">不随输入长度 n 增长</text>

  <!-- output -->
  <rect x="380" y="70" width="120" height="40" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="440" y="94" text-anchor="middle" fill="#ededf0" font-size="12">输出 token</text>
  <line x1="220" y1="90" x2="380" y2="90" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowA)"/>

  <text x="440" y="140" fill="#94a3b8" font-size="11" text-anchor="middle">宽度可以随 n</text>
  <text x="440" y="155" fill="#94a3b8" font-size="11" text-anchor="middle">任意扩大</text>
  <text x="440" y="175" fill="#f87171" font-size="11" text-anchor="middle">但深度不能</text>
</svg>

这一点电路复杂性理论家一眼就能认出来：这就是所谓的**"低深度、高并行"电路**。而这类电路早在 Transformer 出现之前几十年，就已经被数学家研究出了它们能力的上限。

## 门的种类决定了电路能算什么：从 AC⁰ 到 TC⁰

在电路复杂性里，不同类型的"逻辑门"划分出不同层级的能力，就像"工具箱里有什么工具"决定了"你能造出什么东西"。

**最基础的一档：AC⁰。** 这类电路只允许用 AND、OR、NOT 门，而且每个门的"输入根数"（fan-in）可以任意多——一个 OR 门可以同时接收一百万根输入线。这类电路擅长做的事情是"存在性判断"："这一百万个数里有没有至少一个是 1？"（用一个大 OR 门瞬间搞定）、"是不是所有位置都是 1？"（用一个大 AND 门）。

但 AC⁰ 电路有一个已经被严格证明的死穴：**它计算不了 PARITY**。这是电路复杂性理论里一个经典的、影响深远的结果（1980 年代由 Furst-Saxe-Sipser 和 Ajtai 分别独立证明）。直觉上为什么？因为 AND/OR/NOT 门本质上都在做"局部的、单调的"判断——它们能感知"有没有"，但感知不了"数量的奇偶"这种需要把全局信息精确汇总、且对每一位的翻转都极度敏感的性质。你翻转任意一个输入位，PARITY 的答案就必须翻转；但 AC⁰ 电路里任何一个固定深度的门组合，都无法对"任意一位翻转"保持这种全局的、处处敏感的响应——这正是深度不够的代价。

**更强一档：TC⁰。** 这类电路在 AND/OR/NOT 之外，额外允许一种叫**阈值门（threshold gate，也叫 MAJORITY 门）**的部件——它可以同时看一大堆输入，然后判断"1 的个数是不是超过了某个阈值"。有了这种门，PARITY 反而变得很好处理：你可以用阈值门做"计数"，再组合出"奇偶性判断"。TC⁰ 严格地比 AC⁰ 更强大——凡是 AC⁰ 能算的它都能算，还能多算一些 AC⁰ 算不了的东西（比如 PARITY、比如整数加法乘法这类需要"计数进位"的运算）。

那 Transformer 的 self-attention 到底相当于哪一档的电路？答案是关键所在：**Softmax attention 本质上就是一种阈值/加权平均操作**——它对一堆数值做加权求和再归一化，这恰好就是阈值门（Majority/threshold gate）的连续版本。所以，一个用有限精度数字表示、层数固定的 Transformer，可以被一个**常数深度的 TC⁰ 电路**完整地模拟出来。

这个结论最早由 William Merrill 和 Ashish Sabharwal 在 2022 年那篇被广泛引用的论文《The Parallelism Tradeoff: Limitations of Log-Precision Transformers》中给出严格证明：**只要 Transformer 的数值精度是关于输入长度对数级的（这已经是一个相当宽松、几乎覆盖所有实际情况的假设），它就可以被 constant-depth、logspace-uniform 的阈值电路模拟。**

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">电路能力的阶梯</text>

  <rect x="40" y="50" width="150" height="50" rx="8" fill="#1e1e2a" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="115" y="72" text-anchor="middle" fill="#ededf0" font-size="12">AC⁰</text>
  <text x="115" y="90" text-anchor="middle" fill="#94a3b8" font-size="10">AND/OR/NOT</text>

  <line x1="190" y1="75" x2="240" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowB)"/>
  <text x="215" y="65" text-anchor="middle" fill="#6e8eff" font-size="10">加阈值门</text>

  <rect x="245" y="50" width="150" height="50" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="320" y="72" text-anchor="middle" fill="#ededf0" font-size="12">TC⁰</text>
  <text x="320" y="90" text-anchor="middle" fill="#94a3b8" font-size="10">+ MAJORITY 门</text>
  <text x="320" y="115" text-anchor="middle" fill="#34d399" font-size="10">← Transformer 前向传播 在这里</text>

  <line x1="395" y1="75" x2="445" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowB)"/>
  <text x="420" y="65" text-anchor="middle" fill="#6e8eff" font-size="10">深度可变</text>

  <rect x="450" y="50" width="150" height="50" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="525" y="72" text-anchor="middle" fill="#ededf0" font-size="12">NC¹ / P</text>
  <text x="525" y="90" text-anchor="middle" fill="#94a3b8" font-size="10">可解决线性方程组、CFG 识别</text>

  <text x="320" y="150" text-anchor="middle" fill="#f87171" font-size="11">PARITY ∉ AC⁰，但 PARITY ∈ TC⁰</text>
  <text x="320" y="170" text-anchor="middle" fill="#f87171" font-size="11">线性方程求解 / 图连通性 ∉ TC⁰（若 L ≠ P）</text>
</svg>

这句话听起来技术性很强，但它带来的推论极其具体，也极其令人震惊：**如果计算机科学界一个众所周知很可能成立的猜想（L ≠ P，逻辑空间不等于多项式时间）是对的，那么 Transformer 在数学上就永远无法可靠地解出线性方程组，永远无法可靠地判断一个字符串是否属于某个带空产生式的上下文无关语法。**

这不是"当前的 Transformer 训练得不够好"，这是"这一类架构（只做一次前向传播、不写中间步骤）在理论上就够不到那个复杂度"。这正是论文标题里"parallelism tradeoff（并行性代价）"这个词的含义：Transformer 之所以能高度并行、训练飞快，恰恰是因为它牺牲了处理某一类"本质上需要串行计算"的问题的能力——鱼与熊掌不可兼得。

## 硬注意力 vs 饱和注意力：现实版本会更弱还是更强？

如果你去查阅这方面更早的文献，会发现故事其实是分阶段展开的，而且越来越贴近真实模型。

最早，Michael Hahn 在 2020 年那篇经常被引用的论文《Theoretical Limitations of Self-Attention in Neural Sequence Models》里，研究了一种理想化的**硬注意力（hard attention）**——也就是每次 attention 只能把权重全部押在一个位置上（相当于一个 one-hot 的选择，而不是像真实模型那样柔和地分散在多个位置）。他证明了这种硬注意力模型无法处理 PARITY，也无法很好地处理需要层级结构的语言（比如括号是否匹配这类需要"栈"结构的任务）。紧接着 Yiding Hao、Dana Angluin 和 Robert Frank 在 2022 年进一步证明：硬注意力 Transformer 能识别的语言恰好被限制在 **AC⁰** 这一档——比 TC⁰ 还弱一档。

但硬注意力毕竟是一个理想化假设，真实的 Transformer 用的是柔软的 softmax attention，权重是分散的、连续的。这引出一个自然的问题：如果放松这个假设，模型会不会变得更强？

Bhattamishra、Ahuja 和 Goyal 等人在后续工作中研究了一种叫**饱和注意力（saturated attention）**的模型——它是硬注意力的一种推广，更贴近训练之后真实收敛的 attention 模式（softmax 的温度趋近于极限时，权重会集中在得分最高的若干个位置上，这种"极限行为"正是饱和注意力刻画的东西）。结果是：饱和注意力确实比硬注意力更强大，能超出 AC⁰ 的边界；但即便如此，它依然被严格限制在 **TC⁰** 这一档之内，逃不出去。

这几篇论文串起来讲的是同一个渐进收紧的故事：从最理想化的"硬注意力只能到 AC⁰"，到更真实的"饱和注意力/log 精度模型顶多到 TC⁰"——不管怎么放松假设，Transformer 一次前向传播的能力上限都被牢牢锁在了一个相当具体、相当"低"的复杂度等级里，够不到 P（一般计算机程序能算的所有问题）。

## 为什么这件事对现实很重要——不只是数学游戏

看到这里你可能会想：这些都是很抽象的理论构造（"识别形式语言"），跟真实场景中大模型做数学题、写代码、做推理有什么关系？

关系其实非常直接。这些理论结果精确预言了大模型在实际测试中会失败的那一类任务的"形状"——它们几乎全部是**需要把大量步骤严格串行地累积、且每一步都依赖前一步精确结果**的问题：

- **多位数乘法和加法的进位链**——每一位的进位依赖前面所有位的计算结果，这本质上是一条不能打断的串行链条，跟 PARITY 是同一类问题（都需要"计数/进位"能力，恰好卡在 AC⁰ 和 TC⁰ 的分界线附近）
- **图的连通性判断**（这个点和那个点之间是否有路径能连起来）——这是一个经典的、被证明不在 TC⁰ 里（在合理复杂度假设下）的问题
- **判断一个逻辑表达式是否可满足**（3-SAT 这类问题）——这些是 P 完全或更难的问题，同样超出边界
- **精确地模拟一台图灵机/有限状态机执行足够多步**——步数一旦超过模型层数所能表达的深度，模型就会丢失精确的状态追踪能力

2023 年的著名论文《Faith and Fate: Limits of Transformers on Compositionality》用大量实证研究反复验证了这个预言：GPT-4 在三位数乘法上只有 59% 的准确率，而且随着数字位数增加，准确率会急剧下降——这正是"进位链越长、串行深度需求越大、越接近甚至超出 TC⁰ 天花板"的具体体现。这不是训练数据不够多、模型不够大能解决的问题，因为**问题的根源在于架构的深度是固定的，而某些问题所需要的串行计算步数会随着输入规模增长而线性甚至更快地增长**——你不可能用一个"深度恒定"的东西去装下一个"深度随输入增长"的问题，除非你想办法让它的"有效深度"也跟着变。

## 那 Chain-of-Thought 到底改变了什么？

现在我们可以回到文章开头那个悬念了：为什么让模型"多写几步"就能让它算出 PARITY、算出更长的乘法？

答案其实非常直白，一旦看懂电路视角就会觉得理所当然：**Chain-of-Thought 本质上是在把"空间上的深度不足"换成"时间上的深度"。**

回想一下电路图：一次不写中间步骤的前向传播，相当于信息只流过固定的 L 层——深度锁死在 L。但如果允许模型生成一个中间 token，再把这个 token 重新喂回输入、进行第二次前向传播，相当于额外叠加了一层"深度 L"的电路，两次前向传播首尾相连；生成 T 个中间 token，就相当于把 T 个深度-L 电路串接起来，总深度变成了 T×L。

<svg viewBox="0 0 680 190" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="340" y="22" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">CoT = 把固定深度的电路首尾串联起来</text>

  <rect x="20" y="60" width="110" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="75" y="82" text-anchor="middle" fill="#ededf0" font-size="11">前向 1</text>
  <text x="75" y="100" text-anchor="middle" fill="#94a3b8" font-size="10">深度 = L</text>

  <line x1="130" y1="87" x2="170" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowC)"/>
  <text x="150" y="75" text-anchor="middle" fill="#6e8eff" font-size="10">生成 token</text>

  <rect x="175" y="60" width="110" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="230" y="82" text-anchor="middle" fill="#ededf0" font-size="11">前向 2</text>
  <text x="230" y="100" text-anchor="middle" fill="#94a3b8" font-size="10">深度 = L</text>

  <line x1="285" y1="87" x2="325" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowC)"/>
  <text x="305" y="75" text-anchor="middle" fill="#6e8eff" font-size="10">...</text>

  <rect x="330" y="60" width="110" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="385" y="82" text-anchor="middle" fill="#ededf0" font-size="11">前向 T</text>
  <text x="385" y="100" text-anchor="middle" fill="#94a3b8" font-size="10">深度 = L</text>

  <line x1="440" y1="87" x2="480" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowC)"/>

  <rect x="485" y="60" width="130" height="55" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="550" y="82" text-anchor="middle" fill="#ededf0" font-size="11">最终答案</text>

  <line x1="20" y1="140" x2="615" y2="140" stroke="#a78bfa" stroke-width="1.2" stroke-dasharray="4,3"/>
  <text x="320" y="160" text-anchor="middle" fill="#a78bfa" font-size="12">有效总深度 ≈ T × L —— 随生成步数 T 线性增长，不再是常数</text>
</svg>

这正是 2024 年那篇《Chain of Thought Empowers Transformers to Solve Inherently Serial Problems》（Li, Liu, Zhou, Ma）给出的定量结果：**不用 CoT，常数深度、常数位精度的 Transformer 只能解决 AC⁰ 里的问题；一旦允许 T 步 CoT，同样的模型就能解决任何"规模为 T 的布尔电路"能解决的问题**——也就是说，模型能解决的问题复杂度，直接和你允许它"多想几步"的步数挂钩，而不再被架构深度锁死。

另一篇几乎同时的论文《The Expressive Power of Transformers with Chain of Thought》（Merrill & Sabharwal, 2023）给出了更精细的刻度：CoT 步数和能力提升之间不是"有没有"的关系，而是一个连续的谱系——

- **对数步数**（比如输入长度 n 的 log n 步）的 CoT，只能小幅度突破原本的边界，效果有限
- **线性步数**（跟输入长度 n 成正比）的 CoT，配合稍微推广一点的架构设定，能让模型识别**所有正则语言**——这是形式语言里非常宽泛的一大类
- **多项式步数**的 CoT，配合同样的架构设定，能让模型精确地覆盖 **P（所有多项式时间可解问题）**——这是历史上第一次有人给出某种 Transformer 变体和一个标准复杂度类之间的**精确**（不只是上界或下界）刻画

翻译成一句直白的话：**CoT 步数越多，模型能解决的问题的"串行深度"上限就越高**——这不是经验总结，是可以严格证明的数学事实。这也解释了为什么"让模型说得越详细、拆得越细"往往对复杂推理任务越有效：你实际上是在给它租借更多的"有效计算深度"。

顺带一提，2025 年还有一篇有意思的后续研究《Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers》，研究了一种更"懒"的做法——不要求模型生成有意义的推理内容，只是插入一堆无意义的占位符号（比如省略号 "..."）。结果发现：即使内容毫无意义，只要给模型足够多这样的"停顿 token"，同样能严格提升表达能力，让常数深度模型达到 TC⁰ 的完整能力（配合对数精度设定）。这进一步印证了核心结论——**关键变量是"允许模型多做几次前向传播"这个动作本身，内容是否有实际语义反而是次要的**（当然，在真实训练里，有意义的推理内容显然还是更有助于模型学会怎么正确地利用这些额外步骤）。

## 这意味着什么

把这条脉络串起来看，故事其实相当清晰：

Transformer 之所以能做到极高的并行度、支撑起今天动辄千亿参数的规模化训练，代价是它把自己的"思考深度"焊死成了一个和层数一样的常数。用电路复杂性理论的语言精确刻画，这个常数深度、有限精度的架构最多只能触达 **TC⁰** 这个复杂度等级——它能做大量的模式匹配、局部聚合、加权投票式的判断,但天生不擅长任何要求"把很多步骤严格串起来、一步不能少"的计算,比如长链条的进位加法、图的连通性判断、逻辑可满足性判断。

而 Chain-of-Thought 之所以有效,并不是因为它给模型"喂了新知识"——就像文章开头那张草稿纸一样,它给模型的是**额外的计算步骤和额外的状态存储空间**,让原本被焊死的常数深度,变成了可以随生成长度自由伸展的可变深度。这不是工程技巧上的偶然发现,而是精确对应着电路复杂性理论里"串联多个电路能提升总能力"这条早已被证明过的规律。

这条理论线索给我们的实际启示是:当你发现一个大模型在某个看起来很简单的任务上莫名其妙地失败时,值得先问一句——**这是一个需要精确串行累积的问题吗?** 如果是,那么无论怎么加大模型、加数据,只要它被要求"立刻给答案",失败几乎是必然的;而给它增加中间推理步骤(不管是显式的 CoT,还是哪怕看起来毫无意义的停顿 token),才是从架构层面真正对症的解法。

## 参考来源

- Merrill, W., & Sabharwal, A. (2023). *The Parallelism Tradeoff: Limitations of Log-Precision Transformers*. TACL. [arxiv.org/abs/2207.00729](https://arxiv.org/abs/2207.00729)
- Merrill, W., & Sabharwal, A. (2023). *The Expressive Power of Transformers with Chain of Thought*. [arxiv.org/abs/2310.07923](https://arxiv.org/abs/2310.07923)
- Li, Z., Liu, H., Zhou, D., & Ma, T. (2024). *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*. [arxiv.org/abs/2402.12875](https://arxiv.org/abs/2402.12875)
- Hahn, M. (2020). *Theoretical Limitations of Self-Attention in Neural Sequence Models*. TACL. [arxiv.org/abs/1906.06755](https://arxiv.org/abs/1906.06755)
- Merrill, W., Sabharwal, A., & Smith, N. A. (2022). *Saturated Transformers are Constant-Depth Threshold Circuits*. [arxiv.org/abs/2106.16213](https://arxiv.org/abs/2106.16213)
- Chiang, D., Cholak, P., & Pillay, A. (2023). *Tighter Bounds on the Expressivity of Transformer Encoders*. [arxiv.org/abs/2301.10743](https://arxiv.org/abs/2301.10743)
- Strobl, L., et al. (2024). *What Formal Languages Can Transformers Express? A Survey*. TACL. [arxiv.org/abs/2311.00208](https://arxiv.org/abs/2311.00208)
- Dziri, N., et al. (2023). *Faith and Fate: Limits of Transformers on Compositionality*. [Allen AI Blog](https://allenai.org/blog/faith-and-fate-limits-of-transformers-on-compositionality-d90726d635ef)
- (2025). *Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers*. [arxiv.org/abs/2505.21024](https://arxiv.org/abs/2505.21024)
