---
title: "LoRA：为什么只训练 0.01% 的参数就够了"
date: 2025-05-16
level: 3
series: "LLM 原理深度解析"
series_order: 3
series_total: 3
tags: [LoRA, 微调, 低秩适应, 参数高效, PEFT]
summary: "从内在维度假说出发，解释 LoRA 低秩适应为什么能用万分之一的参数达到全量微调的效果，以及 rank、alpha、初始化背后的数学直觉。"
---

# LoRA：为什么只训练 0.01% 的参数就够了

> GPT-3 有 1750 亿参数。如果你想让它学会一个新任务，难道真的需要更新所有 1750 亿个数字吗？LoRA 说：不，16 个"方向"就够了。

## 一个反直觉的事实

想象你是一位交响乐指挥，乐团有 100 个乐手。现在你要指挥他们演奏一首新曲子——不是从零训练 100 个人重新学乐器，而是给他们一些简短的指示："铜管组再轻一点"、"弦乐在这里加强"、"打击乐提前半拍"。

几句话，就能让整个乐团的表演焕然一新。

这正是 LoRA（Low-Rank Adaptation，低秩适应）背后的核心直觉：**一个已经训练好的大模型，要适应新任务时，需要的"调整"其实维度极低——远低于模型本身的参数量。**

2021 年，微软研究院的 Edward Hu 等人提出了 LoRA。他们发现，对 GPT-3 175B 做微调时，可以将可训练参数减少 **10000 倍**（从 1750 亿降到约 1800 万），GPU 显存需求降低 3 倍，而最终效果和全量微调几乎一样。

这件事的反直觉之处在于：一个有 1750 亿个参数的模型，学一个新任务居然只需要调整万分之一的参数？信息论告诉我们，这意味着**任务适应所需的信息量极低**。为什么会这样？

## 内在维度：高维空间里的"低维公路"

### 问题是什么

要理解 LoRA 为什么有效，我们需要先理解一个更基础的现象。

2020 年，Aghajanyan 等人（Meta AI）做了一组有趣的实验：他们把模型的参数空间"压缩"到一个随机的低维子空间里，然后只在这个小空间里做微调。结果令人震惊——

对于 RoBERTa-large（3.55 亿参数），在大部分 NLP 任务上，你只需要大约 **200-800 个自由度**就能达到 90% 的完整微调效果。

3.55 亿 vs 200。差了 6 个数量级。

这个数字——一个学习问题真正需要的最少自由度——就叫做**内在维度**（intrinsic dimensionality）。

### 直觉：为什么内在维度这么低？

想象你在一个巨大的体育馆里，有十万个座位。但今天的音乐会只有 200 个观众。虽然他们可以坐在十万个位置中的任何一个，但实际上他们聚集在一小片区域里——因为他们来听同一场演出，目的相似，行为相关。

大模型的参数空间也是类似的情况。预训练已经让模型学会了语言的通用结构——语法、语义、世界知识。当你微调它去做情感分析时，你不需要重新学这些通用知识。你只需要调整"怎么用这些已有知识来判断情感"的那一小部分行为。

更技术性地说：预训练模型的权重矩阵已经编码了非常丰富的表征。微调不是在白纸上画画，而是在一幅已经画好的画上做微小修改——把色调调暖一点，把某个区域提亮一点。这些修改具有**高度的结构性和相关性**，可以用很少的"方向"来描述。

### 从内在维度到低秩

现在关键的连接来了：如果微调只需要在一个低维子空间里操作，那么**权重的变化量 ΔW 本质上就是低秩的**。

什么是"低秩"？一个矩阵的秩表示它包含多少个线性独立的"方向"。一个 4096×4096 的矩阵理论上可以有 4096 个独立方向（满秩），但如果实际的权重变化只沿着 16 个方向发生，那这个变化矩阵就是秩为 16 的——即使它看起来是一个 4096×4096 的大矩阵。

这就像一张 A0 大纸上的内容其实只在一条窄带上有字。纸很大，但信息很少。

## LoRA 的核心公式：把"大矩阵"拆成"两个小矩阵"

### 问题是什么

我们知道微调的权重变化 ΔW 是低秩的。但在训练之前，我们不知道它具体是什么——我们需要让模型自己学出来。

如果直接让模型学一个 d×k 的 ΔW 矩阵（比如 4096×4096 = 1677 万参数），参数量还是太多。有没有办法在**结构上强制**这个矩阵是低秩的，同时大幅减少参数量？

### 核心想法

LoRA 的答案极其优雅：不要直接学 ΔW，而是把它**分解**成两个小矩阵的乘积。

$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- $W_0 \in \mathbb{R}^{d \times k}$ 是冻结的预训练权重（不更新）
- $B \in \mathbb{R}^{d \times r}$ 是一个"细高"矩阵
- $A \in \mathbb{R}^{r \times k}$ 是一个"矮宽"矩阵  
- $r \ll \min(d, k)$，通常 $r = 4, 8, 16, 32$

乘积 $BA$ 的秩最多为 $r$。这就像你用一条只有 $r$ 车道的高速公路连接两座城市——所有的"适应信息"都必须经过这 $r$ 个车道的瓶颈。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-lora" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Input -->
  <rect x="20" y="120" width="80" height="80" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="60" y="155" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">输入 x</text>
  <text x="60" y="175" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">k 维</text>
  
  <!-- Main path: W0 -->
  <line x1="100" y1="140" x2="250" y2="80" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrow-lora)"/>
  <rect x="250" y="50" width="140" height="55" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="320" y="75" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">W₀ · x</text>
  <text x="320" y="93" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">冻结（不训练）</text>
  
  <!-- LoRA path: A -->
  <line x1="100" y1="180" x2="180" y2="220" stroke="#6e8eff" stroke-width="2" marker-end="url(#arrow-lora)"/>
  <rect x="180" y="195" width="100" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="230" y="218" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">A · x</text>
  <text x="230" y="238" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">压缩到 r 维</text>
  
  <!-- LoRA path: B -->
  <line x1="280" y1="222" x2="360" y2="222" stroke="#6e8eff" stroke-width="2" marker-end="url(#arrow-lora)"/>
  <rect x="360" y="195" width="100" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="410" y="218" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">B · z</text>
  <text x="410" y="238" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">扩展回 d 维</text>
  
  <!-- Bottleneck label -->
  <text x="320" y="275" text-anchor="middle" fill="#ff7b7b" font-size="12" font-family="system-ui">↑ r 维瓶颈（如 r=16）</text>
  
  <!-- Addition -->
  <line x1="390" y1="77" x2="510" y2="155" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrow-lora)"/>
  <line x1="460" y1="222" x2="510" y2="170" stroke="#6e8eff" stroke-width="2" marker-end="url(#arrow-lora)"/>
  <circle cx="530" cy="160" r="20" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="530" y="165" text-anchor="middle" fill="#34d399" font-size="18" font-family="system-ui">+</text>
  
  <!-- Output -->
  <line x1="550" y1="160" x2="600" y2="160" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow-lora)"/>
  <rect x="600" y="130" width="80" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="640" y="155" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">输出 h</text>
  <text x="640" y="172" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">d 维</text>
  
  <!-- Labels -->
  <text x="320" y="30" text-anchor="middle" fill="#8888aa" font-size="12" font-family="system-ui">LoRA 前向传播：h = W₀x + (α/r) · BAx</text>
</svg>

### 参数量对比

具体算一下。假设一个 Transformer 层的注意力权重是 4096×4096：

| 方式 | 参数量 | 相对比例 |
|------|--------|---------|
| 全量微调 | 4096 × 4096 = 16,777,216 | 100% |
| LoRA (r=16) | 4096×16 + 16×4096 = 131,072 | **0.78%** |
| LoRA (r=4) | 4096×4 + 4×4096 = 32,768 | **0.20%** |
| LoRA (r=1) | 4096×1 + 1×4096 = 8,192 | **0.05%** |

用 r=16，参数量就降到了不到 1%。对于 GPT-3 175B 的全部注意力层，LoRA 把 350 亿可训练参数降到约 1800 万——缩小了将近 2000 倍。

### 前向传播的计算

前向传播时，LoRA 不需要显式构造那个 d×k 的 ΔW 矩阵，而是按顺序做两次小矩阵乘法：

1. **压缩**：$z = Ax$，把 k 维输入压缩到 r 维（花费 $O(rk)$）
2. **扩展**：$\Delta h = Bz$，把 r 维表示扩展回 d 维（花费 $O(dr)$）
3. **加和**：$h = W_0 x + \frac{\alpha}{r} \cdot \Delta h$

总计算量是 $O(r(d+k))$，远小于构造完整矩阵的 $O(dk)$。当 $d = k = 4096, r = 16$ 时，LoRA 路径的计算量只有约 0.8%。

## 缩放因子 α/r：一个容易忽略的细节

### 问题是什么

如果你增大 rank（比如从 r=8 改到 r=32），乘积 BA 的输出幅度会因为更多"通道"的贡献而自然增大。这意味着换个 rank 就要重新调学习率——非常烦人。

### 解决方案

LoRA 引入了一个缩放因子 $\alpha/r$：

$$h = W_0 x + \frac{\alpha}{r} \cdot BAx$$

其中 $\alpha$ 是一个固定的超参数（常设为 16 或 32）。除以 $r$ 的目的是**归一化掉 rank 对输出幅度的影响**，这样改变 rank 时不需要重新调学习率。

可以把 $\alpha/r$ 理解成一个"音量旋钮"：它控制 LoRA 适应的"声音"有多大。太大会压过预训练的知识，太小则学不到新东西。

**实践中的常见设置：**
- $\alpha = r$（即缩放因子为 1，最简单）
- $\alpha = 2r$（缩放因子为 2，稍微强调适应）
- $\alpha = 16$ 或 $32$ 固定不变（rank 变化时自动调节）

## 初始化的巧思：从预训练模型"无损启动"

### 问题是什么

训练开始时，我们希望 LoRA 不改变模型原始行为——即 $\Delta W = BA = 0$。这样模型从预训练的状态出发，然后逐渐学习适应。

如果 A 和 B 都随机初始化，$BA \neq 0$，模型一上来就偏离了预训练状态，可能破坏已有能力。

### LoRA 的初始化方案

- **B 初始化为全零**：$B = 0$
- **A 用 Kaiming 正态分布随机初始化**：$A \sim \mathcal{N}(0, \sigma^2)$

这样，在训练开始时 $BA = 0 \cdot A = 0$，完美地从预训练模型出发。

但为什么是"B=0, A=随机"而不是反过来？这里有一个微妙的梯度流考量：

对 A 的梯度是 $\frac{\partial L}{\partial A} = B^T \frac{\partial L}{\partial h} x^T$，对 B 的梯度是 $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial h} (Ax)^T$。

如果 A=0 且 B 随机，那么对 B 的梯度中 $Ax = 0$（因为 $A=0$），所以 B 收到的梯度也是零——训练卡死了！

而 B=0 且 A 随机时：对 A 的梯度中有 $B^T = 0$，但这没关系，因为**对 B 的梯度** $\frac{\partial L}{\partial h}(Ax)^T$ 是非零的（A 是随机的，$Ax \neq 0$）。B 先得到梯度开始更新，然后 A 也能跟着动起来。

## 实际中 LoRA 加在哪里？

### 选择目标层

Transformer 中有很多权重矩阵可以做 LoRA：

- **注意力层**：$W_Q, W_K, W_V, W_O$（查询/键/值/输出投影）
- **FFN 层**：$W_{up}, W_{down}, W_{gate}$

LoRA 原始论文发现：**只对 $W_Q$ 和 $W_V$ 做 LoRA**就能达到不错效果。但后续研究表明，对所有线性层都加 LoRA（用更小的 rank）通常效果更好。

现在的主流做法是：对所有注意力和 FFN 权重都加 LoRA，rank 设为 8-32。

### 推理时零开销

LoRA 最优雅的性质之一：**推理时可以把适应合并回原始权重，不增加任何计算开销**。

训练完成后，直接计算：

$$W_{merged} = W_0 + \frac{\alpha}{r} \cdot BA$$

合并后的模型和普通模型结构完全一样，推理速度毫无差别。这比 Adapter（在网络中插入额外层）优越得多——Adapter 推理时有额外延迟，LoRA 没有。

而且，你可以保存多个 LoRA 适配器（每个只有几 MB），共享同一个基础模型，按需加载不同的"人格"或"能力"。就像一件衣服配不同配饰。

## 秩的选择：一个关键的超参数

### 为什么 r=16 通常就够了？

回到内在维度的视角。Hu 等人在原论文中做了一个关键实验：他们用 SVD 分析全量微调得到的 $\Delta W$，发现**其奇异值衰减极快**——最大的几个奇异值包含了绝大部分信息，后面的近乎为零。

具体来说，在 GPT-3 的实验中，他们发现：
- r=1 就能达到可观的效果
- r=4 已经非常接近全量微调
- r=8 或 r=16 基本上和全量微调无法区分
- 进一步增大 rank 几乎没有收益

这意味着真实的微调信号确实集中在极少数方向上。

### 但有时候 r=16 不够

"LoRA Learns Less and Forgets Less"（2024）的研究发现，LoRA 在以下场景可能明显落后于全量微调：

1. **学习全新知识**（vs. 激活已有知识）：如果任务需要模型学习预训练中完全没见过的模式（比如一种新编程语言），低秩约束确实会限制表达力
2. **复杂推理任务**：数学推理、代码生成等需要精细调整的任务可能需要更高的 rank
3. **大数据集微调**：数据越多，可能需要更高的 rank 来充分利用信息

不过有一个有趣的补偿：LoRA 虽然"学得少"，但也"忘得少"——它对预训练知识的遗忘比全量微调轻得多。这在很多实际应用中反而是优势。

<svg viewBox="0 0 650 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-chart" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="80" y1="230" x2="600" y2="230" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-chart)"/>
  <line x1="80" y1="230" x2="80" y2="30" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-chart)"/>
  
  <!-- Y axis label -->
  <text x="30" y="130" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" transform="rotate(-90, 30, 130)">任务性能 (%)</text>
  
  <!-- X axis label -->
  <text x="340" y="265" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">LoRA Rank (r)</text>
  
  <!-- Full FT reference line -->
  <line x1="80" y1="55" x2="590" y2="55" stroke="#34d399" stroke-width="1" stroke-dasharray="6,4"/>
  <text x="595" y="59" fill="#34d399" font-size="11" font-family="system-ui">全量微调</text>
  
  <!-- Performance curve (rapid rise then plateau) -->
  <path d="M 120 200 Q 180 100, 240 75 Q 300 60, 400 58 Q 500 56, 580 55" fill="none" stroke="#6e8eff" stroke-width="2.5"/>
  
  <!-- Data points -->
  <circle cx="120" cy="200" r="5" fill="#6e8eff"/>
  <circle cx="180" cy="110" r="5" fill="#6e8eff"/>
  <circle cx="240" cy="75" r="5" fill="#6e8eff"/>
  <circle cx="340" cy="62" r="5" fill="#6e8eff"/>
  <circle cx="500" cy="57" r="5" fill="#6e8eff"/>
  
  <!-- X axis ticks -->
  <text x="120" y="248" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">1</text>
  <text x="180" y="248" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">4</text>
  <text x="240" y="248" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">8</text>
  <text x="340" y="248" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">16</text>
  <text x="500" y="248" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">64</text>
  
  <!-- Y axis ticks -->
  <text x="70" y="203" text-anchor="end" fill="#8888aa" font-size="11" font-family="system-ui">80</text>
  <text x="70" y="133" text-anchor="end" fill="#8888aa" font-size="11" font-family="system-ui">90</text>
  <text x="70" y="63" text-anchor="end" fill="#8888aa" font-size="11" font-family="system-ui">100</text>
  
  <!-- Annotation: sweet spot -->
  <rect x="210" y="85" width="90" height="25" rx="4" fill="#1e1e2a" stroke="#ff7b7b" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="255" y="102" text-anchor="middle" fill="#ff7b7b" font-size="11" font-family="system-ui">甜蜜区间</text>
  
  <!-- Arrow to sweet spot -->
  <text x="370" y="120" fill="#8888aa" font-size="11" font-family="system-ui">r=8~16 已接近</text>
  <text x="370" y="135" fill="#8888aa" font-size="11" font-family="system-ui">全量微调效果</text>
  
  <!-- Title -->
  <text x="340" y="20" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">LoRA Rank vs 任务性能（典型模式）</text>
</svg>

### rank 选择的实践指南

| 场景 | 推荐 rank | 理由 |
|------|-----------|------|
| 简单分类/情感分析 | 4-8 | 任务简单，内在维度极低 |
| 指令跟随/对话 | 16-32 | 需要调整多种行为模式 |
| 代码生成 | 32-64 | 需要精细的语法/逻辑调整 |
| 学习新领域知识 | 64-128 | 需要编码新信息 |
| 不确定时 | 16 | 安全的默认值 |

## QLoRA：当极致压缩遇上 LoRA

2023 年，Dettmers 等人提出 QLoRA，把 LoRA 的效率推到了新极限：

**核心思路**：把冻结的基础模型量化到 4-bit（用一种叫 NormalFloat4 的特殊格式），但 LoRA 的 A、B 矩阵保持 16-bit 精度。

这意味着：
- 基础模型 175B 参数 × 4 bit ≈ 87.5 GB（vs 原来的 350 GB）
- LoRA 适配器还是 16-bit 计算，保持精度
- 实际效果：**在单张 48GB A100 上微调 65B 模型**

QLoRA 还引入了"双重量化"——对量化常数本身再做一次量化，进一步节省 0.37 bit/参数。

关键洞察是：冻结的权重只需要做前向传播，精度损失可以被 LoRA 的高精度适配器"补偿"回来。实验表明 QLoRA 和全精度 LoRA 几乎没有性能差距。

## DoRA：更精细的分解

2024 年提出的 DoRA（Weight-Decomposed Low-Rank Adaptation）观察到 LoRA 和全量微调在**权重更新模式**上的一个细微差异：

全量微调同时改变权重的"方向"和"幅度"，但 LoRA 的低秩约束让这两者耦合在一起。DoRA 的解法是把权重分解为：

$$W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

其中 $m$ 是一个可学的幅度向量。这让方向的调整（通过 LoRA）和幅度的调整（通过 $m$）解耦，在多个任务上比标准 LoRA 有 1-3% 的提升。

## 为什么低秩假设成立：更深的理解

回到最初的问题：为什么模型适应只需要低秩变化？

从几个不同角度理解：

**1. 预训练的"过完备"表征**

预训练模型学到的表征远比任何单个下游任务需要的丰富。它就像一本百科全书——做情感分析时你只需要翻到"情感词汇"那几页。权重变化集中在"如何使用已有特征"，而非"创造新特征"。

**2. 奇异值分解的视角**

如果你对全量微调的 $\Delta W$ 做 SVD：$\Delta W = U\Sigma V^T = \sum_i \sigma_i u_i v_i^T$，会发现奇异值 $\sigma_i$ 衰减得极快。前 16 个奇异值通常占了总能量的 95% 以上。这意味着虽然 $\Delta W$ 形式上是个大矩阵，但它的信息几乎全集中在少数方向上。

**3. 优化景观的视角**

从损失函数的角度看，微调的"有效参数空间"（能显著降低损失的方向）远小于名义参数空间。大部分参数方向上，梯度接近零——模型已经在那些方向上找到了好的解。

**4. 任务相似性**

下游任务（翻译、摘要、问答）虽然表面不同，但底层都依赖相似的语言能力。从一个通用模型适应到特定任务，本质上是在一个共享的"任务流形"上做小幅位移。

## LoRA 的影响与生态

LoRA 发表后迅速成为 LLM 微调的**事实标准**。它的影响远超一种技术方法：

1. **民主化微调**：不再需要数千美元的 GPU 集群，个人开发者用一张消费级显卡就能微调大模型
2. **适配器生态**：CivitAI（图像）、HuggingFace（语言）上数以万计的 LoRA 适配器被社区共享
3. **多任务部署**：一个基础模型 + 多个 LoRA 适配器 = 一台服务器服务多个任务
4. **衍生方法爆发**：QLoRA、DoRA、AdaLoRA、LoRA+、rsLoRA... 一整个研究方向因它而生

## 总结：LoRA 的本质

LoRA 的深层含义超越了"省参数"这个实用优势。它揭示了一个关于大模型的根本洞察：

> **预训练模型已经知道"几乎一切"。微调不是教它新知识，而是告诉它"该怎么用你已有的知识"——而这个指令的信息量极低。**

16 个方向、万分之一的参数、几 MB 的文件——就够了。这不是工程技巧，是大模型的信息结构使然。

## 下一篇预告

我们讲了 LoRA 如何高效地"教"模型新行为。但还有一个更神奇的现象：大模型不需要更新任何参数，只靠 prompt 中的几个例子就能"学会"新任务——这就是 In-Context Learning。它为什么有效？和权重更新有什么数学关系？下一篇我们来揭开这个谜团。
