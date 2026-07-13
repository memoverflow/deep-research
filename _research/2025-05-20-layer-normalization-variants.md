---
title: "归一化的进化：从 LayerNorm 到 RMSNorm 再到 DeepNorm"
date: 2025-05-20
level: 3
series: "LLM 原理深度解析"
series_order: 11
series_total: 39
tags: [normalization, LayerNorm, RMSNorm, DeepNorm, training-stability]
summary: "为什么深度网络需要归一化？LayerNorm 做了什么几何操作？RMSNorm 为什么删掉均值还能用？DeepNorm 怎么让 1000 层 Transformer 稳定训练？"
---

# 归一化的进化：从 LayerNorm 到 RMSNorm 再到 DeepNorm

> 每个现代 LLM 的每一层里都藏着一个不起眼的操作——归一化。它只有两三行代码，却决定了模型能否成功训练。

## 故事从一个简单的问题开始

想象你在叠积木。一块一块往上叠，前几块很稳，但叠到第 20 块时开始晃动，叠到第 50 块时整栋积木楼轰然倒塌。

深度神经网络面对的就是这个问题。每一层的计算都会改变数据的分布——某些数值越来越大，某些趋近于零。当网络有几十层时，这种分布漂移会像多米诺骨牌一样级联放大，导致梯度要么爆炸（变成天文数字）要么消失（变成零），训练直接崩溃。

2016 年之前，人们用 Batch Normalization 来解决这个问题：把一个 mini-batch 里所有样本的同一个神经元的值拉回到均值 0、方差 1 的标准分布。效果很好，但它有个致命缺点——**依赖 batch 里的其他样本**。对于语言模型来说，每个句子长度不同，batch 内统计量很不稳定；推理时 batch 可能只有 1 个样本，统计量根本无法计算。

这就引出了今天的主角：**Layer Normalization**——不看 batch 中的邻居，只看自己这一个样本，在特征维度上做归一化。

## Layer Normalization：给每个向量「校准」

### 问题是什么

一个 Transformer 层的输出是一个向量（比如 4096 维）。经过矩阵乘法和激活函数后，这个向量的各个分量可能差异巨大——有些是 100，有些是 0.001。这种尺度不均匀会给下一层制造麻烦：它的参数是按照某个"正常"尺度初始化的，突然来了一个尺度完全不同的输入，学习就会变得极不稳定。

### 直觉：核心想法

LayerNorm 做的事情可以用一句话概括：**把向量拉到一个标准的"球面"上，然后让模型学习该怎么重新缩放它。**

更具体地说，两步操作：

1. **中心化**：算出向量所有分量的均值 $\mu$，然后每个分量减去 $\mu$。就像把一根有偏移的尺子重新对齐零点。

2. **归一化**：算出标准差 $\sigma$，每个分量除以 $\sigma$。就像把一把刻度不规则的尺子统一缩放到标准刻度。

最后，一对可学习参数 $\gamma$（缩放）和 $\beta$（偏移）让模型在标准化之后可以自由调整分布——如果某个维度确实应该大一些，模型可以通过学习来恢复。

### 几何视角

这里有一个优美的几何解释：LayerNorm 实际上是在做**投影**。

想象一个 $d$ 维空间里的向量 $\mathbf{x}$。减去均值等价于把它投影到垂直于全 1 向量 $\mathbf{1} = (1,1,...,1)$ 的超平面上（因为均值就是 $\mathbf{x}$ 在 $\mathbf{1}$ 方向上的投影长度除以 $\sqrt{d}$）。除以标准差等价于把它归一化到单位长度。

所以，**LayerNorm 把任意向量投影到一个 $(d-1)$ 维单位超球面上**。所有经过 LayerNorm 的向量都住在同一个球面上，只是方向不同。这就是为什么它能有效抑制尺度问题——不管输入有多大多小，输出永远在球面上。

<svg viewBox="0 0 650 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-ln" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Step 1: Original vector -->
  <rect x="10" y="30" width="180" height="220" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="100" y="20" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">原始向量</text>
  <text x="100" y="60" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">x = [3.2, -1.5, 0.8, 7.1]</text>
  <text x="100" y="90" text-anchor="middle" fill="#999" font-size="11" font-family="system-ui">尺度不一，方向混乱</text>
  <!-- arrow representing vector with unequal magnitudes -->
  <line x1="50" y1="170" x2="160" y2="110" stroke="#ff6b6b" stroke-width="2" marker-end="url(#arrow-ln)"/>
  <text x="100" y="210" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">长度和方向都不规范</text>

  <!-- Arrow between boxes -->
  <line x1="195" y1="140" x2="235" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-ln)"/>
  <text x="215" y="130" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">−μ</text>

  <!-- Step 2: Centered -->
  <rect x="240" y="30" width="180" height="220" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="330" y="20" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">中心化（减均值）</text>
  <text x="330" y="60" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">x−μ = [0.8, -3.9, -1.6, 4.7]</text>
  <text x="330" y="90" text-anchor="middle" fill="#999" font-size="11" font-family="system-ui">投影到垂直于 1⃗ 的超平面</text>
  <line x1="280" y1="170" x2="380" y2="120" stroke="#ffd93d" stroke-width="2" marker-end="url(#arrow-ln)"/>
  <text x="330" y="210" text-anchor="middle" fill="#ffd93d" font-size="10" font-family="system-ui">去掉了均值偏移</text>

  <!-- Arrow between boxes -->
  <line x1="425" y1="140" x2="465" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-ln)"/>
  <text x="445" y="130" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">÷σ</text>

  <!-- Step 3: Normalized -->
  <rect x="470" y="30" width="170" height="220" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="555" y="20" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">归一化（除以标准差）</text>
  <text x="555" y="60" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">x̂ = [0.24, -1.17, ...]</text>
  <text x="555" y="90" text-anchor="middle" fill="#999" font-size="11" font-family="system-ui">投影到单位超球面</text>
  <!-- unit circle -->
  <circle cx="555" cy="165" r="45" fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="555" y1="165" x2="585" y2="135" stroke="#34d399" stroke-width="2" marker-end="url(#arrow-ln)"/>
  <text x="555" y="230" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">所有向量都在球面上</text>
</svg>

### 技术细节

对于一个 $d$ 维向量 $\mathbf{x} = (x_1, x_2, ..., x_d)$：

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma + \epsilon} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$（均值）
- $\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}$（标准差）
- $\gamma, \beta$ 是可学习的 $d$ 维参数
- $\epsilon$ 是防止除零的小常数（通常 $10^{-5}$）

翻译成人话：先算均值和标准差 → 标准化 → 用可学习参数重新调整。

## Pre-Norm vs Post-Norm：一个"小"选择引发的大争论

### 位置决定命运

LayerNorm 放在哪里，这个看似微不足道的选择，实际上深刻影响训练动态。

**Post-Norm**（原始 Transformer 2017）：先做子层计算，加上残差，然后归一化。

$$\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + F_l(\mathbf{x}_l))$$

**Pre-Norm**（GPT-2 之后的标准）：先归一化，然后做子层计算，最后加残差。

$$\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\text{LN}(\mathbf{x}_l))$$

为什么这个区别如此重要？

### 跷跷板难题

Post-Norm 的问题在于梯度回传路径。当你对输出求导时，梯度必须穿过每一层的 LayerNorm。LayerNorm 包含除以标准差的操作，这会**缩小**梯度。层数一多，梯度被反复缩小，到底层时已经消失了——前面的层几乎学不到任何东西。

Pre-Norm 解决了这个问题：残差连接创造了一条"高速公路"，梯度可以直接从顶层畅通无阻地流到底层，不经过任何归一化操作。所以 Pre-Norm 可以轻松训练上百层。

但天下没有免费的午餐。Pre-Norm 的代价是**表示崩塌**（representation collapse）——随着网络加深，残差主路上的信号越来越大（因为每层都在累加），而每一层的贡献相对于整个残差流来说越来越渺小。深层的 Transformer 层变得几乎无用，好像模型"实际有效深度"远小于标称深度。

这就是一个跷跷板：Pre-Norm 稳定但浅，Post-Norm 深但容易崩。

### 现代 LLM 的选择

几乎所有现代 LLM（GPT 系列、LLaMA、Qwen、DeepSeek）都选择了 Pre-Norm，因为训练稳定性在工程上远比"理论上更优的表示"重要——你总可以通过加更多参数来弥补表示能力，但训练崩了就什么都没了。

但这个故事还没结束——DeepNorm 试图打破这个跷跷板，我们后面会讲到。

## RMSNorm：删掉一半操作，性能不变

### 一个大胆的假设

2019 年，张博华和 Rico Sennrich 提出了一个看似简单到不可能成功的想法：如果我们把 LayerNorm 里的"减均值"这一步直接删掉，只保留"除以尺度"呢？

这背后有一个数学直觉：LayerNorm 实际上在做两件事——**重新中心化**（re-centering，减均值）和**重新缩放**（re-scaling，除以标准差）。作者假设，真正重要的是缩放操作，中心化其实可有可无。

为什么这个假设合理？在实际训练中，可学习的偏置参数 $\beta$ 可以弥补缺失的中心化。而且高维向量中，均值通常很小（各分量正负抵消），减不减它对结果影响不大。

### 核心想法

RMSNorm 的操作极其简单：用向量的**均方根**（Root Mean Square）来归一化，然后乘以可学习的缩放参数。

用人话说：不关心向量的中心在哪（不减均值），只关心向量有多"大"（用 RMS 衡量），然后把它缩放到标准大小。

### 为什么这能加速

删掉均值计算带来的好处不只是少了一次加法。在 GPU 实现中：

1. **少一次 reduce 操作**：计算均值需要对 $d$ 个元素求和（一次 reduce），计算方差又要一次 reduce。RMSNorm 只需要一次 reduce（求平方和）。在 GPU 上，reduce 操作需要线程间同步，这是昂贵的操作。

2. **更简单的内存模式**：LayerNorm 需要保存 $\mathbf{x} - \mu$ 的中间结果等待方差计算完成。RMSNorm 的计算是严格的单 pass——读一遍数据就够了。

3. **更容易融合**：简单的计算图让 kernel fusion（算子融合）更容易优化。

实测中，RMSNorm 通常比 LayerNorm 快 **10-30%**，具体取决于隐藏维度和硬件。对于一个 70B 参数的模型训练几万步来说，这些节省加起来相当可观。

### 技术细节

$$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x}) + \epsilon}$$

其中：

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

注意对比 LayerNorm：
- 没有减均值（$\mu$）
- 除的不是标准差（$\sigma$），而是 RMS
- 没有可学习的偏置 $\beta$（只有缩放 $\gamma$）

翻译回人话：直接把向量除以它的"平均大小"（RMS），然后乘以可学习的缩放因子。

### 谁在用

今天几乎所有一线 LLM 都用 RMSNorm + Pre-Norm 的组合：
- **LLaMA** 全系列（Meta）
- **Qwen** 系列（阿里巴巴）
- **DeepSeek** 系列
- **Gemma**（Google）
- **OLMo**（AI2）
- **GPT-OSS**（OpenAI 开源版）

LayerNorm 在 LLM 领域已经基本退休了。

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-rms" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>

  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">LayerNorm vs RMSNorm 计算流程对比</text>

  <!-- LayerNorm path (top) -->
  <text x="40" y="65" fill="#6e8eff" font-size="12" font-family="system-ui" font-weight="bold">LayerNorm</text>
  
  <rect x="30" y="80" width="100" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="80" y="105" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">输入 x</text>
  
  <line x1="130" y1="100" x2="160" y2="100" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-rms)"/>
  
  <rect x="165" y="80" width="100" height="40" rx="8" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="215" y="105" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">求均值 μ</text>
  
  <line x1="265" y1="100" x2="295" y2="100" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-rms)"/>
  
  <rect x="300" y="80" width="100" height="40" rx="8" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="350" y="105" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">x − μ</text>
  
  <line x1="400" y1="100" x2="430" y2="100" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-rms)"/>
  
  <rect x="435" y="80" width="100" height="40" rx="8" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="485" y="105" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">求方差 σ²</text>
  
  <line x1="535" y1="100" x2="565" y2="100" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-rms)"/>
  
  <rect x="570" y="80" width="100" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="620" y="105" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">÷σ → γ,β</text>

  <!-- Cost label -->
  <text x="350" y="145" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">2 次 reduce + 中间存储 + 偏置参数</text>

  <!-- Divider -->
  <line x1="30" y1="165" x2="670" y2="165" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- RMSNorm path (bottom) -->
  <text x="40" y="195" fill="#34d399" font-size="12" font-family="system-ui" font-weight="bold">RMSNorm</text>
  
  <rect x="30" y="210" width="100" height="40" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="80" y="235" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">输入 x</text>
  
  <line x1="130" y1="230" x2="200" y2="230" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow-rms)"/>
  
  <rect x="205" y="210" width="140" height="40" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="275" y="235" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">求 RMS = √(Σx²/d)</text>
  
  <line x1="345" y1="230" x2="415" y2="230" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow-rms)"/>
  
  <rect x="420" y="210" width="120" height="40" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="480" y="235" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">x ÷ RMS → γ</text>

  <!-- Cost label -->
  <text x="350" y="275" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">1 次 reduce，无中间存储，无偏置 → 快 10-30%</text>
</svg>

## DeepNorm：让 1000 层 Transformer 成为可能

### 问题：为什么不能直接叠更多层？

2022 年之前，最深的 Transformer 大约是几十到一百层。不是人们不想叠更深——理论上更深的网络应该有更强的表示能力——而是训练会崩溃。

即使用了 Pre-Norm，深层 Transformer 也面临困难。随着层数增加到数百层，训练初期的更新幅度如果稍大，就会导致后续层的输出剧烈变化，引发不稳定。人们不得不用极小的学习率和长时间的 warmup 来小心翼翼地"热车"，这大大降低了训练效率。

微软的研究者发现了一个关键洞察：**稳定的训练意味着每一层的输出变化应该是"渐进式"的**。如果某一步参数更新导致某层输出剧烈变化，这个变化会被后续层放大，最终导致灾难。

### 核心想法：放大残差，缩小更新

DeepNorm 的思路出奇地简洁：既然问题是每一层的更新太大、会扰乱后续层，那就**让残差连接更强、让每层的初始贡献更小**。

具体来说，DeepNorm 修改了 Post-Norm 的公式：

$$\mathbf{x}_{l+1} = \text{LN}(\alpha \cdot \mathbf{x}_l + F_l(\mathbf{x}_l))$$

注意那个 $\alpha$——它是一个大于 1 的常数（对于 1000 层的网络，$\alpha$ 可以大到几十）。这意味着残差连接被**放大**了，而子层的贡献 $F_l(\mathbf{x}_l)$ 相对来说变得很小。

同时，DeepNorm 要求在初始化时把子层的权重**缩小**一个因子 $\beta$（远小于 1）。这确保在训练初期，每一层的影响微乎其微，信号主要通过残差直通。随着训练进行，模型逐渐学到每层应该贡献多少。

### 类比理解

想象一条宽阔的河流（残差连接）。每一层像一条小溪汇入河流。

- **普通 Post-Norm**：河流和小溪差不多宽，每条小溪都能显著改变河流方向。100 条小溪之后，河流的方向完全不可预测。
- **DeepNorm**：河流非常宽阔（$\alpha$ 放大），每条小溪非常细小（$\beta$ 缩小初始化）。即使有 1000 条小溪汇入，河流的整体方向始终稳定。但随着时间推移（训练进行），某些小溪可以逐渐加宽（参数更新），开始有意义地贡献方向变化。

### $\alpha$ 和 $\beta$ 的具体取值

论文通过理论推导给出了具体公式。对于一个 $N$ 层的模型：

| 架构类型 | $\alpha$ | $\beta$ |
|---------|----------|---------|
| 仅编码器 | $(2N)^{1/4}$ | $(8N)^{-1/4}$ |
| 仅解码器 | $(2M)^{1/4}$ | $(8M)^{-1/4}$ |
| 编码-解码器 | 编码器 $(2N)^{1/4}$，解码器 $(16NM^2)^{1/16}$ | 相应缩小 |

以一个 200 层的 decoder-only 模型为例：$\alpha = (400)^{0.25} \approx 4.47$，$\beta = (1600)^{-0.25} \approx 0.16$。

翻译成人话：残差路径被放大约 4.5 倍，而每层的初始权重被缩小到正常的 16%。

### 效果

DeepNorm 让微软成功训练了超过 **1000 层**的 Transformer，而且不需要学习率 warmup。同等参数量下，更深的模型在翻译和语言建模任务上都优于更宽但更浅的模型，验证了"深度有意义"的直觉。

更重要的是，DeepNorm 打破了 Pre-Norm 和 Post-Norm 之间的跷跷板：它使用了 Post-Norm 的结构（归一化在残差加法之后），获得了 Post-Norm 更好的表示质量，同时通过 $\alpha / \beta$ 的精心设计获得了 Pre-Norm 级别的训练稳定性。

## 三种归一化的全景对比

| | LayerNorm | RMSNorm | DeepNorm |
|---|---|---|---|
| **操作** | 减均值 + 除标准差 + γ,β | 除 RMS + γ | α 放大残差 + LayerNorm |
| **几何含义** | 投影到 (d-1) 维超球面 | 投影到 d 维超球面（保留均值方向） | 加权残差后投影到超球面 |
| **可学习参数** | γ（缩放）+ β（偏移）| 只有 γ（缩放）| LayerNorm 的 γ,β |
| **计算代价** | 2 次 reduce | 1 次 reduce | 1 次标量乘法 + LayerNorm |
| **搭配位置** | Post-Norm 或 Pre-Norm | 几乎总是 Pre-Norm | 修改版 Post-Norm |
| **最大稳定深度** | ~100 层（Pre-Norm）| ~100 层（Pre-Norm）| 1000+ 层 |
| **代表模型** | GPT-2, BERT | LLaMA, Qwen, DeepSeek | DeepNet (微软) |
| **论文年份** | 2016 | 2019 | 2022 |

## 一个容易被忽视的细节：不可逆性

LayerNorm 有一个信息论层面的特性很少被讨论：**它是不可逆的**。

当你减去均值时，你把向量在全 1 方向上的投影永久丢弃了。即使后面有可学习的 $\beta$，它也只能提供一个**统一的**偏移，而无法恢复被减掉的那个**因输入而异**的均值信息。

这意味着如果向量的均值本身携带有用信息（比如整体的"能量"或"置信度"），LayerNorm 会把它擦除。相比之下，RMSNorm 保留了均值方向的信息（因为它不减均值），这可能是它在某些任务上不输甚至微胜 LayerNorm 的一个原因。

## 这意味着什么

归一化的进化故事其实是一个不断追问"哪些操作是真正必要的"的过程：

1. **BatchNorm → LayerNorm**：去掉了对 batch 的依赖，让每个样本独立归一化
2. **LayerNorm → RMSNorm**：去掉了减均值操作，发现只保留缩放就够了
3. **Post-Norm → Pre-Norm**：移动归一化位置，换取训练稳定性
4. **Pre-Norm → DeepNorm**：通过加权残差连接，同时获得稳定性和表示质量

每一步进化都在做减法或重新平衡——删除不必要的计算，或者在稳定性和表示能力之间找到更好的平衡点。

对于今天的 LLM 实践者来说：如果你在训练一个标准深度（32-128 层）的模型，**RMSNorm + Pre-Norm** 是最安全的选择，也是所有一线实验室的默认配置。如果你需要极端深度（500+ 层），DeepNorm 提供了一条经过验证的路径。

而下一代归一化技术已经在路上——比如结合 Pre-Norm 和 Post-Norm 优势的 SiameseNorm，以及完全去掉归一化层的探索（nGPT 等）。归一化这个"不起眼的小操作"，远比它看起来更深刻。

## 下一篇预告

我们花了大量篇幅讨论如何让训练**稳定**，但还有一个同样重要的问题：如何让训练**高效**。当你的模型有几十亿参数、数据有万亿 token 时，学习率到底该怎么调？Warmup 为什么有效？Cosine 衰减的理论依据是什么？下一篇，我们将深入学习率调度策略的原理。
