---
title: "权重初始化的数学：从 Xavier 到 Kaiming 再到 µP"
date: 2025-06-28
level: 3
series: "LLM 原理深度解析"
series_order: 17
series_total: 17
tags: [initialization, xavier, kaiming, muP, gradient, training]
summary: "为什么深度网络训练之前要精心设置初始权重？从方差守恒的直觉出发，推导 Xavier 和 Kaiming 初始化，再到 µP 如何让超参数跨尺度迁移。"
---

# 权重初始化的数学：从 Xavier 到 Kaiming 再到 µP

> 深度学习训练的第一步不是梯度下降，而是决定从哪里出发。初始化做错了，再好的优化器也救不回来。

## 故事从这里开始

想象你正站在一座巨大的山脉前，准备开始一段徒步旅行。你手里有一张地图（梯度），告诉你哪个方向是下坡。但有一个问题：**你的起点决定了一切**。

如果你出发的位置恰好在一片平坦的高原上，地图会告诉你"各个方向都差不多"——你会迷失方向，永远走不下去。如果你恰好站在一个极陡的悬崖边缘，一步就会把你甩到谷底以下，然后弹射到另一个山峰，永远在极端之间震荡。

这就是深度神经网络面对的初始化困境：**起点太"小"，信号消失；起点太"大"，信号爆炸。**

在 2010 年之前，人们用随机数（比如标准正态分布）初始化网络权重，然后祈祷训练能顺利进行。结果是：超过 5-6 层的网络几乎无法训练。不是算法不对，不是数据不够——纯粹是因为信号在层间传播时被反复放大或缩小，最终变成了 0 或 ∞。

这篇文章讲的就是：怎么找到那个"刚刚好"的起点。

## 问题的本质：方差的指数效应

### 一个简单的思想实验

假设你有一个 100 层的网络，每层做的事情很简单：把输入乘以一个权重矩阵。暂时忘掉激活函数，只考虑线性变换。

如果每一层将信号的方差乘以一个常数 $c$，那么经过 100 层后，信号的方差变成了原来的 $c^{100}$ 倍。

- 当 $c = 1.1$ 时：$1.1^{100} ≈ 13,781$ — 信号爆炸
- 当 $c = 0.9$ 时：$0.9^{100} ≈ 0.0000265$ — 信号消失
- 当 $c = 1.0$ 时：$1.0^{100} = 1$ — 完美保持

看到了吗？哪怕每层只偏离 10%，100 层之后就是天壤之别。**这就是为什么初始化必须精确——不是"大概对"就行，而是必须在数学上让 $c$ 尽量等于 1。**

### 前向传播的方差分析

让我们严格地计算这个 $c$。考虑一个全连接层 $y = Wx$，其中 $W$ 是 $d_{out} \times d_{in}$ 的权重矩阵。

输出的第 $i$ 个元素是：

$$y_i = \sum_{k=1}^{d_{in}} W_{ik} \cdot x_k$$

这是 $d_{in}$ 个随机变量的乘积之和。如果权重 $W_{ik}$ 和输入 $x_k$ 互相独立，且权重的均值为零，那么：

$$\text{Var}(y_i) = d_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

翻译回人话：**输出的方差 = 输入维度 × 权重方差 × 输入方差**。

所以那个"每层的放大系数" $c$ 就是 $d_{in} \cdot \text{Var}(W)$。要让 $c = 1$，我们需要：

$$\text{Var}(W) = \frac{1}{d_{in}}$$

这就是方差守恒的核心条件。但故事才刚刚开始——因为我们还没考虑反向传播。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Layer boxes -->
  <rect x="30" y="100" width="100" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="80" y="135" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">第 1 层</text>
  <rect x="200" y="100" width="100" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="250" y="135" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">第 2 层</text>
  <rect x="370" y="100" width="100" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="420" y="135" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">第 L 层</text>
  <rect x="540" y="100" width="120" height="60" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="600" y="135" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">输出 / Loss</text>
  <!-- Forward arrows -->
  <line x1="130" y1="130" x2="195" y2="130" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="300" y1="130" x2="330" y2="130" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="340" y="135" fill="#6e8eff" font-size="16">···</text>
  <line x1="355" y1="130" x2="365" y2="130" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="470" y1="130" x2="535" y2="130" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Variance labels -->
  <text x="80" y="85" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">Var = σ²</text>
  <text x="250" y="85" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">Var = c·σ²</text>
  <text x="420" y="85" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">Var = c^L · σ²</text>
  <!-- Bottom labels for c values -->
  <text x="165" y="180" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">×c</text>
  <text x="500" y="180" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">×c</text>
  <!-- Explosion / vanish indicators -->
  <text x="350" y="230" text-anchor="middle" fill="#f87171" font-size="12" font-family="system-ui">c > 1 → 信号爆炸 💥</text>
  <text x="350" y="255" text-anchor="middle" fill="#fbbf24" font-size="12" font-family="system-ui">c < 1 → 信号消失 🫥</text>
  <text x="350" y="275" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">c = 1 → 方差守恒 ✓</text>
</svg>

## Xavier 初始化：第一个优雅的解

### 2010 年的突破

2010 年，Xavier Glorot 和 Yoshua Bengio 发表了一篇里程碑论文："Understanding the Difficulty of Training Deep Feedforward Neural Networks"。他们做了一件看起来很简单但无人做过的事：**认真追踪信号方差在每一层是怎么变化的。**

他们发现了当时网络难以训练的直接原因：使用 sigmoid/tanh 激活函数时，如果权重初始化不当，深层的激活值会全部挤到饱和区（接近 0 或 ±1），梯度几乎为零，训练停滞。

### 核心直觉：信号既不能放大也不能缩小

Xavier 的想法很直觉：**让信号在每一层通过时保持"原来的音量"**。

想象你在一条很长的走廊里说话，走廊里每隔几米有一面墙（代表网络的每一层）。如果每面墙会把声音放大一点点——到了走廊尽头，你的耳朵会被震聋（梯度爆炸）。如果每面墙吸收一点声音——到了尽头你什么都听不到（梯度消失）。理想情况是每面墙既不放大也不缩小声音。

但这里有一个微妙的问题：我们不仅要让信号在**前向传播**时保持方差，还要让梯度在**反向传播**时保持方差。

前向传播的条件是 $\text{Var}(W) = 1/d_{in}$（fan-in）。

反向传播的条件呢？梯度从输出往回传时，经过的运算是 $\nabla_x \mathcal{L} = W^T \nabla_y \mathcal{L}$。类似的分析给出条件 $\text{Var}(W) = 1/d_{out}$（fan-out）。

两个条件不能同时满足（除非 $d_{in} = d_{out}$）。Xavier 的解决方案是**取折中**：

$$\text{Var}(W) = \frac{2}{d_{in} + d_{out}}$$

如果用均匀分布，就是从 $U\left[-\sqrt{\frac{6}{d_{in}+d_{out}}},\ \sqrt{\frac{6}{d_{in}+d_{out}}}\right]$ 中采样。

### 为什么它对 Sigmoid/Tanh 有效

Xavier 初始化隐含了一个假设：**激活函数在工作区间内近似线性**。对于 sigmoid 和 tanh，在零点附近确实近似线性（tanh 在 0 点的导数恰好是 1）。所以 Xavier 的方差分析对这些激活函数是准确的。

但 2012 年之后，一个新的激活函数横空出世，彻底打破了 Xavier 的假设。

## Kaiming 初始化：为 ReLU 而生

### ReLU 的"一半"问题

ReLU（Rectified Linear Unit）的定义极其简单：$\text{ReLU}(x) = \max(0, x)$。它把所有负值都变成零。

这对方差分析意味着什么？如果输入是均值为零的正态分布，ReLU 会把**一半的值**（负数部分）直接砍掉。这相当于输出的方差变成了输入方差的一半：

$$\text{Var}(\text{ReLU}(x)) = \frac{1}{2}\text{Var}(x)$$

如果你仍然用 Xavier 初始化，每经过一层 ReLU，信号的方差就减半。20 层之后，信号只剩 $0.5^{20} ≈ 0.000001$ 倍——彻底消失了。

### He（何恺明）的修正

2015 年，何恺明（Kaiming He）在论文 "Delving Deep into Rectifiers" 中给出了一个简洁优雅的修正：**既然 ReLU 砍掉一半方差，那初始化时就多给两倍方差来补偿。**

$$\text{Var}(W) = \frac{2}{d_{in}}$$

为什么是 $2/d_{in}$ 而不是 $2/(d_{in}+d_{out})$？因为对于 ReLU 网络，实践表明只保证前向传播的方差守恒（fan-in 模式）效果已经很好。当然也可以用 fan-out 模式 $2/d_{out}$ 来保证反向传播。

这个看似微小的改动——把分子从 1 变成 2——让何恺明成功训练了当时前所未有的 30 层 CNN，并在 ImageNet 上首次超越了人类水平的图像识别精度。

### 推广到 Leaky ReLU

如果激活函数是 Leaky ReLU: $f(x) = \max(\alpha x, x)$（其中 $\alpha$ 通常是 0.01），那么负半部分不是完全砍掉，而是乘以 $\alpha$。方差的保留比例变成 $(1 + \alpha^2)/2$，对应的初始化是：

$$\text{Var}(W) = \frac{2}{(1+\alpha^2) \cdot d_{in}}$$

当 $\alpha = 0$ 时退化为标准 ReLU 的 Kaiming 初始化。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">三种初始化方案对比</text>
  <!-- Xavier box -->
  <rect x="20" y="50" width="200" height="120" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="120" y="75" text-anchor="middle" fill="#6e8eff" font-size="13" font-weight="bold" font-family="system-ui">Xavier / Glorot (2010)</text>
  <text x="120" y="100" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Var(W) = 2/(n_in + n_out)</text>
  <text x="120" y="120" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">适用：Sigmoid / Tanh</text>
  <text x="120" y="140" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">假设：激活函数线性</text>
  <text x="120" y="160" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">折中前向+反向</text>
  <!-- Kaiming box -->
  <rect x="250" y="50" width="200" height="120" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="350" y="75" text-anchor="middle" fill="#34d399" font-size="13" font-weight="bold" font-family="system-ui">Kaiming / He (2015)</text>
  <text x="350" y="100" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Var(W) = 2/n_in</text>
  <text x="350" y="120" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">适用：ReLU / Leaky ReLU</text>
  <text x="350" y="140" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">补偿 ReLU 砍掉的一半</text>
  <text x="350" y="160" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">只看前向 (fan-in)</text>
  <!-- muP box -->
  <rect x="480" y="50" width="200" height="120" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="580" y="75" text-anchor="middle" fill="#a78bfa" font-size="13" font-weight="bold" font-family="system-ui">µP (2022)</text>
  <text x="580" y="100" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Var(W) = σ²_base / m_d</text>
  <text x="580" y="120" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">适用：任意宽度缩放</text>
  <text x="580" y="140" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">+ 学习率缩放 η/m_d</text>
  <text x="580" y="160" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">前向+反向+更新全控制</text>
  <!-- Timeline arrow -->
  <line x1="50" y1="210" x2="650" y2="210" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <!-- Timeline dots -->
  <circle cx="120" cy="210" r="5" fill="#6e8eff"/>
  <text x="120" y="235" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">2010</text>
  <text x="120" y="252" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">方差守恒</text>
  <circle cx="350" cy="210" r="5" fill="#34d399"/>
  <text x="350" y="235" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">2015</text>
  <text x="350" y="252" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">ReLU 修正</text>
  <circle cx="580" cy="210" r="5" fill="#a78bfa"/>
  <text x="580" y="235" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">2022</text>
  <text x="580" y="252" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">训练动态全控制</text>
  <!-- Bottom insight -->
  <text x="350" y="290" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">核心进化：从"只管开始"到"管好整个训练过程"</text>
  <text x="350" y="310" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">初始化 → 初始化 + 宽度感知 → 初始化 + 学习率 + 宽度迁移</text>
</svg>

## 但 Xavier/Kaiming 还不够：训练动态的问题

### 初始化只解决了第一步

Xavier 和 Kaiming 解决了一个关键问题：让网络在**第 0 步**（训练开始前）有合理的信号传播。但训练不只是第 0 步——一旦梯度开始更新权重，新的问题出现了：

**权重更新的幅度对不同宽度的模型意味着什么？**

设想你在调一个 128 维隐藏层的小模型，找到最优学习率是 $\eta = 0.001$。现在你把模型放大到 4096 维隐藏层——同样的学习率还能用吗？

答案是：在标准参数化（Standard Parameterization, SP）下，**不能**。因为当宽度增大时，权重更新对激活值的影响也在变化。GPT-3、LLaMA 等大模型的训练者们通过昂贵的反复试错，经验性地发现了"模型越大，学习率越小"的规律。但能不能从数学上精确地告诉我们该怎么调？

### 尺度迁移的梦想

如果你能用一个 1 亿参数的小模型做超参数搜索（花几个 GPU 小时），然后把找到的最优超参数**直接**用到 700 亿参数的大模型上——这将节省数百万美元的计算成本。

这正是 µP（Maximal Update Parameterization）要解决的问题。

## µP：让超参数跨尺度迁移

### 从"保持方差"到"保持训练动态"

µP 是 Greg Yang（先后在微软研究院和 xAI 工作）通过一系列 "Tensor Programs" 论文发展出来的理论框架。它的核心思想可以用一句话概括：

> **不仅要让激活值的大小独立于模型宽度，还要让梯度和权重更新的效果也独立于模型宽度。**

Xavier/Kaiming 只管了第一个条件（前向传播方差守恒）。µP 同时管三件事：
1. **前向传播**：激活值的大小不随宽度变化
2. **反向传播**：梯度的大小不随宽度变化
3. **权重更新**：学习率的效果不随宽度变化

### 直觉：为什么学习率需要调整？

让我们用一个具体例子理解第三个条件。

考虑一个全连接层 $y = xW$，其中 $x$ 是 $d$ 维输入，$W$ 是 $d \times d$ 的权重矩阵。

训练一步后，权重变成 $W + \Delta W$，输出变成：
$$y' = x(W + \Delta W) = xW + x\Delta W$$

新的那一项 $x\Delta W$ 就是权重更新对输出的影响。它有多大？

$x\Delta W$ 本质上是一个点积求和：$x$ 的 $d$ 个元素分别乘以 $\Delta W$ 的对应列元素再相加。根据大数定律，当 $d$ 很大时，这个和会**稳定地**趋向其期望值，而且大小正比于 $d$。

所以如果你把宽度从 $d_{base}$ 放大到 $m_d \cdot d_{base}$（$m_d$ 是宽度倍数），权重更新对输出的影响会变大 $m_d$ 倍。为了抵消这个效应，学习率需要除以 $m_d$：

$$\eta_{\mu P} = \frac{\eta_{base}}{m_d}$$

### µP 的完整配方

对于一个 Transformer 模型，µP 的改动归结为一张简洁的表格：

| 组件 | 标准参数化 (SP) | µP |
|------|----------------|-----|
| 隐藏层初始化方差 | $\sigma^2_{base}$ | $\sigma^2_{base} / m_d$ |
| 隐藏层学习率 (Adam) | $\eta_{base}$ | $\eta_{base} / m_d$ |
| 输出 logit 前向 | $x W_{emb}^T$ | $x W_{emb}^T / m_d$ |
| Attention logits | $Q^T K / \sqrt{d_{head}}$ | $Q^T K / d_{head}$ |
| Embedding 初始化/学习率 | 不变 | 不变 |

其中 $m_d = d / d_{base}$ 是宽度相对于基线模型的放大倍数。

### 为什么 Attention 的缩放变了？

一个有趣的细节：标准 Transformer 用 $1/\sqrt{d_{head}}$ 缩放 attention logits，而 µP 改用 $1/d_{head}$。为什么？

在训练初期，Q 和 K 是随机的、不相关的。这时 $1/\sqrt{d}$ 的缩放是对的（就像 Xavier 初始化的逻辑）。但随着训练进行，**Q 和 K 会逐渐对齐**——它们不再是独立随机变量了。对齐意味着点积的期望值不再是零，而是正比于维度 $d$。这时需要 $1/d$ 来抵消这个增长。

µP 选择从一开始就用 $1/d$，确保整个训练过程中 attention logits 的尺度都是受控的。

### 实际验证：Coordinate Check

怎么验证你的 µP 实现是正确的？EleutherAI 和 Cerebras 推荐一个简单的测试：

1. 用不同宽度（256, 512, 1024, 2048...）训练同一个模型 10 步
2. 记录每步各层激活值的平均绝对值
3. 如果实现正确，**不同宽度的曲线应该重叠**

在标准参数化下，宽度越大的模型，激活值越大——曲线会扇形散开。这就是为什么大模型需要更小的学习率。在 µP 下，所有宽度的曲线聚拢在一起，意味着小模型的最优超参数可以直接迁移到大模型。

### µP 的实际收益

Greg Yang 等人的实验显示：
- 在 40M 参数的代理模型上做 200 次随机超参数搜索
- 把找到的最优超参数直接用到 GPT-3 6.7B 上
- 达到了 GPT-3 13B 的性能——相当于 **2 倍计算效率提升**

Cerebras 团队用 111M 代理模型的超参数训练 3B 模型，达到了同时期 7B 模型的性能，节省了 3.3 倍训练 FLOP。

## 现代 LLM 中的初始化实践

### GPT 系列的选择

现代大语言模型（GPT-4、LLaMA、Claude 等）通常使用：
- **RMSNorm 预归一化**：每层之前做归一化，减轻对初始化的依赖
- **Kaiming 风格初始化**：隐藏层用 $\mathcal{N}(0, \sqrt{2/d})$ 或类似方案
- **输出层缩小**：最后一层初始化方差除以 $\sqrt{2L}$（$L$ 是层数），防止残差流方差随深度增长
- **残差连接**：提供"梯度高速公路"，进一步缓解梯度消失

### 为什么残差连接如此重要？

即使初始化完美，100+ 层的网络仍然可能出问题。残差连接 $x_{l+1} = x_l + f(x_l)$ 的存在意味着梯度可以直接"跳过"任意多层，不经过任何权重矩阵乘法。这从根本上绕开了梯度消失的连乘效应。

（关于残差连接的详细分析，参见本系列第 12 篇。）

### DeepNorm 和深度缩放

微软提出的 DeepNorm 进一步解决了**极深**模型（1000+ 层）的训练稳定性：

$$\text{DeepNorm}(x) = \text{LayerNorm}(\alpha \cdot x + f(x))$$

其中 $\alpha > 1$ 是一个与深度相关的缩放因子（大约 $(2L)^{1/4}$），配合更小的初始化方差 $\beta$。核心思想是：**当模型极深时，让残差分支的贡献相对变小，信息主要走"跳跃连接"的主路径。**

## 这意味着什么

让我们回顾整个故事线：

**2010 年之前**：随机初始化，深层网络几乎不可训练。人们以为是算法的问题，实际上是初始化的问题。

**2010 年 Xavier**：第一次从数学上理解了为什么——方差在层间累乘会指数增长或衰减。解决方案：让方差每层保持不变。

**2015 年 Kaiming**：ReLU 打破了线性假设，需要额外的 ×2 补偿。这个小修正让 30 层网络首次从零训练成功。

**2022 年 µP**：初始化不够，还需要控制训练过程中的动态。通过精确的宽度缩放规则，让超参数可以从小模型迁移到大模型，节省数百万美元的调参成本。

初始化理论的发展揭示了一个深刻的教训：**深度学习中很多看似"不可能"的困难，其实是数学细节上的失误。一旦你理解了信号传播的数学，解决方案往往惊人地简洁。**

## 下一篇预告

我们讨论了权重的初始化，但训练时权重是怎么被更新的？为什么 Adam 几乎统治了 LLM 训练？它比 SGD 好在哪里——又有什么隐藏的问题？下一篇我们深入 Adam/AdamW 优化器的数学直觉。（已发布，见本系列第 1 篇。）

---

**参考来源：**
- Glorot & Bengio, "Understanding the Difficulty of Training Deep Feedforward Neural Networks" (2010)
- He et al., "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet" (2015)
- Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2022)
- Dey, Anthony & Hestness, "The Practitioner's Guide to the Maximal Update Parameterization" (EleutherAI/Cerebras, 2024)
