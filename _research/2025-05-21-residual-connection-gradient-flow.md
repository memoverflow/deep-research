---
title: "残差连接：深度网络的梯度高速公路"
date: 2025-05-21
level: 3
series: "LLM 原理深度解析"
series_order: 12
series_total: 37
tags: [residual-connection, gradient-flow, transformer, deep-learning, resnet]
summary: "残差连接看似只是一条「跳线」，实则彻底改变了深度网络的优化景观——它让梯度拥有了不经过任何非线性的高速通道，让网络从「学变换」变成「学修正」，甚至暗含了指数级路径集成和连续微分方程的深刻数学。"
---

# 残差连接：深度网络的梯度高速公路

> 一根简单的「跳线」，为什么能让网络从 20 层训到 1000 层？

## 故事从一个反直觉的实验开始

2015 年，何恺明在 ImageNet 上做了一个实验，结果让整个深度学习社区困惑：一个 56 层的网络，表现竟然**不如** 20 层的网络。

这完全违反直觉。更深的网络理论上拥有更强的表达能力——一个 56 层网络能表达的函数，20 层网络不一定能。那为什么更深反而更差？

答案不是过拟合。56 层网络在**训练集**上的 loss 就比 20 层的高。这意味着问题不在泛化，而在优化——网络有能力表达更好的解，但优化器找不到那个解。

这就是所谓的「退化问题」（degradation problem）。注意它和梯度消失不完全一样：梯度消失是梯度变为零导致参数不更新；退化是即使梯度没有消失，优化器也陷在了一个次优解里。

何恺明的解决方案惊人地简单：**加一条跳线，把输入直接加到输出上。** 就这么一个改动，网络从 20 层成功训到了 152 层，然后是 1000 层，最终 Transformer 模型动辄近百层也能稳定训练。

这根跳线为什么有这么大的魔力？

## 从「学变换」到「学修正」：思维范式的转变

### 传统网络的困境

在没有残差连接的传统网络中，每一层要学习的是一个完整的变换函数 $H(\mathbf{x})$。比如你希望第 10 层的输出是某个特定表示，那第 10 层就得从头学会怎么把输入映射到那个表示。

打个比方：假设你在翻译一篇文章，每一层相当于一个翻译员。传统网络的做法是，每个翻译员拿到的是白纸，必须独立完成完整翻译。

### 残差连接的巧思

残差连接改变了学习目标。它把每一层的输出定义为：

$$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$$

其中 $\mathbf{x}$ 是输入（直接传过来），$F(\mathbf{x})$ 是这一层要学的「残差」。

翻译回人话：**每一层不再需要学完整的变换，只需要学「在输入基础上改什么」。** 

回到翻译的比喻：现在每个翻译员拿到的是上一位的译文，只需要做修改和润色。从零写一篇翻译很难，但在别人的基础上改改措辞、修修语法，就容易得多。

这个转变有一个深刻的后果：**如果某一层发现自己什么都不需要做，它只要让 $F(\mathbf{x}) = 0$ 就行。** 把一堆权重推向零比让一堆权重精确学出恒等映射要容易得多。这意味着网络可以「自动跳过」不需要的层，深度不再是负担。

## 梯度高速公路：为什么跳线拯救了反向传播

### 没有跳线时梯度怎么死的

要理解残差连接的威力，我们需要看看反向传播时梯度发生了什么。

在传统网络中，梯度从最后一层往回传，经过每一层时要乘以那一层的雅可比矩阵。假设网络有 $L$ 层，损失对第 $l$ 层输入的梯度是：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \prod_{i=l}^{L-1} \frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i}$$

这是一连串矩阵的乘积。如果每个矩阵的特征值大多小于 1，连乘的结果就指数级趋近于零——梯度消失了。如果大于 1，就指数级爆炸。

想象一下：你在玩传话游戏，每一轮信息都要经过一个人的「处理」（可能衰减也可能放大）。经过 50 轮后，原始信息要么消失殆尽，要么面目全非。

### 加了跳线后的魔法

有了残差连接，情况完全不同。由于 $\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\mathbf{x}_l)$，梯度变成：

$$\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} = \mathbf{I} + \frac{\partial F_l(\mathbf{x}_l)}{\partial \mathbf{x}_l}$$

这里的关键是那个**恒等矩阵 $\mathbf{I}$**。不管 $F_l$ 的梯度有多小，恒等矩阵始终保证梯度至少有一条「直通道路」可以不经任何衰减地传回去。

把多层串起来，损失对第 $l$ 层的梯度可以展开为：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \prod_{i=l}^{L-1}\left(\mathbf{I} + \frac{\partial F_i}{\partial \mathbf{x}_i}\right)$$

展开这个连乘，你会得到 $2^{L-l}$ 个求和项。其中有一项是纯粹的恒等——梯度从第 $L$ 层一路不变地传到第 $l$ 层。其他项分别经过 1 层、2 层、……、$L-l$ 层的变换。

**翻译成人话：** 梯度不再只有一条路可走（必须穿越所有层），而是拥有了指数级数量的路径——有些很长，有些很短。即使长路径的梯度消失了，短路径的梯度依然健在。这就是「梯度高速公路」的本质。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#34d399"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="22" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">传统网络 vs 残差网络的梯度路径</text>
  
  <!-- Traditional Network (top) -->
  <text x="80" y="55" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">传统网络：唯一路径</text>
  <rect x="30" y="65" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="75" y="89" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 1</text>
  <line x1="120" y1="85" x2="170" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <rect x="170" y="65" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="215" y="89" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 2</text>
  <line x1="260" y1="85" x2="310" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <rect x="310" y="65" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="355" y="89" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 3</text>
  <line x1="400" y1="85" x2="450" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <rect x="450" y="65" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="495" y="89" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 4</text>
  <line x1="540" y1="85" x2="590" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <rect x="590" y="65" width="70" height="40" rx="8" fill="#1e1e2a" stroke="#ef4444" stroke-width="1.5"/>
  <text x="625" y="89" text-anchor="middle" fill="#ef4444" font-size="11" font-family="system-ui">∂L/∂x₁≈0</text>
  
  <!-- Residual Network (bottom) -->
  <text x="80" y="160" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">残差网络：指数级路径</text>
  <rect x="30" y="170" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="75" y="194" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 1</text>
  <rect x="170" y="170" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="215" y="194" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 2</text>
  <rect x="310" y="170" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="355" y="194" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 3</text>
  <rect x="450" y="170" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="495" y="194" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Layer 4</text>
  <rect x="590" y="170" width="70" height="40" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="625" y="194" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">∂L/∂x₁✓</text>
  
  <!-- Skip connections (arcs) -->
  <line x1="120" y1="190" x2="170" y2="190" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="260" y1="190" x2="310" y2="190" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="400" y1="190" x2="450" y2="190" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="540" y1="190" x2="590" y2="190" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  
  <!-- Direct highway path (green) -->
  <path d="M 75 210 Q 75 260 215 260 Q 355 260 495 260 Q 580 260 625 210" stroke="#34d399" stroke-width="2" fill="none" stroke-dasharray="6,3" marker-end="url(#arrow-green)"/>
  <text x="350" y="280" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">恒等直通路径（梯度无衰减）</text>
  
  <!-- Short paths -->
  <path d="M 75 210 Q 75 235 145 237 Q 215 240 215 210" stroke="#a78bfa" stroke-width="1.2" fill="none" opacity="0.6"/>
  <path d="M 215 210 Q 215 240 355 240 Q 495 240 495 210" stroke="#a78bfa" stroke-width="1.2" fill="none" opacity="0.6"/>
  <text x="350" y="305" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui" opacity="0.8">+ 多条短路径（经过 1-2 层变换）</text>
</svg>

## 隐式集成：一个网络 = 2^L 个子网络

### Veit 的洞察

2016 年，Andreas Veit 提出了一个精彩的观察：一个 $L$ 层的残差网络，本质上是 $2^L$ 个不同深度子网络的**隐式集成**。

怎么理解？把残差网络的计算展开：

$$\mathbf{x}_L = \mathbf{x}_0 + F_1(\mathbf{x}_0) + F_2(\mathbf{x}_0 + F_1(\mathbf{x}_0)) + \cdots$$

虽然写起来是串行的，但每一层的残差分支都有「用」和「不用」两种选择。3 层网络就有 $2^3 = 8$ 条路径：不经过任何层、只经过第 1 层、只经过第 2 层、只经过第 3 层、经过 1 和 2、经过 1 和 3、经过 2 和 3、经过所有层。

**这意味着残差网络天生自带「集成学习」效果。** 而 Veit 通过实验发现，训练时网络主要依赖那些较短的路径（经过的层数少的）。删掉单独一层几乎不影响性能，但同时删掉多层影响就大了——这正是集成模型的行为特征。

### 为什么这解释了深度网络为什么能训

传统观点认为 100 层网络需要梯度穿越 100 层才能学习，这很难。但集成视角告诉我们：**网络实际上从大量较浅的「有效路径」中学习。** 一个 100 层的残差网络有效路径长度的分布集中在 50 层左右（二项分布），大部分梯度贡献来自中等长度的路径。

这也解释了为什么残差网络对层的删除具有鲁棒性：删掉一层只是去掉了包含该层的那些路径（占一半），其他路径完全不受影响。

## Transformer 中的「残差流」

### 从跳线到信息总线

在 Transformer 中，残差连接不仅仅是梯度辅助——它们构成了整个模型的「通信骨干」。机械可解释性（Mechanistic Interpretability）领域把它称为**残差流**（residual stream）。

想象一条河流从第一层流到最后一层。每个 Attention 层和 FFN 层就像河流两岸的工厂——它们从河里取水（读取信息）、加工后倒回河中（写入信息）。河水本身始终在流动，任何下游工厂都能看到所有上游工厂的产出。

这个视角有一个重要推论：**每一层不必按顺序建立在上一层的输出之上。** 第 50 层可以直接读取第 3 层写入的信息，因为那个信息一直保持在残差流中。这就是为什么 Transformer 能学到跨越很多层的复杂回路（circuit）。

### Pre-Norm vs Post-Norm：跳线放哪儿很重要

残差连接和 Layer Normalization 的相对位置，对梯度流有质的影响。

**Post-Norm**（原始 Transformer 的做法）：
$$\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l + F_l(\mathbf{x}_l))$$

**Pre-Norm**（GPT-2 之后的主流做法）：
$$\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\text{LN}(\mathbf{x}_l))$$

区别在哪？看梯度。Pre-Norm 中，残差连接的直通路径是**干净的恒等映射**——梯度传回时不经过任何归一化层。而 Post-Norm 中，归一化套在残差外面，直通路径上多了一个 LN 的雅可比矩阵，梯度信号会被 LN 调制。

实践中的后果很明显：Post-Norm 训练不稳定，需要 learning rate warmup 来避免早期崩溃；Pre-Norm 几乎随便训，但最终性能可能略差（因为梯度太均匀，深层和浅层收到的更新差不多）。

<svg viewBox="0 0 680 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
    <marker id="arrow2g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#34d399"/>
    </marker>
  </defs>
  <text x="170" y="20" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">Post-Norm</text>
  <text x="510" y="20" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">Pre-Norm</text>
  
  <!-- Post-Norm -->
  <rect x="40" y="35" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="70" y="55" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">x_l</text>
  <line x1="70" y1="65" x2="70" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  
  <!-- F block -->
  <rect x="40" y="90" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="70" y="110" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">F(x_l)</text>
  <line x1="70" y1="120" x2="70" y2="140" stroke="#6e8eff" stroke-width="1.5"/>
  
  <!-- Add -->
  <circle cx="70" cy="150" r="12" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="70" y="154" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">+</text>
  
  <!-- Skip connection -->
  <path d="M 100 50 L 130 50 L 130 150 L 82 150" stroke="#34d399" stroke-width="1.5" fill="none" marker-end="url(#arrow2g)"/>
  
  <line x1="70" y1="162" x2="70" y2="180" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  
  <!-- LN after -->
  <rect x="40" y="185" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="70" y="205" text-anchor="middle" fill="#f59e0b" font-size="10" font-family="system-ui">LN</text>
  <line x1="70" y1="215" x2="70" y2="240" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  
  <rect x="40" y="245" width="60" height="25" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="70" y="262" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">x_{l+1}</text>
  
  <text x="170" y="265" text-anchor="middle" fill="#ef4444" font-size="10" font-family="system-ui">梯度必经 LN ⚠️</text>
  
  <!-- Divider -->
  <line x1="330" y1="30" x2="330" y2="270" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
  
  <!-- Pre-Norm -->
  <rect x="380" y="35" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="410" y="55" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">x_l</text>
  <line x1="410" y1="65" x2="410" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  
  <!-- LN first -->
  <rect x="380" y="90" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="410" y="110" text-anchor="middle" fill="#f59e0b" font-size="10" font-family="system-ui">LN</text>
  <line x1="410" y1="120" x2="410" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  
  <!-- F block -->
  <rect x="380" y="145" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="410" y="165" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">F(LN(x))</text>
  <line x1="410" y1="175" x2="410" y2="195" stroke="#6e8eff" stroke-width="1.5"/>
  
  <!-- Add -->
  <circle cx="410" cy="205" r="12" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="410" y="209" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">+</text>
  
  <!-- Skip connection (clean!) -->
  <path d="M 440 50 L 470 50 L 470 205 L 422 205" stroke="#34d399" stroke-width="2" fill="none" marker-end="url(#arrow2g)"/>
  <text x="510" y="130" text-anchor="start" fill="#34d399" font-size="10" font-family="system-ui">干净直通 ✓</text>
  
  <line x1="410" y1="217" x2="410" y2="245" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="380" y="245" width="60" height="25" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="410" y="262" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">x_{l+1}</text>
  
  <text x="510" y="265" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">梯度直通无阻 ✓</text>
</svg>

## 深层缩放：当 1000 层不再是梦

### 方差累积问题

残差连接虽然保证了梯度流，但有一个新问题：每加一层残差分支，输出的方差就增大一些。经过 $L$ 层后，残差流的方差大致与 $L$ 成正比——100 层的模型，其残差流的振幅是单层输出的 10 倍。

这在几十层时还可控，但到了几百层，累积的方差会让 Attention 的 softmax 输入过大（产生极端的一个 token 独占所有注意力的现象），训练变得不稳定。

### DeepNet 的解法：残差缩放

2022 年，微软的 DeepNet 论文提出了一个优雅的方案：在残差分支的输出乘以一个与深度相关的缩放因子 $\alpha$：

$$\mathbf{x}_{l+1} = \alpha \cdot \mathbf{x}_l + F_l(\mathbf{x}_l)$$

其中 $\alpha$ 随着网络变深而增大（对 encoder 取 $(2N)^{1/4}$，$N$ 是层数），同时初始化时将残差分支的权重缩小为 $\beta$（取 $(8N)^{-1/4}$）。

直觉：**既然每一层的残差贡献会累积，那就在初始化时让每一层的贡献更小，同时放大恒等路径，确保信号不会在早期训练中被残差分支的随机噪声淹没。**

DeepNet 用这个方法成功训练了 1000 层的 Transformer——500 层 encoder + 500 层 decoder。

### GPT 系列的简化做法

OpenAI 在 GPT-2 中采用了更简单的策略：在每个残差分支的最后一个线性层，初始化权重时除以 $\sqrt{2N}$（$N$ 是总层数）。这确保在初始化时，所有残差分支的方差贡献加起来大致等于 1。

这个技巧看似简单，但没有它，训练超过 48 层的 GPT 就会变得不稳定。

## 连续极限：残差网络即微分方程

### 从离散到连续的视角

如果把残差网络的层间步长看作一个小量 $\Delta t$，那么：

$$\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\mathbf{x}_l) \approx \mathbf{x}(t) + F(\mathbf{x}(t), t) \cdot \Delta t$$

当层数趋于无穷、步长趋于零时，这就是一个**常微分方程**（ODE）：

$$\frac{d\mathbf{x}}{dt} = F(\mathbf{x}(t), t)$$

这个洞察催生了 2018 年 NeurIPS 最佳论文「Neural ODEs」：残差网络的连续极限是一个神经常微分方程，其中「深度」对应连续的「时间」。

这不只是数学上的巧合。ODE 视角提供了几个实用洞见：

1. **自适应深度**：ODE 求解器可以根据输入的复杂度自动选择步数——简单输入用几步就够，复杂输入多算几步
2. **理论保证**：ODE 理论告诉我们什么样的 $F$ 能保证存在唯一解（Lipschitz 连续性）
3. **正则化**：加速方法、随机深度（Stochastic Depth）等技巧都能从 ODE 角度自然推导

### 为什么 Transformer 的残差层间隔是 1？

有趣的是，标准 Transformer 的残差步长固定为 1（每层完整地加回残差），而没有用更小的步长（比如乘以 0.1）。从 ODE 角度看，这相当于用欧拉法做数值积分，步长 $\Delta t = 1$。

这其实不太「优雅」——大步长的欧拉法精度很差。但在实践中，因为每层的 $F$ 是独立参数化的（不共享权重），所以它可以自己学会调整输出的幅度。DeepNet 的缩放因子可以看作显式引入了更灵活的步长控制。

## 实验数据：残差连接到底带来了多少改善

让我们用具体数字来感受残差连接的效果：

| 设置 | 可训练深度 | 训练稳定性 |
|------|-----------|-----------|
| 无残差连接 | ~20 层 | Loss 经常不收敛 |
| 标准残差连接 | ~100 层 | 需要 warmup |
| 残差连接 + Pre-Norm | ~200 层 | 稳定，无需特殊技巧 |
| DeepNet (残差缩放) | 1000 层 | 稳定 |

在 Transformer 中的具体表现：
- GPT-3（96 层）：使用 Pre-Norm + 初始化缩放
- LLaMA（80 层）：Pre-Norm + RMSNorm
- DeepSeek-V2（60 层）：带改进的残差连接设计

没有残差连接，这些模型根本无法训练。

## 这意味着什么

残差连接看似是深度学习中最简单的技巧之一——就是一个加法。但它的影响是根本性的：

1. **优化景观**：从学习完整变换变成学习微小修正，loss landscape 变得更平滑
2. **梯度流**：提供指数级路径集成，彻底解决了深度网络的梯度问题
3. **模块性**：每层可以独立地「选择」贡献多少，坏层自动被跳过
4. **信息架构**：在 Transformer 中构建了全局信息总线，使跨层通信成为可能
5. **理论优雅**：连接了离散网络和连续动力系统，打开了用微分方程理论分析深度学习的大门

如果没有 2015 年何恺明的那篇论文，今天的大语言模型——动辄 80-100 层的 Transformer——可能根本不会存在。残差连接不只是一个「训练技巧」，它是现代深度学习架构的基石。

## 下一篇预告

我们已经知道残差连接让信息自由流动，归一化让信号保持稳定。但网络里还有一个关键组件我们还没深入讨论过：**Embedding 层**。token 是怎么变成向量的？这些向量在高维空间里形成什么样的几何结构？词义相近的 token 真的靠得更近吗？下一篇我们来探索 Embedding 的几何世界。
