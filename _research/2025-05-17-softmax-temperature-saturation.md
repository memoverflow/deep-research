---
title: "Softmax 的秘密：温度、饱和与数值稳定性"
date: 2025-05-17
level: 3
series: "LLM 原理深度解析"
series_order: 4
series_total: 4
tags: [softmax, temperature, numerical-stability, attention, transformer]
summary: "Softmax 看似简单——exp 再归一化——但它在 Transformer 里引发的连锁反应远比你想象的深刻：从梯度消失到注意力坍缩，从数值爆炸到知识蒸馏。"
---

# Softmax 的秘密：温度、饱和与数值稳定性

> Softmax 可能是深度学习里最被低估的函数。它只有一行公式，却决定了模型"看向哪里"、"说什么话"、以及训练能不能收敛。

## 故事从一个 bug 开始

假设你正在从零实现一个 Transformer。你写下了注意力的核心计算：算出 Q 和 K 的点积，然后过一个 softmax，再乘以 V。代码看起来完全正确，但训练到一半，loss 突然变成 NaN。

你检查了半天，最终发现问题出在 softmax 上——当输入数值太大时，`exp(1000)` 直接溢出到正无穷。或者另一种情况：所有注意力权重都集中在一个 token 上，梯度几乎为零，模型停止学习。

这两个问题——数值不稳定和梯度饱和——都指向同一个看似人畜无害的函数：softmax。今天我们要彻底搞明白它。

## Softmax 到底在做什么？

### 问题：怎么把任意数字变成概率？

模型的输出是一组"原始分数"（logits）——可能是 `[2.0, 1.0, 0.1]`，也可能是 `[1000, 999, -500]`。我们需要一个函数，把这些分数变成"概率分布"：每个值都是正的，加起来等于 1。

最直觉的想法是：每个数除以总和。但如果分数有负数呢？负数不能当概率。

### 核心直觉：先用 exp 把所有数变正，再归一化

Softmax 的想法非常朴素：

1. 先对每个分数取指数（$e^{z_i}$），这保证了结果是正数
2. 再除以所有指数之和，保证加起来等于 1

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

为什么用 $e^x$ 而不是别的函数（比如 $2^x$ 或 $x^2$）？因为指数函数有一个优美的性质：**它保持排序，并且差距被放大**。如果 $z_1 > z_2$，那么 $e^{z_1}$ 和 $e^{z_2}$ 的比值比 $z_1$ 和 $z_2$ 的差距更大。这意味着 softmax 天然倾向于"突出赢家"。

你可以把它想象成一场投票：每个候选者的得票数是 $e^{分数}$，然后按得票比例分配权重。分数高的候选者不是线性领先，而是**指数级**领先。

### 统计力学的渊源

有趣的是，softmax 不是机器学习发明的。它就是统计力学中的 **Boltzmann 分布**（也叫 Gibbs 分布）：

$$P(状态_i) = \frac{e^{-E_i / kT}}{\sum_j e^{-E_j / kT}}$$

在物理学中，一个系统更倾向于待在低能量的状态，$T$ 是温度。温度越高，各状态概率越均匀（热力学平衡）；温度趋向零，系统冻结在最低能量状态。

这直接引出了我们下一个话题——温度。

## 温度：旋钮背后的数学

### 问题：模型的输出太"确定"或太"随机"怎么办？

当你用 ChatGPT 时，有时候你希望它创意一点（写诗、编故事），有时候你希望它精确一点（回答数学题）。但模型训练完后，它的"确定程度"是固定的——该怎么外部调节？

### 核心直觉：给 softmax 加一个"温度旋钮"

做法极其简单：在过 softmax 之前，把所有 logits 除以一个温度参数 $T$：

$$\text{softmax}(z_i / T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

这个 $T$ 就是"温度"。我们来直觉理解三种极端情况：

**$T \to 0$（冰点）**：除以一个很小的数 = 所有分数被放大很多倍。比如 `[3, 1, 0.5]` 除以 0.01 变成 `[300, 100, 50]`。经过 softmax 后，最大值几乎独占所有概率。这就是 **argmax**——硬选择，完全确定。

**$T = 1$（室温）**：不做任何缩放，标准 softmax。

**$T \to \infty$（炼炉）**：除以一个很大的数 = 所有分数被压成接近零。比如 `[3, 1, 0.5]` 除以 1000 变成 `[0.003, 0.001, 0.0005]`。经过 softmax 后，每个选项概率接近均匀分布 $1/K$。完全随机。

<svg viewBox="0 0 650 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="325" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">温度对 Softmax 输出分布的影响</text>
  <!-- T→0 box -->
  <rect x="20" y="45" width="180" height="200" rx="8" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="110" y="70" text-anchor="middle" fill="#ff6b6b" font-size="13" font-family="system-ui" font-weight="bold">T → 0（冰点）</text>
  <!-- Bar chart for T→0 -->
  <rect x="50" y="85" width="30" height="140" rx="3" fill="#ff6b6b" opacity="0.8"/>
  <rect x="95" y="215" width="30" height="10" rx="3" fill="#ff6b6b" opacity="0.3"/>
  <rect x="140" y="220" width="30" height="5" rx="3" fill="#ff6b6b" opacity="0.2"/>
  <text x="65" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">A</text>
  <text x="110" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">B</text>
  <text x="155" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">C</text>
  <text x="110" y="258" text-anchor="middle" fill="#999" font-size="10" font-family="system-ui">≈ argmax</text>
  <!-- T=1 box -->
  <rect x="230" y="45" width="180" height="200" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="320" y="70" text-anchor="middle" fill="#6e8eff" font-size="13" font-family="system-ui" font-weight="bold">T = 1（标准）</text>
  <!-- Bar chart for T=1 -->
  <rect x="260" y="110" width="30" height="115" rx="3" fill="#6e8eff" opacity="0.8"/>
  <rect x="305" y="160" width="30" height="65" rx="3" fill="#6e8eff" opacity="0.6"/>
  <rect x="350" y="185" width="30" height="40" rx="3" fill="#6e8eff" opacity="0.4"/>
  <text x="275" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">A</text>
  <text x="320" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">B</text>
  <text x="365" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">C</text>
  <text x="320" y="258" text-anchor="middle" fill="#999" font-size="10" font-family="system-ui">soft 选择</text>
  <!-- T→∞ box -->
  <rect x="440" y="45" width="180" height="200" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="530" y="70" text-anchor="middle" fill="#34d399" font-size="13" font-family="system-ui" font-weight="bold">T → ∞（熔炉）</text>
  <!-- Bar chart for T→∞ -->
  <rect x="470" y="175" width="30" height="50" rx="3" fill="#34d399" opacity="0.6"/>
  <rect x="515" y="178" width="30" height="47" rx="3" fill="#34d399" opacity="0.6"/>
  <rect x="560" y="180" width="30" height="45" rx="3" fill="#34d399" opacity="0.6"/>
  <text x="485" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">A</text>
  <text x="530" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">B</text>
  <text x="575" y="240" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">C</text>
  <text x="530" y="258" text-anchor="middle" fill="#999" font-size="10" font-family="system-ui">≈ 均匀分布</text>
</svg>

### 温度的三大应用场景

**1. 文本生成的创意控制**

这是你最熟悉的场景。LLM 生成下一个 token 时，用温度来控制随机性：
- $T = 0.2$：几乎总选概率最高的词，输出确定、重复
- $T = 0.7$：适度随机，兼顾质量和多样性（大多数 chatbot 的默认值）
- $T = 1.5$：高度随机，可能生成惊喜内容，也可能胡说八道

**2. 知识蒸馏的"暗知识"**

2015 年 Hinton 提出的知识蒸馏有一个关键洞察：大模型输出的"次优选项"包含宝贵信息。比如手写数字识别中，模型判断一张图是"7"（概率 0.95），但它也给了"1"一个 0.03 的概率——这说明模型"知道"7 和 1 长得有点像。

如果直接用硬标签（one-hot），这些信息全丢了。解决方案：用高温度（$T=5$ 或 $T=20$）软化 teacher 模型的输出，让次优选项的概率被放大，然后让 student 模型学这个"软目标"。温度越高，分布越平滑，**暗知识**暴露得越多。

**3. Transformer Attention 中的 $\sqrt{d_k}$**

等等——Transformer 里 attention score 除以 $\sqrt{d_k}$ 不就是温度吗？没错！ 

在原始论文 *"Attention Is All You Need"* 中，Vaswani 等人写道：当 $d_k$（key 维度）很大时，点积的方差也很大（大约等于 $d_k$），这会把 softmax"推入梯度极小的区域"。

除以 $\sqrt{d_k}$ 就是在设置温度 $T = \sqrt{d_k}$，把点积的方差拉回 1 附近，让 softmax 工作在"敏感区"。这是我们下一节要深入讨论的"饱和"问题。

## 饱和：当 softmax 变成"死区"

### 问题：为什么大的输入会杀死梯度？

想象 softmax 的输入是 `[10, 0, 0]`。输出约为 `[0.9999, 0.00005, 0.00005]`。现在如果输入变成 `[100, 0, 0]`，输出变成 `[1.0, ~0, ~0]`——已经完全饱和了。

问题在于：**一旦输出接近 one-hot，无论你怎么调整输入，输出几乎不变**。反映到数学上，就是梯度趋近于零。模型学不动了。

### 核心直觉：softmax 的"甜蜜区"很窄

我们来看 softmax 的梯度（Jacobian 矩阵）。对于输出 $p_i = \text{softmax}(z_i)$：

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1-p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

翻译成人话：
- 对角线元素是 $p_i(1-p_i)$——这是一个倒 U 形的函数，在 $p_i = 0.5$ 时最大（等于 0.25），在 $p_i \to 0$ 或 $p_i \to 1$ 时趋向零
- 非对角线是 $-p_i p_j$——当任何一个概率趋向零时，这个梯度也消失了

所以 softmax 梯度最大的时候，是**输出分布比较均匀**的时候。一旦某个概率接近 1（其余接近 0），所有梯度都近乎为零。这就是"饱和"。

### 这在 Transformer 中意味着什么？

在 Transformer 的 self-attention 中，softmax 的输入是 $QK^T / \sqrt{d_k}$。如果两个 token 的 query 和 key 非常相似，点积会很大，softmax 后这个位置几乎得到全部注意力——其他位置的梯度全部消失。

**注意力坍缩（Attention Collapse/Entropy Collapse）** 正是这个问题：某些注意力头把几乎所有权重集中在一个 token 上（通常是第一个 token 或某个特殊 token），失去了"看全局"的能力。

2026 年 Varre 等人在 *"Gradient Flow Polarizes Softmax Outputs towards Low-Entropy Solutions"*（arxiv 2603.06248）中从理论上证明了：**梯度流天然将 softmax 输出推向低熵（稀疏）解**。这是 softmax 内在的归纳偏置——不管损失函数是什么，训练都倾向于让注意力变得越来越集中。

这解释了为什么我们会观察到"注意力沉降"（attention sinks）现象——某些 token 莫名其妙地吸引了大量注意力，以及为什么有些层会出现"巨大激活值"（massive activations）。

<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="300" y="22" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Softmax 梯度 p(1−p) 与饱和区</text>
  <!-- Axes -->
  <line x1="60" y1="200" x2="550" y2="200" stroke="#3a3a4a" stroke-width="1.5"/>
  <line x1="60" y1="200" x2="60" y2="40" stroke="#3a3a4a" stroke-width="1.5"/>
  <!-- X axis labels -->
  <text x="60" y="220" text-anchor="middle" fill="#999" font-size="11" font-family="system-ui">0</text>
  <text x="305" y="220" text-anchor="middle" fill="#999" font-size="11" font-family="system-ui">0.5</text>
  <text x="550" y="220" text-anchor="middle" fill="#999" font-size="11" font-family="system-ui">1</text>
  <text x="305" y="242" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">p_i（softmax 输出概率）</text>
  <!-- Y axis label -->
  <text x="25" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui" transform="rotate(-90, 25, 120)">梯度大小</text>
  <!-- Curve p(1-p) -->
  <path d="M 60,200 Q 100,195 140,175 Q 180,145 220,110 Q 260,75 305,60 Q 350,75 390,110 Q 430,145 470,175 Q 510,195 550,200" fill="none" stroke="#6e8eff" stroke-width="2.5"/>
  <!-- Peak marker -->
  <circle cx="305" cy="60" r="4" fill="#6e8eff"/>
  <text x="305" y="48" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">最大值 0.25</text>
  <!-- Dead zones -->
  <rect x="60" y="35" width="80" height="175" rx="4" fill="#ff6b6b" opacity="0.08"/>
  <text x="100" y="55" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">死区</text>
  <rect x="470" y="35" width="80" height="175" rx="4" fill="#ff6b6b" opacity="0.08"/>
  <text x="510" y="55" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">死区</text>
  <!-- Sweet spot -->
  <rect x="200" y="35" width="210" height="175" rx="4" fill="#34d399" opacity="0.06"/>
  <text x="305" y="235" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">← 甜蜜区：梯度有效流动 →</text>
</svg>

### 应对饱和的工程手段

1. **缩放点积**：Transformer 除以 $\sqrt{d_k}$，让 softmax 输入方差 ≈ 1
2. **熵正则化**：在损失函数中加一项，惩罚过低的注意力熵，防止坍缩
3. **Sigmoid 替代 Softmax**：Apple 在 2025 ICLR 发表的工作证明，用 sigmoid 逐元素替代 softmax 可以避免"赢者通吃"，同时匹配 softmax 的性能
4. **QK-Norm**：对 Q 和 K 做 L2 归一化，限制点积的范围在 $[-1, 1]$ 之间

## 数值稳定性：一行代码的生死之差

### 问题：exp 太容易爆炸或消失了

让我们回到开头的 NaN bug。如果 softmax 输入是 `[1000, 1000, 1000]`：
- `exp(1000)` ≈ $10^{434}$，远超 float32 的最大值 $3.4 \times 10^{38}$ → **上溢为 inf**
- `inf / inf` = NaN → 程序崩溃

如果输入是 `[-1000, -1000, -1000]`：
- `exp(-1000)` ≈ $10^{-434}$，下溢为 0
- `0 / 0` = NaN → 同样崩溃

### 核心直觉：减去最大值不改变结果

Softmax 有一个救命性质：**平移不变性**。对所有输入加或减同一个常数，输出完全不变：

$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_j e^{z_j} \cdot e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

所以实践中的做法是：**先减去输入的最大值**，再算 exp：

```python
def safe_softmax(z):
    z_max = z.max()           # 找最大值
    exp_z = exp(z - z_max)    # 减去最大值后取 exp（最大的 exp 值为 1，不会溢出）
    return exp_z / exp_z.sum()
```

减去 $z_{\max}$ 后，最大的输入变成 0（$e^0 = 1$），其余都是负数（$e^{负数} < 1$）。没有任何数会溢出。这就是 **safe softmax** 或 **stable softmax**。

### Log-Sum-Exp（LSE）技巧

在深度学习中，我们经常需要的其实不是 softmax 本身，而是 **log-softmax**（因为交叉熵损失 = 负的 log 概率）：

$$\log \text{softmax}(z_i) = z_i - \log\left(\sum_j e^{z_j}\right)$$

直接算 $\sum_j e^{z_j}$ 还是会溢出。解决方案是 **Log-Sum-Exp 技巧**：

$$\log\sum_j e^{z_j} = z_{\max} + \log\sum_j e^{z_j - z_{\max}}$$

这就是为什么 PyTorch 的 `nn.CrossEntropyLoss` 要求你传入 **原始 logits** 而不是 softmax 后的概率——它内部会用 LSE 技巧做数值稳定的 log-softmax 计算。如果你先自己算 softmax，再传给 NLLLoss，不仅多算了一步，还失去了数值稳定性保证。

### Safe Softmax 的代价：三次遍历

标准的 safe softmax 需要三次遍历数据：
1. **第一遍**：找最大值 $z_{\max}$
2. **第二遍**：计算 $\sum_j e^{z_j - z_{\max}}$
3. **第三遍**：每个元素除以总和

对于长序列（比如 128K tokens 的注意力矩阵），每多一次遍历就多一次内存读取。2018 年 Milakov 和 Gimelshein 提出了 **Online Softmax**（arxiv 1805.02867），把前两遍合并成一遍——边遍历边更新最大值和累积和。后来 FlashAttention 正是基于这个思想，把 softmax 的分块计算和注意力矩阵的分块乘法融合在一起，实现了"一边算 softmax 一边算加权求和"的魔法。

<svg viewBox="0 0 650 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="325" y="22" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Softmax 实现演进：从朴素到在线</text>
  <!-- Naive -->
  <rect x="20" y="40" width="150" height="140" rx="8" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="95" y="62" text-anchor="middle" fill="#ff6b6b" font-size="12" font-family="system-ui" font-weight="bold">朴素 Softmax</text>
  <text x="95" y="85" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">exp(z_i)</text>
  <text x="95" y="105" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">sum(exp)</text>
  <text x="95" y="125" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">exp / sum</text>
  <text x="95" y="155" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">❌ 溢出风险</text>
  <text x="95" y="170" text-anchor="middle" fill="#999" font-size="10" font-family="system-ui">2 passes</text>
  <!-- Arrow -->
  <line x1="175" y1="110" x2="210" y2="110" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr3)"/>
  <!-- Safe -->
  <rect x="215" y="40" width="170" height="140" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="300" y="62" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui" font-weight="bold">Safe Softmax</text>
  <text x="300" y="85" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">① max(z)</text>
  <text x="300" y="105" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">② sum(exp(z−max))</text>
  <text x="300" y="125" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">③ 归一化</text>
  <text x="300" y="155" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">✓ 数值稳定</text>
  <text x="300" y="170" text-anchor="middle" fill="#999" font-size="10" font-family="system-ui">3 passes</text>
  <!-- Arrow -->
  <line x1="390" y1="110" x2="425" y2="110" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr3)"/>
  <!-- Online -->
  <rect x="430" y="40" width="190" height="140" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="525" y="62" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui" font-weight="bold">Online Softmax</text>
  <text x="525" y="85" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">①② 边扫描边更新</text>
  <text x="525" y="105" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">   max + sum 合并</text>
  <text x="525" y="125" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">③ 归一化</text>
  <text x="525" y="155" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">✓ FlashAttention 基石</text>
  <text x="525" y="170" text-anchor="middle" fill="#999" font-size="10" font-family="system-ui">2 passes（少 1 次内存读）</text>
</svg>

## Softmax 瓶颈：表达力的天花板

### 问题：softmax 输出的概率分布够丰富吗？

2017 年 Yang 等人发现了一个理论问题（arxiv 1711.03953 *"Breaking the Softmax Bottleneck"*）：语言的概率分布是极其复杂的，但 softmax 层能表示的分布受限于一个矩阵的**秩**。

具体来说，如果隐层维度是 $d$，词表大小是 $V$，那么 softmax 输出的 log 概率矩阵最多是 rank-$d$ 的。而自然语言的真实分布可能需要更高的秩才能精确表示。当 $d < V$（几乎总是如此——比如 $d=4096$ 而 $V=128000$），模型在数学上就**不可能**完美拟合真实分布。

这就是 **softmax 瓶颈**。解决方案包括：Mixture of Softmaxes（MoS）、自适应 softmax、以及更现代的方法如用更大的输出嵌入。

## Softmax 的替代者们

既然 softmax 有这么多问题（饱和、数值不稳定、赢者通吃、表达力瓶颈），有没有替代方案？

### Sparsemax（2016）

直接输出稀疏概率分布——大部分概率为**精确的零**，而非 softmax 那种"接近零但不是零"。好处是可解释性更强（模型明确告诉你哪些位置完全不重要），但梯度在零处不可微，需要用次梯度方法。

### Sigmoid Attention（2024-2025）

Apple 的研究表明，用逐元素 sigmoid 替代行级 softmax：

$$\text{attn}(Q, K, V) = \sigma(QK^T / \sqrt{d_k}) \cdot V$$

不需要归一化（注意力权重不必加到 1），避免了"赢者通吃"效应。在语言和视觉任务上都能匹配 softmax 的性能，而且更适合并行化（sigmoid 是逐元素的，不需要跨序列的 reduce 操作）。

### 线性 Attention（2020-）

彻底去掉 softmax，用核函数近似。$\text{attn} = \phi(Q) \cdot \phi(K)^T \cdot V$，可以改变计算顺序避免 $O(n^2)$。代价是失去了 softmax 的稀疏化能力，实际效果通常略逊于标准 attention。

## 这一切意味着什么

Softmax 是深度学习中最精巧的设计之一，但也是陷阱最多的地方之一：

1. **它不只是"归一化到概率"**——exp 的使用决定了它天然放大差异、倾向于稀疏解
2. **温度参数让它变成一个可调的"确定性旋钮"**——从 argmax 到均匀分布之间的连续谱
3. **它的梯度只在"均匀区"有效流动**——一旦饱和（输出接近 one-hot），训练就停滞。这是 Transformer 中 $\sqrt{d_k}$ 缩放存在的根本原因
4. **朴素实现必崩**——必须用 max-shift + LSE 技巧保证数值安全，online softmax 进一步优化了性能
5. **它有表达力天花板**——softmax 瓶颈限制了模型能拟合的分布复杂度
6. **替代方案正在涌现**——sigmoid attention 和稀疏 attention 在挑战 softmax 近 10 年的统治地位

下次你看到 loss 突然变 NaN、注意力图全黑、或者模型输出总是重复同一个词的时候，记得——很可能是 softmax 在作怪。

## 下一篇预告

我们讲了 softmax 怎么把 logits 变成概率、温度怎么控制确定性、以及数值上的坑。但还有一个更深的问题：**当模型做最终预测时，它为什么要用交叉熵损失？这个损失和信息论有什么关系？Perplexity 又是怎么从这里来的？** 下一篇我们深入信息论视角。
