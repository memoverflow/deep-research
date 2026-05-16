---
title: "为什么 Adam 是深度学习的默认优化器：从直觉到数学"
date: 2025-05-15
level: 3
series: "LLM 原理深度解析"
series_order: 1
series_total: 2
tags: [optimizer, Adam, AdamW, training, LLM]
summary: "从 SGD 的困境出发，一步步推导出 Adam 优化器为什么这样设计，以及 AdamW 解决了什么被忽视了十年的 bug。"
---

# 为什么 Adam 是深度学习的默认优化器：从直觉到数学

> 如果你训练过任何神经网络，你几乎一定用过 Adam。但你有没有想过——为什么偏偏是它？一个 2014 年发表的算法，凭什么统治了整整十年的深度学习？

## 一场关于"下山"的困境

想象你蒙着眼站在一座复杂的山地上，目标是走到最低点。你唯一能感知的信息是脚下地面的倾斜方向——这就是梯度。

最朴素的策略是：哪边倾斜就往哪边走一步。这就是**梯度下降（SGD）**。听起来合理，但实际操作中你会碰到三个让人崩溃的问题：

**问题 1：步子多大？** 如果你每步固定走 1 米，在陡峭的山坡上你可能直接冲过谷底、飞到对面山坡上去（震荡）。但如果你小心翼翼每步走 1 厘米，在平坦的高原上你要走到天荒地老才能走出去。

**问题 2：方向靠不靠谱？** 在深度学习里，你不是用全部数据算梯度（太贵了），而是随机抽一小批数据（mini-batch）估算梯度。这个估算噪声很大——就好像你脚下的地面在不停颤抖，每次感受到的倾斜方向都略有不同。

**问题 3：每个维度的地形不一样。** 有些参数方向上地形很陡（梯度大），有些方向上几乎是平原（梯度小）。用同一个步长走所有方向，注定顾此失彼。

Adam 的故事，就是人们用了十年时间、一步步解决这三个问题的历程。

## 第一块拼图：动量——给球一点惯性

### 问题是什么

纯 SGD 的噪声问题有多严重？想象你在一个狭长的峡谷里（一个方向很陡，另一个方向很平缓）。每次 mini-batch 给你一个略偏的方向，你就会在峡谷两壁之间来回弹跳，而沿着峡谷底部前进的速度慢得可怜。

### 核心直觉：滚动的重球

解决方案极其直觉：给梯度一点"记忆"。

不要只看当前这一步的梯度，把过去几步的方向也考虑进来。如果连续好几步都往同一个方向指，那就加速；如果方向来回摇摆，那就互相抵消、减速。

这就像从一个无质量的点变成了一个有重量的球——球在滚动时有惯性，不会被地面的每一个小坑所左右，而是沿着整体趋势滚动。

### 技术实现

数学上，我们维护一个"动量向量" $m$，它是梯度的**指数移动平均（EMA）**：

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

翻译成人话：新的动量 = 90% 的旧动量 + 10% 的当前梯度（$\beta_1 = 0.9$ 时）。

这意味着：
- 如果最近 10 步梯度都指向右边，$m$ 就会积累成一个很大的"向右"的速度
- 如果梯度忽左忽右，正负抵消，$m$ 会很小——自动减速

这就是 SGD with Momentum。它解决了问题 2（噪声平滑），也部分缓解了问题 1（在一致方向上自动加速）。但问题 3（不同参数需要不同步长）还没解决。

<svg viewBox="0 0 650 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- SGD path (zigzag) -->
  <text x="80" y="20" text-anchor="middle" fill="#ef4444" font-size="12" font-family="system-ui">纯 SGD：来回震荡</text>
  <polyline points="30,180 60,50 90,170 120,60 150,160 180,70 210,150" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="210" cy="150" r="4" fill="#ef4444"/>
  <!-- Momentum path (smooth) -->
  <text x="430" y="20" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">带动量：平滑前进</text>
  <path d="M 300,180 C 330,140 360,100 390,90 C 420,80 450,75 500,70" fill="none" stroke="#34d399" stroke-width="2"/>
  <circle cx="500" cy="70" r="4" fill="#34d399"/>
  <!-- Valley walls -->
  <rect x="20" y="35" width="200" height="160" rx="8" fill="none" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="2,2"/>
  <rect x="290" y="35" width="220" height="160" rx="8" fill="none" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="2,2"/>
  <!-- Labels -->
  <text x="120" y="210" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">峡谷（高曲率×低曲率）</text>
  <text x="400" y="210" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">惯性帮助穿越平坦方向</text>
  <!-- Star at minimum -->
  <text x="190" y="125" fill="#fbbf24" font-size="16">★</text>
  <text x="490" y="80" fill="#fbbf24" font-size="16">★</text>
  <text x="560" y="78" fill="#94a3b8" font-size="10" font-family="system-ui">最低点</text>
</svg>

## 第二块拼图：自适应学习率——每个参数有自己的步长

### 问题是什么

动量解决了方向的问题，但步长呢？考虑一个自然语言模型：词"the"每个 batch 都出现（梯度一直在更新），而词"serendipity"可能训练了很久才遇到一次。

用同一个学习率对待这两个参数不合理——频繁更新的参数应该用小步长（已经学得差不多了），偶尔更新的参数应该用大步长（好不容易来了一次，多学点）。

### AdaGrad 的想法：累计梯度的历史大小

**AdaGrad（2011）**的解决方案很直接：对每个参数，记录它历史上所有梯度的平方和 $v_t = \sum_{i=1}^t g_i^2$，然后用 $\frac{\eta}{\sqrt{v_t}}$ 作为学习率。

直觉很妙：梯度大的参数（频繁/强烈更新的参数），$v_t$ 积累得快，学习率就自动缩小；梯度小的参数（稀疏/微弱更新的参数），$v_t$ 积累得慢，学习率保持较大。

但 AdaGrad 有一个致命缺陷：$v_t$ 只增不减。训练到后期，所有参数的学习率都会衰减到接近零——模型还没收敛就"冻住"了。

### RMSProp 的修复：别记住所有历史，只看最近的

**RMSProp（2012，Hinton 在课堂上提出）**的修复极其简洁：把"累加所有历史"换成"指数移动平均"：

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

这样 $v_t$ 反映的是**最近**梯度大小的趋势，不会无限增长。$\beta_2 = 0.999$ 意味着大约记住最近 1000 步的信息。

学习率变成了 $\frac{\eta}{\sqrt{v_t} + \epsilon}$（$\epsilon$ 防止除零）。

现在，我们有两块拼图了：
- **动量（Momentum）**→ 处理梯度**方向**的噪声
- **RMSProp** → 自适应调整每个参数的**步长**

能不能把它们合在一起？

## Adam：两全其美

### 核心想法：一句话版本

**Adam = Momentum + RMSProp + 偏差修正。** 就是这么简单。它同时维护两个统计量：
- **一阶矩 $m_t$**（梯度的移动平均）→ 告诉你"最近这些步整体在往哪走"
- **二阶矩 $v_t$**（梯度平方的移动平均）→ 告诉你"最近这些步梯度有多大"

然后用 $m_t$ 的方向、$v_t$ 的大小来决定每一步怎么走：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

翻译成人话：**沿着平滑后的梯度方向走，步长根据这个参数最近梯度的大小自动调整。** 梯度一直很大的参数走小步（不要冲过头），梯度一直很小的参数走大步（别在平原上爬太慢）。

### 完整算法（一步步拆解）

让我把 Adam 的每一步都翻译成人话：

**第 1 步：计算当前梯度** $g_t = \nabla L(\theta_t)$
> "脚下的地面往哪边斜？"

**第 2 步：更新一阶矩（动量）** $m_t = 0.9 \cdot m_{t-1} + 0.1 \cdot g_t$
> "结合过去的经验，总体来说应该往哪走？"

**第 3 步：更新二阶矩（梯度大小的记忆）** $v_t = 0.999 \cdot v_{t-1} + 0.001 \cdot g_t^2$
> "这个参数最近的梯度波动有多大？"

**第 4 步：偏差修正**（后面详细解释为什么需要）
$$\hat{m}_t = \frac{m_t}{1 - 0.9^t}, \quad \hat{v}_t = \frac{v_t}{1 - 0.999^t}$$

**第 5 步：更新参数** $\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
> "沿平滑方向走，步长大小由该参数的历史波动决定"

### 偏差修正：一个容易被忽视的精巧设计

这里有个微妙的问题。$m$ 和 $v$ 都初始化为 0。在训练最初几步：

- $m_1 = 0.9 \times 0 + 0.1 \times g_1 = 0.1 g_1$

但真实的梯度均值应该接近 $g_1$，而不是 $0.1 g_1$！因为 $m$ 从零开始，前几步的估计被严重低估（**偏向零**）。

修正方法是除以 $(1 - \beta^t)$：
- 第 1 步：$\hat{m}_1 = \frac{0.1 g_1}{1 - 0.9^1} = \frac{0.1 g_1}{0.1} = g_1$ ✓
- 第 10 步：$1 - 0.9^{10} = 0.65$，修正因子约 1.5
- 第 100 步：$1 - 0.9^{100} \approx 1$，几乎不修正了

这就是为什么说偏差修正是"启动时的助推器"——只在前几十步有明显影响，之后自动消失。2025 年的研究（"Simplifying Adam: Bias Correction Debunked"）甚至表明，对于足够长的训练，去掉偏差修正影响微乎其微。但对于少量步数的微调（fine-tuning），它仍然重要。

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Timeline arrow -->
  <line x1="50" y1="200" x2="650" y2="200" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <text x="660" y="205" fill="#94a3b8" font-size="10" font-family="system-ui">时间</text>
  <!-- SGD box -->
  <rect x="60" y="40" width="110" height="70" rx="8" fill="#1e1e2a" stroke="#ef4444" stroke-width="1.5"/>
  <text x="115" y="65" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">SGD</text>
  <text x="115" y="85" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">固定学习率</text>
  <text x="115" y="100" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">噪声大</text>
  <line x1="115" y1="110" x2="115" y2="190" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="115" y="220" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">~1950s</text>
  <!-- Momentum box -->
  <rect x="200" y="40" width="110" height="70" rx="8" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="255" y="65" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">+ Momentum</text>
  <text x="255" y="85" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">方向平滑</text>
  <text x="255" y="100" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">加速收敛</text>
  <line x1="255" y1="110" x2="255" y2="190" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="255" y="220" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">1964</text>
  <!-- AdaGrad box -->
  <rect x="340" y="40" width="110" height="70" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="395" y="65" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">AdaGrad</text>
  <text x="395" y="85" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">自适应步长</text>
  <text x="395" y="100" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">会衰减到 0</text>
  <line x1="395" y1="110" x2="395" y2="190" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="395" y="220" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">2011</text>
  <!-- RMSProp box -->
  <rect x="470" y="40" width="110" height="70" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="525" y="65" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">RMSProp</text>
  <text x="525" y="85" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">EMA 修复衰减</text>
  <text x="525" y="100" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">但无动量</text>
  <line x1="525" y1="110" x2="525" y2="190" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="525" y="220" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">2012</text>
  <!-- Adam box (highlighted) -->
  <rect x="590" y="30" width="90" height="80" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="2.5"/>
  <text x="635" y="55" text-anchor="middle" fill="#6e8eff" font-size="13" font-family="system-ui" font-weight="bold">Adam</text>
  <text x="635" y="75" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">动量+自适应</text>
  <text x="635" y="92" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">+偏差修正</text>
  <line x1="635" y1="110" x2="635" y2="190" stroke="#6e8eff" stroke-width="1.5" stroke-dasharray="3,3"/>
  <text x="635" y="220" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui" font-weight="bold">2014</text>
  <!-- Arrows between boxes -->
  <line x1="170" y1="75" x2="198" y2="75" stroke="#3a3a4a" stroke-width="1" marker-end="url(#arrow2)"/>
  <line x1="310" y1="75" x2="338" y2="75" stroke="#3a3a4a" stroke-width="1" marker-end="url(#arrow2)"/>
  <line x1="450" y1="75" x2="478" y2="75" stroke="#3a3a4a" stroke-width="1" marker-end="url(#arrow2)"/>
  <line x1="580" y1="75" x2="588" y2="75" stroke="#3a3a4a" stroke-width="1" marker-end="url(#arrow2)"/>
</svg>

## $\epsilon$ 的角色：比你想的重要得多

Adam 公式里那个不起眼的 $\epsilon$（默认 $10^{-8}$）看似只是"防止除零"的技术细节，但最近的研究揭示了更深的故事。

当 $\sqrt{\hat{v}_t}$ 很小（梯度接近零）而 $\epsilon$ 也很小时，更新步长 $\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ 会变得异常大——模型在梯度几乎为零的"平原"上突然迈出巨大一步。

2026 年 Sifal Klioui 的分析（"The Epsilon Trap: When Adam Stops Being Adam"）指出：当 $\epsilon$ 设得太大（比如 $10^{-4}$ 或 $1$），Adam 的自适应性质会消失，退化为带动量的 SGD。设得太小则可能在平坦区域产生不稳定的大跳跃。

**实践建议：** 对于 LLM 训练，$\epsilon = 10^{-8}$ 的默认值通常 work well。但如果你用了 FP16 混合精度训练，可能需要调大到 $10^{-5}$（因为 FP16 的最小正数约为 $6 \times 10^{-8}$，太小的 $\epsilon$ 在半精度下会被舍入为零）。

## AdamW：一个被忽视了三年的 Bug

### Weight Decay 的本意

正则化是防止模型过拟合的标准手段。最经典的方式是 **L2 正则化**：在损失函数后面加一个惩罚项 $\frac{\lambda}{2}|\theta|^2$，让权重不要长得太大。

在纯 SGD 中，L2 正则化等价于在每步更新时让权重"衰减"一点点：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t - \eta \lambda \theta_t = (1 - \eta\lambda)\theta_t - \eta \cdot g_t$$

这个 $(1 - \eta\lambda)$ 的乘法效果，就叫 **weight decay（权重衰减）**。

在 SGD 里，"L2 正则化"和"weight decay"是一回事——数学上完全等价。

### 问题出在哪

但当你把 L2 正则化放进 Adam 时，等价性**被破坏了**。

为什么？因为 Adam 会对梯度进行缩放（除以 $\sqrt{v_t}$）。L2 惩罚产生的梯度 $\lambda\theta$ 也会被这个自适应缩放处理——这意味着：

- 对于梯度大的参数，正则化效果**被缩小**了（因为 $v_t$ 大，分母大）
- 对于梯度小的参数，正则化效果**被放大**了

这完全不是 weight decay 想要的效果！你本来想均匀地让所有权重缩小，结果变成了"选择性衰减"——偏偏大梯度的参数（可能最需要正则化的）反而衰减最少。

### AdamW 的修复：一行代码的改变

Loshchilov & Hutter 在 2017 年指出这个问题，解决方案简洁到令人叹息——把 weight decay **从梯度计算中解耦出来**，直接加在参数更新步骤中：

```
# Adam + L2（有问题的方式）
g_t = gradient + λ * θ    ← 正则化梯度会被 v_t 缩放
m_t = β1 * m + (1-β1) * g_t
v_t = β2 * v + (1-β2) * g_t²
θ = θ - lr * m̂_t / (√v̂_t + ε)

# AdamW（正确的方式）
g_t = gradient              ← 纯梯度，不含正则项
m_t = β1 * m + (1-β1) * g_t
v_t = β2 * v + (1-β2) * g_t²
θ = θ - lr * m̂_t / (√v̂_t + ε) - lr * λ * θ  ← 独立的衰减
```

**一句话总结：** AdamW 让自适应学习率只管优化方向和步长，weight decay 独立执行自己的"让权重缩小"的任务，互不干扰。

这个看似微小的改动对 LLM 训练影响巨大。2019 年论文正式发表后，AdamW 迅速成为所有大模型训练的标准配置——GPT、LLaMA、Gemini 无一例外。

## Adam 的代价：训练一个 LLM 到底需要多少内存？

Adam 的便利不是免费的。对于每一个模型参数，Adam 需要额外存储**两个状态**：
- $m$（一阶矩）：和参数一样大
- $v$（二阶矩）：和参数一样大

如果模型本身用 FP16（2 字节/参数），但优化器状态通常保持 FP32（4 字节）以保证数值精度。算一笔账：

| 组成部分 | 每参数字节数 | 70B 模型总量 |
|---------|------------|------------|
| 模型参数 (FP16) | 2 | 140 GB |
| 梯度 (FP16) | 2 | 140 GB |
| Adam $m$ (FP32) | 4 | 280 GB |
| Adam $v$ (FP32) | 4 | 280 GB |
| **合计** | **12** | **840 GB** |

一个 70B 参数的模型，光是 Adam 的两个状态就要 560 GB——比模型本身还大 4 倍！这就是为什么 DeepSpeed ZeRO、FSDP 等分布式训练框架的第一级优化就是"切分优化器状态"。

也是为什么 2024-2025 年出现了大量"省内存优化器"的研究（Adam-mini、APOLLO、Swan 等），它们的核心思路都是：能不能不给每个参数都存一份完整的 $v$？

## 为什么 Adam 统治了 LLM 训练

说了这么多，回到核心问题：为什么是 Adam？

**1. 对超参数不敏感。** Adam 的默认设置（$\eta=0.001, \beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$）在绝大多数场景下直接能用。SGD 的学习率如果设错一个数量级，训练直接崩溃；Adam 则因为自适应特性，容错范围大得多。

**2. 处理稀疏梯度。** Transformer 的 Embedding 层有几万甚至几十万个词的向量，每个 batch 只有很少的词被激活。Adam 的逐参数自适应学习率天然适合这种稀疏更新模式。

**3. 在非平稳目标上表现好。** LLM 训练的 loss landscape 极其复杂，不同训练阶段的梯度统计特性差异很大。Adam 的指数移动平均能自动适应这种变化。

**4. 和学习率调度配合好。** 现代 LLM 训练通常用 warmup + cosine decay 的学习率策略。Adam 的自适应性和外部学习率调度是正交的——前者调整参数间的相对步长，后者调整全局步长大小。

### 但 Adam 不是万能的

公平地说，Adam 也有已知的问题：

- **泛化差距：** 在某些视觉任务上，SGD with momentum 找到的解比 Adam 泛化更好。这可能因为 Adam 倾向于收敛到 sharp minima（尖锐的最小值），而 SGD 的噪声帮助它找到 flat minima（平坦的最小值）
- **内存开销：** 如前所述，状态量是参数量的 2 倍
- **可能不收敛到临界点：** 2024 年的理论分析指出，Adam 在某些设置下可能收敛到非临界点。不过在实践中这很少成为问题

## Adam 之后：优化器还在进化

Adam 统治了十年，但研究从未停止：

**Adam-mini (2024)**：观察到 Transformer 不同层的 Hessian 结构（二阶信息）有显著模式——同一层内的参数共享相似的曲率。因此可以用一个 block 级别的平均 $v$ 代替逐参数的 $v$，内存减半而性能不降。

**SOAP (2024)**：对 Adam 的二阶矩做 Shampoo 式的结构化处理，在 nanogpt 等 benchmark 上比 AdamW 收敛更快。

**APOLLO (2025)**：用随机投影把 $v$ 压缩到极低维度，实现 SGD 级别的内存开销 + Adam 级别的性能。

这些后续工作的共同特点是：它们都不是"推翻 Adam"，而是"在 Adam 的框架内做减法"——用结构化的假设来减少冗余存储，核心的"一阶矩 + 二阶矩自适应"思想不变。

## 回顾：从 SGD 到 Adam 的逻辑线

让我把整个故事串起来：

1. **SGD**：最朴素，但固定学习率+噪声+各向同性三个问题
2. **Momentum**：加入方向记忆（一阶矩），解决噪声和震荡
3. **AdaGrad**：加入梯度大小记忆（二阶矩），解决各向同性，但会衰减到死
4. **RMSProp**：用 EMA 替代累加，修复 AdaGrad 的衰减问题
5. **Adam**：Momentum + RMSProp + 偏差修正 = 当前的默认选择
6. **AdamW**：修复 Adam 中 weight decay 被自适应缩放污染的 bug

每一步都是对前一步的自然改进，没有一步是凭空冒出来的。这就是为什么理解 Adam 需要理解整条进化链——它不是一个天才的发明，而是十年集体探索的结晶。

---

*这是"LLM 原理深度解析"系列的第一篇。下一篇我们将探讨学习率调度策略——为什么训练开始时要 warmup？为什么 cosine 比固定学习率好？*
