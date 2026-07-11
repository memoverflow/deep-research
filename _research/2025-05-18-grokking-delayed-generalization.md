---
title: "Grokking：过拟合之后的顿悟时刻"
date: 2025-05-18
level: 3
series: "LLM 原理深度解析"
series_order: 7
series_total: 37
tags: [grokking, generalization, overfitting, mechanistic-interpretability, weight-decay]
summary: "神经网络可以在完全过拟合之后，经过漫长的沉默期突然学会泛化——这个叫 Grokking 的现象颠覆了我们对训练的认知"
---

# Grokking：过拟合之后的顿悟时刻

> 如果一个学生考试作弊拿了满分，你会觉得他已经没救了。但如果告诉你，让他继续坐在考场里一百倍的时间，他会突然真正学会所有知识——你信吗？

## 一个违反直觉的发现

2022 年初，OpenAI 的 Alethea Power 和同事们发表了一篇不长但引发巨震的论文。他们用一个小型 Transformer 做一件看起来很简单的事：学习模运算（比如"23 ÷ 45 mod 97 等于几"）。

结果出现了一个所有人都没预料到的现象：

模型在不到 1000 步训练后就达到了 99.9% 的训练准确率——它完美地"记住"了所有答案。但测试准确率呢？和随机猜没有区别，大约 1%。

按照传统机器学习的认知，故事到这里就该结束了。模型过拟合了，继续训练只会让事情更糟。所有教科书都告诉我们：当验证损失开始上升时，该停了。

但研究者没有停。他们继续训练了十万步、一百万步……然后，在某个谁也说不清的时刻，测试准确率突然从 1% 跳到了接近 100%。

不是慢慢爬升，是"跳"。像一个人沉默了一百天后突然开口说出了完美的演讲。

这个现象被命名为 **Grokking**——来自科幻小说家海因莱因 1961 年的小说《异乡异客》，意思是"完全理解某事物到与之融为一体的程度"。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-grok" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="80" y1="260" x2="650" y2="260" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-grok)"/>
  <line x1="80" y1="260" x2="80" y2="30" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-grok)"/>
  <text x="360" y="295" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">训练步数</text>
  <text x="35" y="150" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" transform="rotate(-90,35,150)">准确率</text>
  <!-- Training accuracy (fast rise) -->
  <path d="M 100 250 Q 130 240 150 80 L 600 75" fill="none" stroke="#34d399" stroke-width="2.5" stroke-dasharray="none"/>
  <text x="610" y="70" fill="#34d399" font-size="12" font-family="system-ui">训练准确率</text>
  <!-- Test accuracy (flat then sudden jump) -->
  <path d="M 100 250 Q 130 248 150 245 L 420 242 Q 440 240 450 200 Q 460 120 470 85 L 600 80" fill="none" stroke="#f472b6" stroke-width="2.5"/>
  <text x="610" y="90" fill="#f472b6" font-size="12" font-family="system-ui">测试准确率</text>
  <!-- Phase labels -->
  <rect x="100" y="5" width="80" height="22" rx="4" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="140" y="20" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">记忆阶段</text>
  <rect x="220" y="5" width="180" height="22" rx="4" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="310" y="20" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">漫长的沉默期（过拟合）</text>
  <rect x="430" y="5" width="80" height="22" rx="4" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="470" y="20" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">顿悟！</text>
  <!-- Vertical dashed lines for phases -->
  <line x1="150" y1="30" x2="150" y2="255" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="430" y1="30" x2="430" y2="255" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
  <!-- Annotations -->
  <text x="115" y="280" fill="#94a3b8" font-size="10" font-family="system-ui">~1K步</text>
  <text x="400" y="280" fill="#94a3b8" font-size="10" font-family="system-ui">~100K步</text>
</svg>

## 为什么这件事如此重要

你可能会想：这不就是一个小实验的奇怪现象吗？跟实际的大模型有什么关系？

关系比你想象的大得多。

**第一**，它颠覆了"何时该停止训练"的基本准则。如果 Grokking 是普遍存在的，那我们可能因为太早停止训练而错过了模型真正"学会"的时刻。

**第二**，它揭示了记忆和理解之间的深层关系。模型不是在"记忆"和"理解"之间做选择——它会先记忆，然后在记忆的基础上慢慢发展出真正的理解。这跟人类学习其实很像：你先死记乘法表，用了足够长时间后才突然"理解"了乘法的本质。

**第三**，2024 年的研究把 Grokking 和大语言模型的"涌现能力"联系了起来。GPT-4 在某些任务上的突然能力飞跃，可能与 Grokking 共享相同的底层机制。

## 到底发生了什么：两套电路的竞争

现在来到核心问题：为什么模型会在过拟合之后突然泛化？

### 直觉：考试作弊者 vs 真正学习者

想象一间教室里有两个学生同时在学数学：

**学生 A（记忆者）**：他把每道练习题的答案都背了下来。"3+5=8, 7+9=16, 12+3=15……"他背得很快，几分钟就能把练习册的所有题背完。但如果你出一道没见过的题，他就傻了。更重要的是——他脑子里要记的东西越来越多。100 道题要记 100 个答案。

**学生 B（理解者）**：他在琢磨"加法到底是什么"。他试了很多想法，画了很多图，想了很久。进度很慢。但一旦他想明白了，他就能算任何两个数的和——不需要背任何东西。

现在关键来了：**教室里有一条规则——你的"笔记本"每天会自动缩小一点。**

对学生 A 来说，这是灾难。他背的东西越多，笔记本越装不下，维持记忆越来越吃力。对学生 B 来说，这根本无所谓——他只需要记住一条规则。

最终的结果就是：学生 A 先交卷（训练完美），但随着笔记本不断缩小，他的记忆系统崩塌；而学生 B 的方法因为足够简洁，反而活了下来并主导了最终的行为。

这就是 Grokking 的本质。

### 技术视角：三个阶段

Neel Nanda 在 2023 年的开创性工作中，完整地逆向工程了一个学习"加法 mod 113"的 Transformer，揭示了 Grokking 经历三个清晰的阶段：

**阶段 1：记忆（Memorization）**  
模型迅速学会了一种"查找表"策略。对于训练集中的每一个 (a, b) 对，模型直接记住了 (a+b) mod 113 的答案。这很快——就像背答案一样。但参数量跟训练样本数成正比。

**阶段 2：电路形成（Circuit Formation）**  
在表面之下，一种完全不同的计算方式正在悄悄生长。模型开始学习用**离散傅里叶变换**来做模运算！具体来说：
- 嵌入层把每个数字 $i$ 映射到圆上的点 $(\cos(2\pi ik/p), \sin(2\pi ik/p))$
- 注意力层利用三角恒等式 $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$ 把加法变成了旋转的复合
- 输出层从最终的旋转角度读出结果

这个"泛化电路"在整个阶段 2 中持续成长，但因为"记忆电路"还在主导输出，所以测试准确率看不到任何变化。

**阶段 3：清理（Cleanup）**  
权重衰减持续对所有参数施压，迫使它们变小。记忆电路因为占用大量参数而承受巨大压力；泛化电路因为只需少量参数就能编码所有情况，所以几乎不受影响。最终，记忆电路崩塌，泛化电路接管——测试准确率瞬间飙升。

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Phase boxes -->
  <rect x="30" y="40" width="180" height="160" rx="8" fill="#1e1e2a" stroke="#f472b6" stroke-width="1.5"/>
  <text x="120" y="30" text-anchor="middle" fill="#f472b6" font-size="14" font-family="system-ui" font-weight="bold">阶段 1：记忆</text>
  <text x="120" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">查找表策略</text>
  <text x="120" y="92" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">(a,b) → 直接记住答案</text>
  <text x="120" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">⚡ 学得快</text>
  <text x="120" y="142" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">📦 参数量 ∝ 样本数</text>
  <text x="120" y="170" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">❌ 不能泛化</text>
  
  <rect x="260" y="40" width="180" height="160" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="350" y="30" text-anchor="middle" fill="#a78bfa" font-size="14" font-family="system-ui" font-weight="bold">阶段 2：电路生长</text>
  <text x="350" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">傅里叶算法萌芽</text>
  <text x="350" y="92" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">数字 → 圆上旋转</text>
  <text x="350" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">🐢 学得慢</text>
  <text x="350" y="142" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">📦 参数量固定（很少）</text>
  <text x="350" y="170" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">✓ 但外部看不见进步</text>
  
  <rect x="490" y="40" width="180" height="160" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="580" y="30" text-anchor="middle" fill="#34d399" font-size="14" font-family="system-ui" font-weight="bold">阶段 3：清理</text>
  <text x="580" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">权重衰减裁剪记忆</text>
  <text x="580" y="92" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">记忆电路成本过高</text>
  <text x="580" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">🧹 记忆崩塌</text>
  <text x="580" y="142" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">🎯 泛化电路接管</text>
  <text x="580" y="170" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">✓ 测试准确率跳升！</text>
  
  <!-- Arrows between phases -->
  <line x1="215" y1="120" x2="255" y2="120" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <line x1="445" y1="120" x2="485" y2="120" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  
  <!-- Bottom annotation -->
  <text x="350" y="230" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">权重衰减 = 裁判，持续惩罚"占地面积大"的方案</text>
</svg>

## 权重衰减：Grokking 的隐形推手

到这里你可能注意到了一个关键角色——**权重衰减（Weight Decay）**。

没有权重衰减，就没有 Grokking。这不是修辞，是实验事实。

### 为什么权重衰减是必需的？

回到我们的教室类比。"笔记本自动缩小"就是权重衰减。它做的事情很简单：每一步训练，所有参数都乘以一个略小于 1 的数（比如 0.999）。

$$\theta_{t+1} = (1 - \lambda) \cdot \theta_t - \eta \cdot \nabla L$$

其中 $\lambda$ 就是权重衰减系数。这个公式翻译回人话就是："每一步，参数先自然缩小一点点，然后才根据梯度更新。"

这看似温和的操作，对两种电路的影响截然不同：

- **记忆电路**需要大量参数来存储每个样本的答案。权重衰减每步都在侵蚀它的存储空间。就像一个背了 1000 个答案的人，每天被强制忘记其中一部分——维持记忆越来越难。
- **泛化电路**只需要少量参数编码一个通用算法。权重衰减对它的影响微乎其微。就像一个只记住"加法规则"的人——你缩小他的笔记本，他照样没问题。

### 一个优雅的数学图景

2026 年初的一篇论文（发表于 MLLog.dev 的数学证明）揭示了一个令人满意的结果：即使在最简单的岭回归（Ridge Regression）中，Grokking 也会发生。这意味着 Grokking 不是深度网络的什么玄学魔法——它是梯度下降 + 正则化的基本动力学。

核心图景是这样的：

模型的参数空间可以分成两部分——"与训练数据相关的方向"和"与训练数据正交的方向"。记忆阶段只关心前者（把训练数据拟合好），但初始化时后者的随机值还在。这些"正交方向"的噪声不影响训练损失（因为训练数据看不到它们），但严重伤害泛化（因为测试数据能看到）。

权重衰减以 $\exp(-\lambda t)$ 的速度指数衰减所有参数。正交方向的噪声最终被压到足够小——这就是泛化突然出现的时刻。

**Grokking 的延迟时间**大约正比于 $1/\lambda$。权重衰减越小，等待越久。在原始论文中，权重衰减设为 1.0（相当大），Grokking 在约 10 万步出现。如果设为 0.01，可能要等一千万步。如果完全没有权重衰减？永远不会 Grok。

## 物理学家看 Grokking：一场相变

物理学为 Grokking 提供了另一个深刻的理解角度。

想象水变成冰的过程。在 0°C 以上，水分子乱跑（高熵态）。降到 0°C 时，并不是每个分子同时停下来——而是某处先形成一个微小的冰晶核，然后这个有序结构迅速扩展，整杯水在很短时间内冻结。这叫**一阶相变**——系统在两个状态之间不是渐变，而是跳跃。

Grokking 就是神经网络中的相变。

Hebrew 大学的研究者从统计物理的角度证明：记忆状态就像"液态"（高熵、无序、每个样本独立编码），泛化状态就像"固态"（低熵、有序、用统一规则覆盖所有情况）。训练过程就像慢慢降温——达到临界点时，系统突然"结晶"。

还有一种更精妙的解读：Grokking 是**计算玻璃态弛豫**。快速记忆把网络"淬火"到了一个非平衡的玻璃态，然后系统通过缓慢的熵驱动弛豫走向真正的平衡——泛化态。

两种解释都指向同一个核心洞察：**泛化解占据参数空间中更高熵的区域**。记忆需要精确的、特定的参数配置（像一块精心雕刻的雕塑），泛化可以由许多等价的配置实现（像一团气体可以有无数种分子排列）。热力学告诉我们，系统在足够长的时间尺度上自然漂移向高熵态——Grokking 是这一普遍原理的神经网络版本。

## 加速顿悟：不想等一百万步怎么办？

如果你是一个训练模型的工程师，你可能在想：这个现象很酷，但我不想等一百万步。有办法加速吗？

### Grokfast：放大慢梯度

2024 年首尔国立大学的一篇论文提出了一个惊人简单的方法——**Grokfast**——只用几行代码就把 Grokking 速度提高了 50 倍以上。

思路是这样的：在训练过程中，梯度信号可以分为两个成分：

- **快变分量**：每一步都在剧烈波动，对应记忆电路的梯度（因为每个样本给出不同的信号）
- **慢变分量**：跨越很多步持续指向同一个方向，对应泛化电路的梯度（因为底层规律是一致的）

Grokfast 用一个指数移动平均（EMA）把慢变分量提取出来，然后放大它：

```python
# Grokfast 核心代码（惊人地简单）
grads_ema = alpha * grads_ema + (1 - alpha) * grads  # EMA 提取慢分量
grads = grads + lambda_amp * grads_ema               # 放大慢分量
```

翻译回人话：如果一个梯度方向持续指向同一边（说明它代表的是真正的规律），就把它放大；如果一个梯度方向每步都在乱跳（说明它只是在记住个别样本），就不管它。

效果？在模加法任务上，原来需要 10 万步的 Grokking，现在不到 2000 步就完成了。

### 其他加速方法

- **增大权重衰减**：最直接的方法，但太大会阻碍学习
- **增大学习率**：加速所有过程，但可能不稳定
- **数据增强**：创造更多训练样本，让记忆策略更昂贵
- **GrekTransfer**：从已经 Grok 的网络迁移嵌入层

## 远不止模运算：Grokking 无处不在

Grokking 最初在模运算这种"玩具任务"上被发现，但后续研究表明它可能是一个**普遍现象**。

2024 年 Rice 大学的研究证明：Grokking 在 CNN 训练 CIFAR-10 上发生，在 ResNet 训练 ImageNet 上发生，甚至在非神经网络模型（如高斯过程、带虚假特征的线性回归）上也发生。

最引人注目的是 Grokking 与大语言模型的联系：

- Ohio State 大学的研究发现，Transformer 学习复杂的隐式推理（如多跳组合推理）**只能通过 Grokking 实现**——充分 Grok 的小模型在推理任务上打败了 GPT-4-Turbo
- 2024 年的统一框架将 Grokking、双重下降（Double Descent）和 LLM 的涌现能力联系起来——它们可能共享相同的"电路竞争"动力学
- 2025-2026 年的研究在 LLM 预训练过程中观察到了"局部 Grokking"——模型在不同知识领域的泛化时间点不同

这意味着什么？当我们看到 GPT-4 在某个基准测试上的能力突然跳跃时，它可能正在经历一次 Grokking——记忆阶段终于让位给了真正理解底层规律的泛化电路。

## Grokking 给我们的启示

### 关于训练

- **不要太早停止**：如果你的模型过拟合了但你有理由相信任务有可学习的结构，继续训练（配合适当的正则化）可能带来惊喜
- **权重衰减不只是防过拟合**：它是驱动模型从记忆走向理解的核心动力
- **监控比表面更深的指标**：训练损失和测试损失之外，权重范数、Fourier 谱的稀疏度等"进度指标"能预测 Grokking 是否即将发生

### 关于理解智能

Grokking 可能是我们见过的最清晰的"理解从记忆中涌现"的案例。它暗示：

- **记忆和理解不是对立的，而是先后的**。先记住，再理解——前提是有足够的压力（正则化）和耐心（训练时间）
- **真正的理解总是更"便宜"的**。泛化电路之所以最终获胜，是因为它用更少的参数做更多的事。这和 Kolmogorov 复杂度的观点完美吻合：最好的模型是对数据的最短描述
- **跳跃式进步是结构发现的自然结果**。当一个足够优雅的解被找到时，它会迅速取代所有笨拙的替代方案

### 压缩即理解

从信息论的角度，Grokking 的过程就是网络的 Kolmogorov 复杂度从高（记忆 = 存储每个样本）到低（泛化 = 找到最短程序）的过程。记忆一百个答案需要一百份存储；理解一条规则只需要一份。

这与"压缩即智能"假说完美呼应——真正的智能不在于能记住多少，而在于能用多短的描述覆盖多大的世界。Grokking 就是这个原则在训练动力学中的具体体现。

## 下一篇预告

我们讲了模型如何在过拟合后突然"顿悟"。但还有另一个同样诡异的现象：**双重下降（Double Descent）**——随着模型越来越大，测试误差先下降、再上升（符合经典理论），然后诡异地再次下降。这跟 Grokking 有什么关系？它们是同一枚硬币的两面吗？下篇来揭开这个谜团。
