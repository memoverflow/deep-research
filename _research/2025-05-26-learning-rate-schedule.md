---
title: "学习率调度的艺术：Warmup、Cosine Decay 与 WSD 背后的原理"
date: 2025-05-26
level: 3
series: "LLM 原理深度解析"
series_order: 14
series_total: 14
tags: [learning-rate, optimization, warmup, cosine-decay, WSD, training]
summary: "为什么训练开始时要小心翼翼地加速？为什么学习率要像日落一样缓慢下降？从损失景观的几何直觉到现代 WSD 调度的工程智慧。"
---

# 学习率调度的艺术：Warmup、Cosine Decay 与 WSD 背后的原理

> 学习率是深度学习中最重要的单个超参数。选错了，再好的架构也是废铁。但比选一个固定值更重要的是——如何在训练过程中动态调整它。

## 故事从一个灾难开始

想象你刚买了一辆超级跑车，引擎冷得冰凉，你一脚油门踩到底。会发生什么？引擎可能直接熄火，变速箱可能打滑，甚至可能把传动轴扭断。

训练一个大语言模型的前几百步，面临的就是完全相同的困境。

模型的所有参数都是随机初始化的——就像一个从未见过世界的婴儿，对一切一无所知。在这个状态下，梯度信号是嘈杂的、方向是混乱的、损失景观是崎岖的。如果你此时就用一个很大的学习率去更新参数，模型会做出巨大的、方向错误的跳跃，可能直接"飞"到损失景观中无法恢复的区域——loss 爆炸、NaN、训练彻底崩溃。

但另一方面，如果你一直用一个很小的学习率，训练会慢得令人绝望。100 万步都走不到一个好的最小值，而你的 GPU 集群每小时烧掉几万美元。

这就是学习率调度（Learning Rate Schedule）要解决的核心问题：**如何在训练的不同阶段，用不同的"油门力度"，既不翻车、又不龟速？**

现代 LLM 训练中，几乎所有成功的模型——GPT-3、LLaMA、Chinchilla、Mistral——都使用某种形式的学习率调度。最经典的模式是三个字：**先升后降**。先小心地加速（Warmup），达到巡航速度后，再缓慢减速直到训练结束（Decay）。

但为什么是这个模式？为什么不能一直用固定学习率？"先升后降"的每个阶段到底在做什么？让我们从物理直觉开始，一步步拆解。

## 第一幕：Warmup——为什么要先慢后快

### 问题：训练开始时到底出了什么事？

训练的前几步有三个同时存在的麻烦：

**麻烦一：Adam 优化器的"冷启动"问题。** Adam 维护两个滑动平均：梯度的均值 $m_t$ 和梯度平方的均值 $v_t$。它们都初始化为零。虽然 Adam 有偏差修正（bias correction），但在第一步，修正因子是 $1/(1-\beta_2^1) = 1/(1-0.999) = 1000$。这意味着 Adam 在第一步的行为完全由一个单独的梯度样本决定——这就像看了一分钟新闻就对世界政治做出全面判断。

**麻烦二：梯度方向极度不稳定。** 随机初始化的模型还没学到任何结构。不同 batch 计算出的梯度可能指向几乎相反的方向。这时大步更新，有一半概率是在往错误方向狂奔。

**麻烦三：损失景观极度崎岖。** 随机初始化点附近的损失景观曲率（sharpness）通常很高。高曲率意味着"悬崖"多——大步一跨就可能跌落，触发 loss 爆炸。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Loss landscape curve -->
  <path d="M 50 200 Q 100 40, 150 180 Q 180 240, 220 160 Q 260 80, 300 140 Q 340 200, 380 120 Q 420 40, 460 100 Q 500 160, 540 80 Q 580 20, 620 60 Q 650 90, 680 70" fill="none" stroke="#6e8eff" stroke-width="2"/>
  <!-- Starting point -->
  <circle cx="150" cy="180" r="6" fill="#ef4444"/>
  <text x="150" y="220" text-anchor="middle" fill="#ef4444" font-size="12" font-family="system-ui">随机初始化</text>
  <!-- Large step arrow (bad) -->
  <line x1="156" y1="178" x2="290" y2="138" stroke="#ef4444" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arr1)"/>
  <text x="220" y="145" text-anchor="middle" fill="#ef4444" font-size="11" font-family="system-ui">大学习率→飞出去</text>
  <!-- Small step arrow (good) -->
  <line x1="156" y1="182" x2="195" y2="192" stroke="#34d399" stroke-width="2" marker-end="url(#arr1)"/>
  <text x="200" y="245" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">小学习率→稳步探索</text>
  <!-- Good minimum -->
  <circle cx="540" cy="80" r="6" fill="#34d399"/>
  <text x="540" y="60" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">好的最小值</text>
  <!-- Labels -->
  <text x="360" y="270" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">参数空间</text>
  <text x="30" y="140" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" transform="rotate(-90 30 140)">Loss</text>
</svg>

### 直觉：Warmup 在做什么？

2024 年 NeurIPS 的一篇论文（Kalra & Barkeshli, "Why Warmup the Learning Rate?"）给出了一个优雅的解释：

> Warmup 的核心作用不是"让模型慢慢学"，而是**把模型推到损失景观中更平坦、条件数更好的区域**。

这是什么意思？想象损失景观像一个山脉。随机初始化可能把你放在一个尖锐的山脊上——稍微偏一点就会跌落。Warmup 的作用是：在学习率还很小的时候，模型做的微小更新帮助它从尖锐的山脊滑到相对平坦的山谷。一旦到了平坦区域，损失对参数变化不那么敏感了，这时再加大学习率就安全了。

用物理学的语言说：warmup 降低了损失函数的 **sharpness**（Hessian 最大特征值），使得系统能够容忍更大的学习率而不失稳。

论文还发现了两个有趣的机制：

1. **渐进锐化（Progressive Sharpening）**：在某些初始化下，模型的 sharpness 会自然增长。此时 warmup 和自然锐化是"竞争关系"——warmup 试图让模型找到平坦区域，而自然演化在增加曲率。这解释了为什么 warmup 步数太短可能不够。

2. **损失弹射（Loss Catapult）**：当学习率超过当前 sharpness 所能承受的阈值时，loss 会突然跳高——但这次跳高反而迫使模型跳到 sharpness 更低的区域。这像是"以毒攻毒"——短期的不稳定换来了长期的稳定。

### 实践：Warmup 怎么做？

最常见的实现是 **线性 Warmup**：

$$\eta_t = \eta_{\text{target}} \cdot \frac{t}{T_{\text{warmup}}}$$

从接近零线性增长到目标学习率。典型的 warmup 步数：

| 模型 | 总训练步数 | Warmup 步数 | 占比 |
|------|-----------|------------|------|
| GPT-3 175B | 300K | 375 | 0.1% |
| LLaMA-2 70B | ~500K | 2000 | 0.4% |
| Chinchilla 70B | 1.4M | 2000 | 0.14% |
| 小模型微调 | 10K | 100-500 | 1-5% |

一个经验法则：预训练通常 warmup 1000-2000 步（占总步数不到 1%），微调可能 5-10%。关键不是步数本身，而是给 Adam 的二阶矩估计足够的时间来收集可靠的统计信息。

## 第二幕：Cosine Decay——为什么学习率要像日落

### 问题：为什么不能一直用峰值学习率？

直觉上，既然大学习率训练得快，为什么不一直保持最大值？

答案涉及 **收敛精度** 和 **泛化** 两个维度：

**收敛精度**：大学习率意味着大步长。在训练后期，模型已经接近一个好的最小值，此时需要小步精确调整。继续用大步长会让模型在最小值附近来回振荡，永远无法真正"坐进"最低点。就像你开车到了目的地附近，继续踩油门只会让你反复冲过停车位。

**泛化（找到平坦最小值）**：大学习率本身有一个好处——它自带隐式正则化，因为大步长的"噪声"效果会让模型跳出尖锐的最小值（尖锐最小值虽然训练 loss 低，但泛化差）。然而在训练末期，我们希望模型安定在一个平坦且低的最小值中。这需要逐渐降低"温度"——减小学习率。

这和物理中的 **模拟退火（simulated annealing）** 是同一个道理：先在高温下充分探索（大学习率），再慢慢降温让系统凝固在能量最低的状态（小学习率）。

### 核心直觉：为什么是 Cosine 而不是线性？

2017 年，Loshchilov 和 Hutter 在 SGDR 论文中提出了余弦退火（Cosine Annealing），随后被整个 LLM 社区采用。公式很简单：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)$$

但为什么是余弦形状而不是简单的线性衰减？

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="60" y1="250" x2="660" y2="250" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arr2)"/>
  <line x1="60" y1="250" x2="60" y2="30" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arr2)"/>
  <text x="360" y="285" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">训练步数</text>
  <text x="25" y="140" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" transform="rotate(-90 25 140)">学习率</text>
  <!-- Cosine curve -->
  <path d="M 60 50 Q 120 50, 180 55 Q 240 65, 300 90 Q 360 130, 420 170 Q 480 210, 540 235 Q 600 245, 640 248" fill="none" stroke="#6e8eff" stroke-width="2.5"/>
  <text x="500" y="195" fill="#6e8eff" font-size="12" font-family="system-ui">Cosine Decay</text>
  <!-- Linear curve -->
  <path d="M 60 50 L 640 248" fill="none" stroke="#a78bfa" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="500" y="135" fill="#a78bfa" font-size="12" font-family="system-ui">Linear Decay</text>
  <!-- Annotations -->
  <rect x="80" y="42" width="120" height="24" rx="4" fill="#1e1e2a" stroke="#34d399" stroke-width="1"/>
  <text x="140" y="58" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">前期：下降慢</text>
  <rect x="440" y="215" width="130" height="24" rx="4" fill="#1e1e2a" stroke="#34d399" stroke-width="1"/>
  <text x="505" y="231" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">后期：也下降慢</text>
  <rect x="250" y="85" width="130" height="24" rx="4" fill="#1e1e2a" stroke="#ef4444" stroke-width="1"/>
  <text x="315" y="101" text-anchor="middle" fill="#ef4444" font-size="11" font-family="system-ui">中期：下降最快</text>
  <!-- eta_max and eta_min labels -->
  <text x="48" y="55" text-anchor="end" fill="#ededf0" font-size="11" font-family="system-ui">η_max</text>
  <text x="48" y="252" text-anchor="end" fill="#ededf0" font-size="11" font-family="system-ui">η_min</text>
</svg>

Cosine 形状的关键优势在于它的"两头慢、中间快"：

1. **前期下降慢**：刚从峰值开始，学习率几乎不变。模型可以在大学习率下充分探索，不会因为过早衰减而困在局部最小值。

2. **中期下降快**：训练中期，大方向已经确定，模型需要从探索模式切换到利用模式。Cosine 的中段斜率最大，快速完成这个过渡。

3. **后期下降慢**：接近最小学习率时，变化趋于平缓。模型可以在低学习率下做精细调整，慢慢"沉淀"到最小值底部。

相比之下，线性衰减在每个阶段的下降速度相同——前期降得太快（浪费了探索时间），后期又不够平稳。

### 一个更深的视角：累积学习率与幂律

2025 年的研究发现了一个惊人的联系：如果你计算"到第 $t$ 步为止所有学习率的累积和" $\sum_{i=1}^{t} \eta_i$，loss 的下降与这个累积学习率之间存在幂律关系。Cosine 调度恰好使得这个累积值的增长曲线在训练全程都保持"最优效率"——在每个时刻都恰好平衡了"探索"与"收敛"的需要。

翻译回人话：Cosine 衰减不是随便选的一条漂亮曲线，而是在"总共能走多远"和"每一步走多稳"之间找到了接近最优的平衡。

## 第三幕：WSD——打破"必须预知终点"的枷锁

### 问题：Cosine 的致命缺陷

Cosine 调度有一个巨大的实际问题：**你必须在训练开始前就决定总步数 $T$。**

为什么这是问题？因为大模型训练往往持续数周甚至数月，期间你可能发现：
- 模型还在快速进步，想继续训练更久
- 数据加载遇到问题，需要延长
- 计算资源临时增加了，想多训一些
- 需要在中途接上新数据做持续训练

但一旦 Cosine 调度开始运行，学习率的整条曲线就被锁死了。如果你决定训练更长，要么从头重新训（代价巨大），要么在学习率已经接近零的时候继续（几乎没有学习能力）。

这就像你给汽车设定了"到北京自动停车"，但开到半路发现其实应该去上海。

### 核心直觉：WSD 的三段式智慧

2024 年，MiniCPM 团队提出了 Warmup-Stable-Decay（WSD）调度，优雅地解决了这个问题：

```
学习率
  ↑
  |     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
  |    /                         \
  |   /                           \
  |  /    Warmup → Stable → Decay  \
  | /                                 \__
  +————————————————————————————————————————→ 训练步数
```

三个阶段：
1. **Warmup**：和以前一样，线性升到峰值
2. **Stable**：保持峰值学习率不变，想训多久训多久
3. **Decay**：需要结束时，执行一个快速衰减（通常用 $1 - \sqrt{t/T_{\text{decay}}}$ 或指数衰减）

WSD 的革命性在于：Stable 阶段可以无限延长。你可以在任何想停的地方"分叉"一个 Decay 分支，得到一个高质量的最终模型——而主干继续以恒定学习率训练。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="50" y1="230" x2="660" y2="230" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arr3)"/>
  <line x1="50" y1="230" x2="50" y2="30" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arr3)"/>
  <!-- Warmup -->
  <line x1="50" y1="220" x2="100" y2="60" stroke="#34d399" stroke-width="2.5"/>
  <text x="75" y="250" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">Warmup</text>
  <!-- Stable -->
  <line x1="100" y1="60" x2="450" y2="60" stroke="#6e8eff" stroke-width="2.5"/>
  <text x="275" y="45" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">Stable（可无限延长）</text>
  <!-- Decay -->
  <path d="M 450 60 Q 500 60, 530 100 Q 560 150, 580 200 Q 590 220, 600 225" fill="none" stroke="#ef4444" stroke-width="2.5"/>
  <text x="550" y="250" text-anchor="middle" fill="#ef4444" font-size="11" font-family="system-ui">Decay</text>
  <!-- Branch points -->
  <circle cx="250" cy="60" r="5" fill="#a78bfa"/>
  <path d="M 250 60 Q 280 60, 300 90 Q 320 130, 330 160 Q 340 180, 345 195" fill="none" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="320" y="210" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">分支1</text>
  <circle cx="350" cy="60" r="5" fill="#a78bfa"/>
  <path d="M 350 60 Q 380 60, 400 90 Q 420 130, 430 160 Q 440 180, 445 195" fill="none" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="420" y="210" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">分支2</text>
  <!-- Arrow showing continuation -->
  <text x="480" y="85" fill="#ededf0" font-size="11" font-family="system-ui">主干可以继续...</text>
  <line x1="450" y1="60" x2="500" y2="60" stroke="#6e8eff" stroke-width="1.5" stroke-dasharray="4,3"/>
</svg>

### 为什么 Stable 阶段有效？

一个合理的疑问：恒定学习率难道不会导致模型在最小值附近振荡，无法收敛？

答案是：**确实如此——但这不一定是坏事。**

ICLR 2025 的论文（"Understanding WSD: A River Valley Loss Landscape"）提出了一个漂亮的类比。把损失景观想象成一个**河谷**：

- 河谷有一个沿着河流方向的缓坡（代表真正有意义的 loss 下降方向）
- 河谷横截面是 U 形的（代表"振荡"方向）

在 Stable 阶段，恒定学习率让模型沿着河谷方向持续前进（loss 稳步下降），同时在横截面方向来回振荡。这个振荡并不浪费——它实际上在帮模型探索河谷中不同的路径。

当 Decay 阶段开始时，振荡被压制，模型安定在河谷底部的最低轨迹上——这最后的 loss 下降就是 decay 带来的额外收益。

### WSD vs Cosine：实际对比

MiniCPM 的实验表明，WSD 在最终性能上与精心调参的 Cosine 调度基本持平，但在灵活性上碾压：

| 特性 | Cosine | WSD |
|------|--------|-----|
| 需要预知总步数 | ✅ 必须 | ❌ 不需要 |
| 支持持续训练 | ❌ 困难 | ✅ 天然支持 |
| 中途取模型 | ❌ 只有最后一个好 | ✅ 任何时候 branch decay |
| 调参难度 | 较高（$T$ 很敏感）| 较低（stable 阶段自由） |
| 最终 loss | 基准 | ≈ 基准（±0.1%）|

JetMoE、DeepSeek、Qwen 等新一代模型也在采用 WSD 或其变体。

## 第四幕：不衰减行不行？——一个反直觉的发现

2026 年 3 月的一篇新论文提出了一个有趣的观点：**如果你的目标不是预训练 loss 最低，而是下游微调效果最好——可能根本不需要学习率衰减。**

他们的发现是：学习率衰减确实降低了预训练 loss，但这个"好处"来自模型过度拟合到当前训练数据分布。当你随后做 SFT（监督微调）时，衰减模型反而不如恒定学习率训练的模型灵活——后者更容易适应新任务。

这就像一个运动员：严格按固定计划训练到极致（cosine decay 到 loss 最低），可能反而不如一个保持"通用体能"的运动员在面对新项目时适应得快。

当然，这是一个很新的结论，需要更多验证。目前的主流实践仍然是使用衰减。

## 数学附录：三种调度的精确公式

对于想在代码中实现这些调度的读者，这里是精确公式。

### Linear Warmup

$$\eta_t = \eta_{\text{target}} \cdot \min\left(1, \frac{t}{T_w}\right)$$

$T_w$ 是 warmup 步数。简单，鲁棒，几乎没有需要调的参数。

### Cosine Decay（含 Warmup）

$$\eta_t = \begin{cases} \eta_{\text{max}} \cdot \frac{t}{T_w} & \text{if } t \leq T_w \\[6pt] \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi(t - T_w)}{T - T_w}\right)\right) & \text{if } t > T_w \end{cases}$$

典型参数：$\eta_{\min} = 0.1 \cdot \eta_{\max}$（GPT-3 用的是 $\eta_{\min} = 0$）。

### WSD

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_w} & \text{Warmup: } t \leq T_w \\[4pt] \eta_{\max} & \text{Stable: } T_w < t \leq T_s \\[4pt] \eta_{\max} \cdot f\left(\frac{t - T_s}{T_d}\right) & \text{Decay: } t > T_s \end{cases}$$

其中衰减函数 $f$ 常用 $f(x) = 1 - \sqrt{x}$（MiniCPM 用的）或 $f(x) = \frac{1}{2}(1 + \cos(\pi x))$（cosine 衰减段）。衰减段通常占总训练的 10-20%。

## 这意味着什么

学习率调度不是一个随便选择的超参数——它编码了我们对训练动态的深层理解：

1. **训练开始时模型脆弱**（warmup 保护它）
2. **训练中期需要大胆探索**（峰值学习率维持住）
3. **训练末期需要精确收敛**（decay 把模型推入最小值）
4. **灵活性比最优性更重要**（WSD 牺牲微小的最优性换来巨大的实用价值）

如果你正在训练自己的模型，一个安全的默认选择是：
- **微调**：Linear warmup 5-10% + Cosine decay（标准且经过验证）
- **预训练**：WSD（灵活、适合不确定训练时长）
- **Warmup 步数**：1000-2000 步或总步数的 1%，取较大者

学习率调度是训练稳定性的"看不见的手"——做对了你不会注意到它，做错了训练直接崩溃。理解它背后的原理，是成为一个合格的 LLM 训练者的必修课。

## 下一篇预告

我们讲了学习率"怎么调"，但还没回答一个更根本的问题：**学习率到底该设多大？** 当你把模型从 7B 放大到 70B，学习率应该怎么变？当你把 batch size 翻倍，学习率要跟着翻倍吗？这涉及到一个精妙的理论——µP（Maximal Update Parameterization），它让超参数可以从小模型直接迁移到大模型。下次我们来拆解这个魔法。
