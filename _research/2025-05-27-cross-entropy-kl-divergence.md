---
title: "交叉熵与 KL 散度：训练 LLM 时我们到底在优化什么"
date: 2025-05-27
level: 3
series: "LLM 原理深度解析"
series_order: 15
series_total: 43
tags: [information-theory, cross-entropy, KL-divergence, loss-function, training]
summary: "从信息论的「惊讶度」出发，解释交叉熵损失为什么是训练语言模型的唯一合理选择，以及 KL 散度如何连接压缩、对齐与蒸馏"
---

# 交叉熵与 KL 散度：训练 LLM 时我们到底在优化什么

> 每次你看到 loss 曲线在下降，本质上是模型在学习如何更高效地"压缩"人类语言。这篇文章讲清楚这背后的数学为什么如此优美。

## 一个关于"惊讶"的故事

假设你住在北京，有人告诉你"明天太阳会从东边升起"。你一点都不惊讶——这几乎是确定事件，信息量为零。但如果有人说"明天北京下陨石雨了"，你会极度震惊——这个事件极其罕见，携带了巨大的信息量。

这就是 Claude Shannon 在 1948 年奠定信息论时的核心直觉：**一个事件的信息量，等于你对它的"惊讶程度"。** 越不可能发生的事情，一旦发生，带来的信息越多。

Shannon 用一个极简的数学公式捕捉了这个直觉：事件 $x$ 发生的信息量（也叫自信息）为：

$$I(x) = -\log p(x)$$

概率越小，$-\log p(x)$ 越大，你越"惊讶"。概率为 1 时，信息量为 0——完全不意外。

那如果我们想衡量一个**整体信息源**（比如中文语言）平均有多让人惊讶呢？只需对所有可能事件取期望：

$$H(p) = -\sum_x p(x) \log p(x)$$

这就是**熵（Entropy）**——一个概率分布的"平均惊讶度"，也是"平均不确定性"。

## 熵的直觉：为什么它等于"压缩的极限"

熵还有一个更实用的意义：**它是你压缩数据的理论极限**。

想象你需要给朋友发送一系列天气预报。如果北京 90% 的日子是晴天、10% 是雨天，你可以给"晴"分配一个很短的编码（比如 1 bit），给"雨"分配一个稍长的编码。平均下来，每条消息只需要约 0.47 bit——这正好等于这个分布的熵 $H(p)$。

反过来，如果天气完全随机（50% 晴 50% 雨），熵就是 1 bit——没有任何压缩空间，每条消息都需要完整的 1 bit。

**关键结论：熵 $H(p)$ 是用最优编码方案传输服从分布 $p$ 的消息所需的最少平均比特数。**

## 交叉熵：当你的"编码本"不完美时

现在问题来了：如果你不知道真实的天气分布 $p$，只能猜一个 $q$（比如你以为晴雨各半），然后基于 $q$ 来设计编码方案——你传输消息需要多少 bit？

答案就是**交叉熵（Cross-Entropy）**：

$$H(p, q) = -\sum_x p(x) \log q(x)$$

注意这里的微妙之处：消息仍然服从真实分布 $p$（真实世界不会因为你的错误假设而改变），但你用基于 $q$ 的编码来传输。因为你的编码不是为真实分布优化的，所以必然需要**更多**的 bit。

用人话说：**交叉熵衡量的是"用错误模型去描述真实世界的代价"。**

这正是训练语言模型时发生的事情：
- $p$ = 训练数据的真实分布（下一个 token 实际是什么）
- $q$ = 模型预测的分布（模型认为下一个 token 是什么）
- $H(p, q)$ = 交叉熵损失 = 模型当前有多"错"

## 三位一体：交叉熵、KL 散度、最大似然

这里藏着一个漂亮的数学等式，把三个看似不同的概念统一了：

$$H(p, q) = H(p) + D_{KL}(p \| q)$$

翻译成人话：

> 交叉熵 = 真实分布的熵（无法消除的不确定性）+ KL 散度（你的模型带来的额外浪费）

**KL 散度（Kullback-Leibler Divergence）** $D_{KL}(p \| q)$ 衡量的是：因为你用了 $q$ 而不是 $p$ 作为编码方案，你**额外**浪费了多少 bit。它永远 ≥ 0，当且仅当 $p = q$ 时等于 0。

<svg viewBox="0 0 650 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- H(p,q) box -->
  <rect x="20" y="30" width="180" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="110" y="58" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">交叉熵 H(p, q)</text>
  <text x="110" y="82" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="system-ui">"用 q 编码 p 的总代价"</text>
  <!-- = sign -->
  <text x="225" y="70" text-anchor="middle" fill="#ededf0" font-size="20" font-family="system-ui">=</text>
  <!-- H(p) box -->
  <rect x="250" y="30" width="160" height="70" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="330" y="58" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">熵 H(p)</text>
  <text x="330" y="82" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="system-ui">"不可消除的不确定性"</text>
  <!-- + sign -->
  <text x="435" y="70" text-anchor="middle" fill="#ededf0" font-size="20" font-family="system-ui">+</text>
  <!-- KL box -->
  <rect x="460" y="30" width="170" height="70" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="545" y="58" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">KL(p ‖ q)</text>
  <text x="545" y="82" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="system-ui">"模型带来的额外浪费"</text>
  <!-- Bottom explanation -->
  <rect x="80" y="150" width="490" height="100" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="325" y="178" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">训练时 H(p) 是常数（数据固定），因此：</text>
  <text x="325" y="205" text-anchor="middle" fill="#6e8eff" font-size="14" font-family="system-ui, monospace">最小化交叉熵 ≡ 最小化 KL 散度 ≡ 最大化似然</text>
  <text x="325" y="232" text-anchor="middle" fill="#9ca3af" font-size="12" font-family="system-ui">三种说法，同一件事：让模型尽可能接近真实分布</text>
</svg>

这个等式揭示了一个重要事实：**在训练过程中，$H(p)$ 是常数**（因为训练数据不变）。所以最小化交叉熵等价于最小化 KL 散度，而最小化 KL 散度又等价于最大化对数似然（Maximum Likelihood Estimation）。

三种看似不同的表述——信息论（最小化编码代价）、统计学（最小化分布差异）、概率论（最大化数据概率）——其实是**同一个优化目标的三副面孔**。

## 为什么偏偏是交叉熵？不能用 MSE 吗？

很多初学者会问：为什么不直接用均方误差（MSE）来衡量预测分布和真实分布的差异？

答案有几层：

**第一层：信息论的必然性。** 交叉熵直接衡量"编码效率"，天然适配概率分布的比较。MSE 把概率值当做普通数字来处理，忽略了概率的特殊结构（比如必须和为 1、取值在 0-1 之间）。

**第二层：梯度的行为。** 当模型对正确答案的预测概率很低时（比如只有 0.01），交叉熵的梯度很大，推动模型快速纠正错误。而 MSE 在这种情况下梯度反而可能很小（因为 sigmoid/softmax 输出的导数在极端值处饱和），导致学习缓慢。

**第三层：与 softmax 的天作之合。** 交叉熵对 softmax 输出求导，得到的梯度形式极其简洁：$\hat{y}_i - y_i$（预测值减真实值）。这不是巧合，而是因为 softmax + 交叉熵构成了指数族分布的自然参数化，梯度天然具有这种优美形式。

## 语言模型训练：逐 token 的交叉熵

在语言模型的训练中，损失函数具体长这样：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log q_\theta(x_t | x_{<t})$$

其中 $x_t$ 是第 $t$ 个位置的真实 token，$q_\theta(x_t | x_{<t})$ 是模型在看到前面所有 token 后对第 $t$ 个位置预测的概率。

逐 token 解读：对于序列中的每一个位置，模型输出一个词表大小的概率分布，我们只看真实 token 对应的那个概率，取 $-\log$，然后对所有位置取平均。

为什么取 $-\log$？因为好的预测（概率接近 1）应该贡献低损失，差的预测（概率接近 0）应该贡献高损失。$-\log$ 完美实现这一点：$-\log(0.9) \approx 0.1$，$-\log(0.01) \approx 4.6$。

**这就是 perplexity 的由来：** $PPL = e^{\mathcal{L}}$。如果交叉熵损失是 3.0 nats，perplexity 就是 $e^3 \approx 20$——直觉上相当于模型在每个位置都要从约 20 个等概率选项中猜测。

## KL 散度的不对称性：前向与反向

KL 散度有一个让很多人困惑的特性：**它不对称**。$D_{KL}(p \| q) \neq D_{KL}(q \| p)$。

这不是一个数学缺陷，而是一个深刻的设计选择，在 LLM 的不同应用场景中发挥着不同作用。

### 前向 KL：$D_{KL}(p \| q)$ — "宁可全覆盖"

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

注意这里的期望是对 $p$ 取的。这意味着：只要真实分布 $p(x) > 0$ 的地方，如果 $q(x)$ 接近 0，惩罚就趋于无穷大。

**行为特征：mode-covering（模式覆盖）。** 最小化前向 KL 会迫使 $q$ 在 $p$ 有概率质量的所有地方都给出非零概率。即使 $q$ 没法精确匹配 $p$ 的形状，它也会"伸展"自己去覆盖 $p$ 的所有模式。

**用途：这正是标准语言模型训练用的。** 我们不能让模型对任何真实出现过的 token 预测概率为 0——那会导致无穷大的损失。模型必须对所有可能性保持开放。

### 反向 KL：$D_{KL}(q \| p)$ — "宁可精确匹配"

$$D_{KL}(q \| p) = \sum_x q(x) \log \frac{q(x)}{p(x)}$$

现在期望是对 $q$ 取的。如果 $q(x) > 0$ 但 $p(x) = 0$（模型认为可能但实际不会发生），惩罚才会趋于无穷大。

**行为特征：mode-seeking（模式寻找）。** 模型会集中概率质量到 $p$ 的一个或几个高密度区域，放弃覆盖全部模式。宁可把一个模式匹配得很准，也不愿"摊薄"去覆盖所有模式。

**用途：** 这在变分推断（VAE）和知识蒸馏中常见。在 RLHF/DPO 对齐训练中，KL 惩罚项通常是 $D_{KL}(\pi_\theta \| \pi_{ref})$（反向 KL），防止对齐后的模型偏离参考模型太远。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="15" font-family="system-ui" font-weight="bold">前向 KL vs 反向 KL 的行为差异</text>
  <!-- Left: Forward KL -->
  <rect x="20" y="45" width="310" height="250" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="175" y="72" text-anchor="middle" fill="#6e8eff" font-size="14" font-family="system-ui" font-weight="bold">前向 KL: D_KL(p ‖ q)</text>
  <!-- Bimodal p distribution sketch -->
  <text x="55" y="100" fill="#9ca3af" font-size="11" font-family="system-ui">真实分布 p（双峰）:</text>
  <ellipse cx="100" cy="140" rx="30" ry="25" fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4"/>
  <ellipse cx="230" cy="140" rx="30" ry="25" fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="165" y="145" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">p</text>
  <!-- Wide q covering both -->
  <ellipse cx="165" cy="140" rx="120" ry="35" fill="none" stroke="#f59e0b" stroke-width="2"/>
  <text x="165" y="195" text-anchor="middle" fill="#f59e0b" font-size="11" font-family="system-ui">q 伸展覆盖两个峰</text>
  <text x="175" y="225" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="system-ui">✓ 覆盖全部模式</text>
  <text x="175" y="245" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="system-ui">✗ 在中间空隙也有概率质量</text>
  <text x="175" y="275" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">→ 语言模型预训练</text>
  <!-- Right: Reverse KL -->
  <rect x="370" y="45" width="310" height="250" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="525" y="72" text-anchor="middle" fill="#a78bfa" font-size="14" font-family="system-ui" font-weight="bold">反向 KL: D_KL(q ‖ p)</text>
  <!-- Bimodal p -->
  <text x="405" y="100" fill="#9ca3af" font-size="11" font-family="system-ui">真实分布 p（双峰）:</text>
  <ellipse cx="450" cy="140" rx="30" ry="25" fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4"/>
  <ellipse cx="580" cy="140" rx="30" ry="25" fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="515" y="145" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">p</text>
  <!-- Narrow q on one mode -->
  <ellipse cx="450" cy="140" rx="25" ry="20" fill="none" stroke="#f59e0b" stroke-width="2"/>
  <text x="525" y="195" text-anchor="middle" fill="#f59e0b" font-size="11" font-family="system-ui">q 锁定一个峰</text>
  <text x="525" y="225" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="system-ui">✓ 匹配精确</text>
  <text x="525" y="245" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="system-ui">✗ 忽略了另一个峰</text>
  <text x="525" y="275" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">→ RLHF/DPO 对齐约束</text>
</svg>

## 交叉熵在 LLM 全生命周期中的角色

理解了交叉熵和 KL 散度之后，让我们看看它们如何贯穿 LLM 的整个生命周期：

### 1. 预训练：最小化交叉熵 = 学习压缩语言

预训练的目标函数就是交叉熵。2024 年的研究（"Learning is Forgetting"，arxiv 2604.07569）进一步确认：LLM 预训练本质上是有损压缩，模型在信息瓶颈（Information Bottleneck）的意义下趋向最优压缩。

直觉上：一个 loss 为 2.0 nats 的模型，相当于每个 token 用 2.0 nats（约 2.9 bits）来编码。对比英语文本的经验熵（约 1.0-1.5 bits/character），模型还有改进空间。当模型的交叉熵逼近真实熵时，它就完美"理解"了语言的统计结构。

### 2. 知识蒸馏：KL 散度传递"暗知识"

在知识蒸馏中，学生模型的损失是：

$$\mathcal{L} = \alpha \cdot H(y_{hard}, q_{student}) + (1-\alpha) \cdot T^2 \cdot D_{KL}(q_{teacher}^{(T)} \| q_{student}^{(T)})$$

第一项是学生与硬标签（真实答案）的交叉熵，第二项是学生与教师软标签分布的 KL 散度。

为什么软标签有效？因为教师模型输出的概率分布包含**暗知识（dark knowledge）**：除了正确答案，其他选项之间的相对概率关系（比如"猫"和"狗"的相似度远高于"猫"和"桌子"）。温度 $T$ 提高时，分布变平滑，这些细微的关系被放大，学生模型能学到更多结构信息。

### 3. RLHF 对齐：KL 惩罚防止"跑偏"

在 RLHF 中，优化目标是：

$$\max_{\pi_\theta} \mathbb{E}[r(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

第一项最大化奖励（人类偏好），第二项是 KL 惩罚，确保对齐后的模型 $\pi_\theta$ 不会偏离预训练模型 $\pi_{ref}$ 太远。

为什么需要 KL 约束？因为没有它，模型会"reward hack"——找到奖励模型的漏洞，生成得分很高但实际质量很差的文本。KL 散度像一根弹性绳子，允许模型在奖励方向上移动，但拉得太远时把它拽回来。

$\beta$ 的选择是一门艺术：太大，模型几乎不变化，对齐无效；太小，模型 reward hack，输出退化。

### 4. Label Smoothing：故意模糊交叉熵的目标

标准训练中，真实分布是 one-hot（正确答案概率为 1，其余为 0）。Label Smoothing 将其修改为：

$$p_{smooth}(x) = (1-\epsilon) \cdot p_{one-hot}(x) + \epsilon / V$$

其中 $\epsilon$ 通常取 0.1，$V$ 是词表大小。

为什么要这样做？因为 one-hot 标签要求模型输出的 logit 趋于无穷大（softmax 才能输出 1.0），这导致过度自信和泛化能力下降。Label Smoothing 本质上是在优化一个"松弛版"的交叉熵，鼓励模型给非正确答案也保留一点概率——这正是 KL 散度关于分布光滑性的直觉在起作用。

## 信息论视角的深层洞察

最后，让我们把视角拉远，看看交叉熵损失揭示的更深层图景：

**训练 LLM = 逼近语言的真实熵。** 一个完美的语言模型，其交叉熵损失会等于人类语言的真实熵——这是不可逾越的理论下限。任何比这更低的"损失"要么意味着过拟合，要么意味着你的评估有 bug。

**Perplexity 下降 = 压缩能力增强。** 从 GPT-2 到 GPT-4，perplexity 的下降等价于压缩比的提升。模型越好，用越少的 bit 就能表示语言——这正是"压缩即智能"假说的基础。

**交叉熵的局限性。** 交叉熵优化的是 token 级别的概率匹配，但人类对文本质量的判断是整体的（连贯性、事实性、风格）。这就是为什么仅靠交叉熵预训练不够，还需要 RLHF/DPO 这类基于 KL 散度的对齐方法来弥补 token 级目标与人类偏好之间的鸿沟。

## 这意味着什么

交叉熵和 KL 散度不只是"损失函数"——它们是连接信息论、统计学和机器学习的桥梁。当你理解了：

- **交叉熵 = 用错误模型编码真实世界的代价**
- **KL 散度 = 两个分布之间的"信息距离"**
- **最小化交叉熵 = 最小化 KL 散度 = 最大化似然 = 学习压缩**

你就掌握了理解 LLM 训练、蒸馏、对齐的统一框架。下次看到 loss 曲线时，你看到的不再是一条抽象的曲线，而是模型在信息论意义上逐步逼近人类语言真实结构的过程。

## 下一篇预告

如果压缩能力等价于智能，那么"Perplexity 与压缩的等价性"就是一个值得深入的话题——Shannon 的源编码定理如何精确地告诉我们 LLM 的压缩效率，以及 Hutter Prize（压缩竞赛）为什么和 AI 研究殊途同归。
