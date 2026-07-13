---
title: "采样的艺术：Temperature、Top-k、Top-p 与 Min-p 如何控制 LLM 的创造力"
date: 2025-06-29
level: 3
series: "LLM 原理深度解析"
series_order: 18
series_total: 39
tags: [sampling, decoding, temperature, top-k, top-p, min-p, nucleus-sampling, information-theory]
summary: "从物理学的玻尔兹曼分布到现代 LLM 的采样策略：理解 temperature、top-k、top-p、min-p 如何在创造力与连贯性之间走钢丝。"
---

# 采样的艺术：Temperature、Top-k、Top-p 与 Min-p 如何控制 LLM 的创造力

> 每次你调用 ChatGPT、Claude 或者本地跑 Llama，模型都在做一件事：从几万个候选词中选出下一个 token。这个"选"的过程，决定了输出是刻板无趣还是天马行空——而你手里的 temperature、top-p 这些参数，就是调节旋钮。

## 一切从一个悖论开始

假设你训练了一个完美的语言模型，它精确地学会了人类语言的概率分布。现在你要用它来生成一段文字。最自然的想法是什么？

**每一步都选概率最高的那个词。**

这叫贪心解码（greedy decoding）。直觉上它应该产生"最像人话"的文本——毕竟每个词都是最可能的选择。但 2019 年，Holtzman 等人在一篇题为 "The Curious Case of Neural Text Degeneration" 的论文中揭示了一个令人困惑的现象：

> 贪心解码产生的文本，读起来反而最不像人写的。

它的典型症状是**退化循环**——模型陷入重复，反复输出同一句话或同一个短语，像唱片卡针一样。更诡异的是，这些重复序列的概率确实很高（模型认为它们很合理），但人类绝不会这么写。

这个悖论揭示了一个深刻的道理：**人类语言不是概率最大化的产物。** 我们说话写字时，并不是每一步都选"最安全"的词。我们会出其不意，会用修辞，会为了表达力而选择次优的词。人类文本在概率分布中并不住在"山顶"，而是分布在山腰——足够合理，但不是最大概率。

这就是为什么我们需要**采样**：不是选最高概率的词，而是按概率分布随机抽取。但纯随机采样也有问题——它会采到概率极低的荒谬选项。于是，各种采样策略应运而生，本质上都在回答同一个问题：

**如何从"合理但不呆板"的范围内选词？**

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Axis -->
  <line x1="80" y1="220" x2="650" y2="220" stroke="#3a3a4a" stroke-width="1.5"/>
  <line x1="80" y1="220" x2="80" y2="30" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="360" y="260" text-anchor="middle" fill="#8a8a9a" font-size="12" font-family="system-ui">Token 排名 →</text>
  <text x="30" y="130" text-anchor="middle" fill="#8a8a9a" font-size="12" font-family="system-ui" transform="rotate(-90,30,130)">概率</text>
  <!-- Distribution curve -->
  <path d="M 100 50 Q 130 55 160 90 Q 190 125 220 150 Q 280 180 350 200 Q 450 212 600 218" stroke="#6e8eff" stroke-width="2.5" fill="none"/>
  <!-- Greedy zone -->
  <rect x="90" y="40" width="40" height="180" rx="4" fill="#ff6b6b" fill-opacity="0.15" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4"/>
  <text x="110" y="25" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">贪心</text>
  <!-- Good zone -->
  <rect x="90" y="40" width="200" height="180" rx="4" fill="#34d399" fill-opacity="0.08" stroke="#34d399" stroke-width="1" stroke-dasharray="4"/>
  <text x="190" y="245" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">✓ 采样的理想范围</text>
  <!-- Danger zone -->
  <rect x="400" y="40" width="220" height="180" rx="4" fill="#fbbf24" fill-opacity="0.08" stroke="#fbbf24" stroke-width="1" stroke-dasharray="4"/>
  <text x="510" y="245" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">⚠️ 低概率"尾巴"</text>
  <!-- Human text annotation -->
  <path d="M 140 85 Q 180 70 250 95" stroke="#ededf0" stroke-width="1" fill="none" stroke-dasharray="3"/>
  <text x="260" y="80" fill="#ededf0" font-size="11" font-family="system-ui">← 人类文本通常在这里</text>
</svg>

## Temperature：从玻尔兹曼到你的 API 参数

### 一个来自 19 世纪物理学的概念

"Temperature"这个名字不是比喻——它真的来自物理学。1868 年，玻尔兹曼（Boltzmann）推导出了描述气体分子能量分布的公式：一个系统中，粒子处于能量 $E_i$ 状态的概率正比于 $e^{-E_i / kT}$，其中 $T$ 就是温度。

温度高→分子活跃→各种能量状态都可能出现；温度低→分子冷却→集中在低能量状态。

LLM 中的 softmax 函数与玻尔兹曼分布**在数学上完全相同**。模型输出的 logit 值 $z_i$ 就是"能量"（带负号），temperature $T$ 就是温度：

$$P(token_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

### 直觉：温度在做什么？

想象你在自助餐厅选菜。菜品按"受欢迎程度"排列：

- **T = 0**（绝对零度）：你永远选最受欢迎的那道菜。每次都是。确定性的。无聊的。
- **T = 1**（标准温度）：你按正常偏好随机选——受欢迎的菜更可能被选中，但偶尔也会尝试冷门菜。
- **T = 2**（高温）：你变得冒险了，冷门菜和热门菜被选中的概率差距缩小。
- **T → ∞**（极高温）：你闭着眼随机指——所有菜概率相同，完全随机。

从数学上看：
- 当 $T → 0$ 时，概率分布坍缩成 one-hot，最大 logit 对应的 token 概率趋向 1
- 当 $T = 1$ 时，就是标准 softmax 输出
- 当 $T → ∞$ 时，分布趋向均匀分布，所有 token 等概率

### 温度的信息论含义

温度实际上在调节分布的**熵（entropy）**。熵衡量的是"不确定性"：

- 低温 → 低熵 → 输出几乎确定 → 重复、安全
- 高温 → 高熵 → 输出高度随机 → 创意、但可能胡言乱语

有个精确的关系：temperature 把原始分布的熵从 $H_{T=1}$ 连续调节到 $\log|V|$（词表大小的对数，即均匀分布的最大熵）。你可以把温度旋钮理解为一个"熵放大器"。

### 温度的局限

温度是"全局"调节——它同等地拉平所有 token 的概率差距。这带来一个问题：

当模型对下一个 token 很确定时（比如 "the United States of" 后面几乎一定是 "America"），提高温度会不必要地引入噪声。但当模型不确定时（比如开放式故事的下一句），同样的温度可能还不够让它变得有创意。

温度是一把大锤，不是手术刀。这就是为什么我们需要截断型策略。

## Top-k：最简单的截断

### 问题：尾巴太长了

一个词表有 32000 到 128000 个 token 的模型，在每一步的概率分布中，绝大多数 token 的概率接近于零。纯随机采样有时会抽到这些"尾巴"里的荒谬选项——比如在一段正常的英文中间突然蹦出一个罕见的 Unicode 字符。

### Top-k 的想法

2018 年，Fan 等人在 "Hierarchical Neural Story Generation" 论文中提出了一个朴素但有效的方案：**只保留概率最高的 k 个 token，其余的概率直接设为零，然后在这 k 个中重新归一化后采样。**

比如 k=50，意味着不管原始分布长什么样，每步只从前 50 个最可能的 token 中选。这直接砍掉了危险的尾巴。

### Top-k 的致命缺陷

问题在于"k"是固定的。但语言概率分布的形状变化极大：

- **确定性上下文**：比如 "Barack" 后面，"Obama" 的概率可能超过 95%。此时候选集真正有意义的可能只有 2-3 个 token。设 k=50 会引入 47 个本不该出现的选项。
- **开放性上下文**：比如 "The recipe calls for" 后面，合理的食材可能有几百种。设 k=50 会不必要地排除掉很多完全合理的选项。

一个固定的 k 不可能同时适应这两种极端情况。Top-k 是第一代截断方法——有效，但粗糙。

## Top-p（核采样）：让分布自己决定边界

### 核心直觉

2019 年，Holtzman 等人在那篇揭示文本退化问题的论文中，同时给出了解决方案：**nucleus sampling（核采样），也叫 top-p 采样。**

想法很直接：与其固定保留多少个 token，不如**固定保留多少概率质量**。

设阈值 p = 0.9，意思是：按概率从高到低排列 token，依次累加它们的概率，直到累计概率达到 0.9。这些被选中的 token 构成"核"（nucleus），从核内重新归一化后采样。

### 为什么这比 top-k 好？

因为核的大小是自适应的：

- 当模型很确定时（概率集中在少数 token 上），核可能只有 3-5 个 token——自动变保守。
- 当模型不确定时（概率分散在很多 token 上），核可能有 200 个 token——自动变开放。

这正是 top-k 做不到的：**让候选集大小随上下文的不确定性自动调节。**

### 数学定义

给定一个按概率降序排列的 token 序列 $x_{(1)}, x_{(2)}, ..., x_{(|V|)}$（其中 $P(x_{(1)}) \geq P(x_{(2)}) \geq ...$），top-p 的核定义为最小的集合 $V^{(p)}$ 使得：

$$\sum_{x_i \in V^{(p)}} P(x_i) \geq p$$

然后从 $V^{(p)}$ 中按重新归一化的概率采样。

### Top-p 的软肋

尽管 top-p 是目前最广泛使用的采样方法（OpenAI、Anthropic 等 API 的默认配置都支持），它在**高温度**下有一个渐渐暴露的问题。

当你把温度调高时，原本集中的概率分布被拉平。这意味着达到 p=0.9 的累计阈值需要纳入更多 token——包括一些在原始分布中概率很低、被高温"抬"上来的 token。结果：高温 + top-p 组合容易生成不连贯的文本。

换句话说，top-p 的截断阈值是"绝对"的——它不关心最高概率 token 有多高，只看累积概率是否达标。这在模型信心波动剧烈时成为弱点。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Top-k vs Top-p 的自适应对比</text>
  <!-- Left: peaked distribution -->
  <text x="175" y="55" text-anchor="middle" fill="#8a8a9a" font-size="11" font-family="system-ui">确定性上下文（"Barack ___"）</text>
  <rect x="30" y="65" width="290" height="120" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <!-- Bars -->
  <rect x="60" y="80" width="30" height="90" rx="3" fill="#6e8eff" fill-opacity="0.8"/>
  <text x="75" y="185" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Obama</text>
  <rect x="100" y="150" width="30" height="20" rx="3" fill="#6e8eff" fill-opacity="0.5"/>
  <text x="115" y="185" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Jr</text>
  <rect x="140" y="158" width="30" height="12" rx="3" fill="#6e8eff" fill-opacity="0.3"/>
  <rect x="180" y="162" width="30" height="8" rx="3" fill="#6e8eff" fill-opacity="0.2"/>
  <rect x="220" y="165" width="30" height="5" rx="3" fill="#6e8eff" fill-opacity="0.15"/>
  <rect x="260" y="166" width="30" height="4" rx="3" fill="#6e8eff" fill-opacity="0.1"/>
  <!-- Top-p bracket -->
  <path d="M 55 195 L 55 200 L 135 200 L 135 195" stroke="#34d399" stroke-width="1.5" fill="none"/>
  <text x="95" y="215" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">top-p=0.9 → 2 tokens</text>
  <!-- Top-k bracket -->
  <path d="M 55 225 L 55 230 L 295 230 L 295 225" stroke="#ff6b6b" stroke-width="1.5" fill="none"/>
  <text x="175" y="245" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">top-k=50 → 50 tokens（太多！）</text>
  <!-- Right: flat distribution -->
  <text x="525" y="55" text-anchor="middle" fill="#8a8a9a" font-size="11" font-family="system-ui">开放上下文（"食谱需要 ___"）</text>
  <rect x="380" y="65" width="290" height="120" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <!-- Bars - more uniform -->
  <rect x="400" y="130" width="22" height="40" rx="3" fill="#6e8eff" fill-opacity="0.7"/>
  <rect x="427" y="133" width="22" height="37" rx="3" fill="#6e8eff" fill-opacity="0.65"/>
  <rect x="454" y="136" width="22" height="34" rx="3" fill="#6e8eff" fill-opacity="0.6"/>
  <rect x="481" y="138" width="22" height="32" rx="3" fill="#6e8eff" fill-opacity="0.55"/>
  <rect x="508" y="140" width="22" height="30" rx="3" fill="#6e8eff" fill-opacity="0.5"/>
  <rect x="535" y="142" width="22" height="28" rx="3" fill="#6e8eff" fill-opacity="0.45"/>
  <rect x="562" y="144" width="22" height="26" rx="3" fill="#6e8eff" fill-opacity="0.4"/>
  <rect x="589" y="146" width="22" height="24" rx="3" fill="#6e8eff" fill-opacity="0.35"/>
  <rect x="616" y="148" width="22" height="22" rx="3" fill="#6e8eff" fill-opacity="0.3"/>
  <!-- Top-p bracket -->
  <path d="M 395 195 L 395 200 L 645 200 L 645 195" stroke="#34d399" stroke-width="1.5" fill="none"/>
  <text x="520" y="215" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">top-p=0.9 → 很多 tokens（自适应！）</text>
  <!-- Top-k bracket -->
  <path d="M 395 225 L 395 230 L 605 230 L 605 225" stroke="#ff6b6b" stroke-width="1.5" fill="none"/>
  <text x="500" y="245" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">top-k=50 → 可能截断合理选项</text>
  <!-- Bottom annotation -->
  <text x="350" y="290" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Top-p 根据分布形状自动调节核的大小；Top-k 不管分布怎样都是固定 k 个</text>
</svg>

## Min-p：让阈值跟着信心走

### Top-p 在高温下的困境

2024 年，一群研究者（Nguyen 等人）注意到 top-p 在高温场景下的弱点，并提出了一种优雅的替代方案：**min-p 采样**。

问题的根源是什么？Top-p 看的是"累积概率达到多少"，但它不关心最高概率 token 本身有多高。考虑两个场景：

- 场景 A：最高概率 token 有 80% 的概率。Top-p=0.9 只需要再加一两个 token 就达标。
- 场景 B：高温下概率被拉平，最高概率 token 只有 5%。Top-p=0.9 需要纳入大量 token。

场景 B 是问题所在——当模型本身就不确定时，top-p 会放进太多低质量选项。

### Min-p 的核心想法

Min-p 换了一个完全不同的视角来做截断：

> **只保留概率 ≥ p × P_max 的 token。**

其中 P_max 是当前步最高概率 token 的概率。

翻译成人话：如果排名第一的 token 有 60% 的概率，min-p=0.1 意味着只保留概率 ≥ 6%（= 0.1 × 60%）的 token。如果排名第一的只有 5%，那阈值是 0.5%——自动放宽。

### 为什么这比 top-p 更好？

Min-p 的阈值是**相对于模型信心的**：

- 模型很确定时（P_max 高）→ 阈值高 → 砍掉更多低概率 token → 自动保守
- 模型不确定时（P_max 低）→ 阈值低 → 保留更多选项 → 自动开放

这和 top-p 的自适应方向相似，但在高温下表现更稳健。因为 top-p 在高温下被"骗"了（所有概率都被拉平，累积到 0.9 需要太多 token），而 min-p 用最大概率作为锚点，即使在高温下也能合理截断。

### 实验验证

Min-p 论文在 GPQA（研究生级推理）、GSM8K（数学）、AlpacaEval（创意写作）等基准上测试，结果表明：

- 在标准温度下，min-p 与 top-p 表现相当
- 在高温（T=1.5-2.0）下，min-p 显著优于 top-p，既保持连贯性又允许更高多样性
- 人类评估者在质量和创意两个维度都明显偏好 min-p

这解释了为什么 min-p 迅速被 HuggingFace Transformers、vLLM、llama.cpp 等主流框架采纳。

## 更深的视角：信息论怎么看采样

### 人类语言的信息率

信息论给了我们一个理解采样策略的更深层框架。Shannon 的一个核心概念是**典型集（typical set）**：当我们从一个随机过程生成足够长的序列时，"看起来合理"的序列并不是概率最高的那些——而是信息含量接近分布熵的那些。

2022 年，Meister 等人的 "Locally Typical Sampling" 论文把这个想法应用到了文本生成中：

> 人类生成的文本，其每个词的信息含量（surprisal）接近于当前上下文的条件熵。太可预测的词（信息量过低）显得呆板；太出人意料的词（信息量过高）显得胡说。

这给了我们一个统一理解各种采样策略的框架：

- **贪心解码**：总是选信息量最低的 token → 退化为重复
- **纯随机采样**：可能选到信息量极高的 token → 胡言乱语
- **Top-k/Top-p/Min-p**：砍掉信息量过高的尾巴，保留"信息量适中"的范围
- **Typical sampling**：直接基于信息论原理，保留 surprisal 接近条件熵的 token

### Mirostat：用控制论来采样

2021 年，Basu 等人提出了一种完全不同思路的方法——**Mirostat**。它不是在每一步做截断，而是用**反馈控制**来维持整个序列的 perplexity 在目标值附近。

想法来自一个观察：序列的 perplexity（困惑度）直接关联着重复程度。Perplexity 太低→文本退化、重复；太高→文本混乱。如果能把 perplexity 稳定在某个"甜蜜点"，就能避免两个极端。

Mirostat 用一个自适应的 k 值（保留多少 token）作为控制变量，根据当前序列的 cross-entropy 和目标 perplexity 的差距来动态调节。这本质上是一个 PID 控制器应用在语言生成中。

### EDT：让温度也动起来

2024 年的 EDT（Entropy-based Dynamic Temperature）更进一步——它不用固定温度，而是**根据每步分布的熵动态选择温度**。

逻辑很直观：如果模型在某一步已经很确定（熵低），用低温度锁定；如果很不确定（熵高），用高温度增加多样性。相当于温度本身也变成了自适应的。

## 实践中的参数组合

### 各家 API 的默认配置

实际部署中，这些方法通常组合使用：

| 平台 | 默认配置 |
|------|---------|
| OpenAI (GPT-4) | temperature=1, top_p=1 (实际用 nucleus sampling) |
| Anthropic (Claude) | temperature=1, top_p=0.999 |
| 开源 (vLLM/TGI) | temperature=0.7, top_p=0.9 |
| llama.cpp | temperature=0.8, top_k=40, top_p=0.95, min_p=0.05 |

注意 llama.cpp 的配置——它**同时使用四种策略**，按照 temperature → top_k → top_p → min_p 的顺序依次应用。这种"层叠"方式在开源社区很常见。

### 什么时候用什么？

- **代码生成 / 数学 / 事实问答**：temperature=0（或极低），top_p 不重要——你要的是确定性
- **通用对话**：temperature=0.7-1.0, top_p=0.9 是安全的默认值
- **创意写作 / 头脑风暴**：temperature=1.0-1.5, min_p=0.05-0.1（比 top_p 在高温下更稳定）
- **角色扮演 / 开放式探索**：temperature=1.5-2.0, min_p=0.02-0.05

### Repetition Penalty：另一个维度

除了截断策略，还有一类参数直接惩罚重复：

- **Frequency penalty**：已出现的 token 每多出现一次，logit 减去一个固定值。出现越多惩罚越重。
- **Presence penalty**：只要 token 出现过一次，logit 就减去固定值（不管出现几次）。鼓励话题多样性。
- **Repetition penalty**：出现过的 token 的 logit 除以一个 > 1 的系数。

这些是在 logit 层面（采样前）操作的，和 top-k/top-p 这些截断策略正交——可以同时使用。

## 从理论到直觉：一个统一的视角

回到最开头的问题：为什么我们需要这么多种采样策略？

本质上，它们都是在回答同一个问题：**给定模型输出的概率分布，如何定义"合理范围"？**

| 方法 | "合理"的定义 |
|------|-------------|
| Greedy | 只有最高概率的是合理的 |
| Top-k | 前 k 名是合理的 |
| Top-p | 累计概率达到 p 的那些是合理的 |
| Min-p | 概率 ≥ 最大概率 × p 的是合理的 |
| Typical | 信息含量接近条件熵的是合理的 |
| Mirostat | 能维持目标 perplexity 的是合理的 |

每种方法背后都有不同的"什么是好文本"的哲学：
- Top-k/Top-p 是**概率视角**：高概率 = 合理
- Typical sampling 是**信息论视角**：典型信息率 = 合理
- Mirostat 是**控制论视角**：稳定的复杂度 = 合理
- Min-p 是**相对置信度视角**：相对于最佳选项足够好 = 合理

没有一种方法在所有场景下都最优。但一个越来越清晰的趋势是：**自适应方法（min-p、typical、mirostat）比固定阈值方法（top-k）更鲁棒**，因为语言本身的不确定性是变化的——有的位置几乎确定（语法词、专有名词），有的位置高度开放（创意表达、同义词选择）。

## 这意味着什么

理解采样策略之后，你能更好地：

1. **调参不再盲目**：知道了 temperature 是调熵的旋钮，top-p 是自适应截断，min-p 是相对截断——你就知道哪个参数该往哪个方向调来解决你的问题。
2. **理解模型"性格"差异**：不同模型用不同默认采样参数，这直接影响你对它"风格"的感知。一个 temperature=0.3 的模型当然比 temperature=1.0 的更"严谨"。
3. **理解为什么同一模型输出不一致**：只要 temperature > 0，输出就是随机的。"不一致"不是 bug，是采样的本质特征。
4. **理解研究前沿**：从 top-k 到 top-p 到 min-p 到 typical 到 mirostat，这条进化线展示了人们对"什么是好的语言生成"的理解不断深化。

采样看似只是工程上的"最后一步"，但它直接决定了用户感知到的模型质量。一个好模型配一个坏的采样策略，效果可能还不如一个一般模型配一个好的采样策略。这就是为什么 min-p 这样一个"小"改动，能被所有主流框架迅速采纳——它实实在在地改善了用户体验。

---

*参考文献：*
- Holtzman et al. "The Curious Case of Neural Text Degeneration" (ICLR 2020) — 提出 nucleus sampling (top-p)
- Fan et al. "Hierarchical Neural Story Generation" (ACL 2018) — 提出 top-k sampling
- Nguyen et al. "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs" (2024) — 提出 min-p
- Meister et al. "Locally Typical Sampling" (TACL 2023) — 信息论视角的典型采样
- Basu et al. "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity" (ICLR 2021)
- EDT: "Improving Large Language Models' Generation by Entropy-based Dynamic Temperature Sampling" (2024)
