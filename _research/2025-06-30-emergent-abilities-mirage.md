---
title: "涌现能力是真的吗？一场关于度量标准的科学论战"
date: 2025-06-30
level: 3
series: "LLM 原理深度解析"
series_order: 25
series_total: 39
tags: [涌现, emergence, scaling, 度量标准, BIG-Bench, 相变]
summary: "LLM 的涌现能力究竟是模型规模增长带来的真实突变，还是度量标准选择制造的幻觉？这场论战揭示了 AI 研究中最深刻的测量哲学问题。"
---

# 涌现能力是真的吗？一场关于度量标准的科学论战

> 当 AI 模型突然学会了之前完全做不到的事情，我们应该惊叹于智能的诞生，还是反思自己的温度计是不是坏了？

## 故事从一张图开始

2022 年，Google Research 的 Jason Wei 等人发表了一篇让整个 AI 圈震动的论文。他们画了一张图：横轴是模型的参数量（从 10 亿到 1000 亿），纵轴是某些任务的准确率。图上的曲线令人不安——在某个临界规模之前，模型在这些任务上的表现和随机猜测没有区别，完全是零分。但一旦跨过那个门槛，性能突然跳升，仿佛一夜之间"开窍"了。

Wei 等人给这个现象起了一个让人浮想联翩的名字：**涌现能力（Emergent Abilities）**。

这个词一出，AI 安全研究者紧张了——如果模型能突然获得无法预测的新能力，那我们怎么保证下一次"涌现"不是危险的？投资人兴奋了——这意味着只要继续砸钱扩大模型，就会有惊喜。而科学家们则分裂了：一些人看到了通向通用人工智能的阶梯，另一些人闻到了统计幻觉的味道。

一年后，Stanford 的一篇论文劈头盖脸地说：**你们看到的"涌现"，可能只是度量标准选错了而已。**

这就是 AI 领域近年来最精彩的一场科学论战。今天，我们来彻底搞懂这场辩论的两面。

## 第一幕："涌现"最初是怎么被发现的

### 物理学家眼中的涌现

在 AI 之前，"涌现"是物理学和复杂系统科学中的经典概念。水分子单独看只有质量和电荷，但当足够多的水分子聚在一起，就出现了波浪、漩涡、表面张力——这些性质在单个分子层面根本不存在。这就是涌现：**整体展现出部分所不具备的性质。**

更戏剧性的例子是相变：水加热到 99°C 还是液态，到 100°C 突然变成气态。温度只多了 1 度，但系统的行为发生了质的飞跃。物理学家管这叫**相变（Phase Transition）**——系统在某个临界点发生突变。

Wei 等人正是借用了这个类比：当语言模型的规模跨过某个临界点时，它突然获得了之前完全没有的能力。

### BIG-Bench 上的"惊奇时刻"

Wei 等人的证据来自 BIG-Bench——一个包含 200 多个任务的大型评测套件。他们观察到，在许多任务上：

- GPT-3 的 1.3B 参数版本：准确率接近 0%
- GPT-3 的 6.7B 参数版本：准确率仍然接近 0%
- GPT-3 的 175B 参数版本：准确率突然跳到 40-60%

三位数加法、多步逻辑推理、国际音标转录……这些任务都展现出了同样的模式：小模型完全做不到，大模型突然就会了。中间没有"半会"的状态。

他们正式定义：**如果一种能力在小模型中完全不存在，但在大模型中突然出现，并且无法通过小模型的表现来预测，那么这种能力就是"涌现的"。**

这个定义有两个关键特征：
1. **突变性（Sharpness）**：不是渐进改善，而是从零到有的跳跃
2. **不可预测性（Unpredictability）**：看小模型的趋势线，你猜不到大模型会突然在某个任务上表现良好

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="80" y1="260" x2="650" y2="260" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="80" y1="260" x2="80" y2="30" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Axis labels -->
  <text x="370" y="295" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">模型参数量 (log scale)</text>
  <text x="30" y="150" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" transform="rotate(-90, 30, 150)">准确率 (%)</text>
  <!-- Scale marks -->
  <text x="130" y="278" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">1B</text>
  <text x="250" y="278" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">10B</text>
  <text x="400" y="278" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">100B</text>
  <text x="550" y="278" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">1000B</text>
  <!-- Y marks -->
  <text x="70" y="260" text-anchor="end" fill="#94a3b8" font-size="11" font-family="system-ui">0%</text>
  <text x="70" y="150" text-anchor="end" fill="#94a3b8" font-size="11" font-family="system-ui">50%</text>
  <text x="70" y="50" text-anchor="end" fill="#94a3b8" font-size="11" font-family="system-ui">100%</text>
  <!-- Dashed grid -->
  <line x1="80" y1="150" x2="640" y2="150" stroke="#3a3a4a" stroke-width="0.5" stroke-dasharray="4,4"/>
  <!-- "Emergent" curve - flat then sudden jump -->
  <path d="M 100 255 L 180 254 L 260 253 L 320 250 L 360 245 L 380 220 L 400 150 L 430 100 L 500 70 L 580 60" fill="none" stroke="#f97316" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Random baseline -->
  <line x1="100" y1="255" x2="380" y2="255" stroke="#ef4444" stroke-width="1" stroke-dasharray="6,4"/>
  <text x="240" y="248" fill="#ef4444" font-size="11" font-family="system-ui">随机猜测水平</text>
  <!-- Critical point annotation -->
  <line x1="380" y1="220" x2="380" y2="30" stroke="#22d3ee" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="380" y="22" text-anchor="middle" fill="#22d3ee" font-size="12" font-family="system-ui">临界规模</text>
  <!-- Arrow pointing to jump -->
  <text x="470" y="85" fill="#f97316" font-size="12" font-family="system-ui">突然"涌现"！</text>
  <!-- Legend -->
  <rect x="480" y="230" width="160" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <line x1="490" y1="245" x2="520" y2="245" stroke="#f97316" stroke-width="2.5"/>
  <text x="530" y="249" fill="#ededf0" font-size="11" font-family="system-ui">Exact-Match 准确率</text>
</svg>

## 第二幕：Stanford 的"海市蜃楼"反击

### 问题出在哪里？

2023 年 4 月，Stanford 的 Rylan Schaeffer、Brando Miranda 和 Sanmi Koyejo 发表了论文《Are Emergent Abilities of Large Language Models a Mirage?》，直指问题核心：

**涌现可能不是模型的性质，而是你选了什么尺子去量。**

他们的论证出奇地简洁优美。核心观察是这样的：

想象你在教一个小孩做三位数加法，比如 `123 + 456 = 579`。这个答案有三个数字都要对。如果小孩算出了 `578`（只错了最后一位），在"精确匹配"评分标准下，他的得分和写了 `000` 的小孩一样——都是 **0 分**。

精确匹配（Exact Match）是一个典型的**非线性、不连续**度量。它只关心"全对还是没全对"。一道三位数加法题，三个数字只对了两个 = 0 分。三个都对 = 1 分。中间没有"半分"。

现在想象模型在逐渐变大的过程中的真实表现：
- 10B 参数：平均每道题答对 0.5 个数字（纯猜）
- 50B 参数：平均答对 1.8 个数字
- 100B 参数：平均答对 2.5 个数字
- 500B 参数：平均答对 2.95 个数字

如果用**逐字符准确率**来衡量，你会看到一条平滑上升的曲线——模型在稳步进步，每一步都比上一步好一点。

但如果用**精确匹配**来衡量，你看到的是：前三个模型全是约 0%（因为很少能三个数字全对），最后一个突然跳到 90%+。"涌现"出现了！

### Schaeffer 的核心论点

Schaeffer 等人把这个观察抽象为一个简洁的数学模型：

**假设模型在某个子任务上的真实能力（per-token probability）是随规模平滑增长的**。那么：

- **如果你用线性/连续度量**（比如每个 token 的准确率、编辑距离、Brier 分数）来测量，你看到的就是平滑增长——没有涌现。
- **如果你用非线性/不连续度量**（比如精确匹配、多步全对才得分）来测量，由于答案需要多个 token 都正确，而每个 token 正确的概率都在缓慢上升，它们的乘积会呈现出 S 型曲线——在某个临界点附近急剧从 0 跳到 1。

用概率的语言说：如果一个三位数答案的每位数字独立地有概率 $p$ 答对，那精确匹配的概率是 $p^3$。当 $p$ 从 0.7 增长到 0.95 时，$p^3$ 从 0.34 飙升到 0.86。$p$ 变化了 36%，$p^3$ 变化了 153%。这就是"涌现"的数学魔术——**指数化放大了渐进改善，制造出了阶跃假象。**

### 实验验证

Schaeffer 团队在 BIG-Bench 上做了系统验证：

1. **选择声称存在涌现的任务**
2. **保持模型输出不变**，只改变评分标准
3. 把"精确匹配"换成"Token-level 准确率"或"编辑距离"

结果立竿见影：那些之前表现出"突然涌现"的任务，在连续度量下全部变成了平滑增长曲线。模型一直在进步，只是之前的尺子看不见而已。

他们还做了反向验证：拿那些本来看起来是平滑进步的任务，故意换成非线性度量来评估——"涌现"立刻被人为制造出来了。

<svg viewBox="0 0 700 340" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">同一个模型，同一组输出，两种度量标准</text>
  <!-- Left panel -->
  <rect x="30" y="40" width="300" height="270" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="180" y="65" text-anchor="middle" fill="#ef4444" font-size="12" font-weight="bold" font-family="system-ui">❌ 精确匹配 (Exact Match)</text>
  <!-- Left axes -->
  <line x1="70" y1="270" x2="300" y2="270" stroke="#6e8eff" stroke-width="1"/>
  <line x1="70" y1="270" x2="70" y2="85" stroke="#6e8eff" stroke-width="1"/>
  <!-- Left curve - S-shaped / step -->
  <path d="M 85 265 L 120 264 L 150 263 L 180 260 L 200 240 L 215 170 L 230 110 L 260 95 L 290 90" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round"/>
  <text x="180" y="290" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">模型规模 →</text>
  <text x="150" y="120" fill="#ef4444" font-size="11" font-family="system-ui">看起来：突然涌现！</text>
  <!-- Right panel -->
  <rect x="370" y="40" width="300" height="270" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="520" y="65" text-anchor="middle" fill="#34d399" font-size="12" font-weight="bold" font-family="system-ui">✅ Token 级准确率</text>
  <!-- Right axes -->
  <line x1="410" y1="270" x2="640" y2="270" stroke="#6e8eff" stroke-width="1"/>
  <line x1="410" y1="270" x2="410" y2="85" stroke="#6e8eff" stroke-width="1"/>
  <!-- Right curve - smooth linear -->
  <path d="M 425 260 L 460 245 L 500 220 L 540 185 L 580 145 L 620 110 L 640 100" fill="none" stroke="#34d399" stroke-width="2.5" stroke-linecap="round"/>
  <text x="520" y="290" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">模型规模 →</text>
  <text x="530" y="120" fill="#34d399" font-size="11" font-family="system-ui">真相：一直在平滑进步</text>
  <!-- Center arrow -->
  <text x="350" y="180" text-anchor="middle" fill="#a78bfa" font-size="22" font-family="system-ui">⟺</text>
  <text x="350" y="200" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">换个尺子</text>
</svg>

## 第三幕：但是……这真的能解释一切吗？

### 来自 NeurIPS 2024 的新证据

Schaeffer 的论文确实解释了很多"伪涌现"。但故事没有到此结束。

2024 年，清华大学和智谱 AI 的 Du Zhengxiao 等人在 NeurIPS 2024 上发表了一篇关键论文：《Understanding Emergent Abilities of Language Models from the Loss Perspective》。他们的发现为这场辩论增加了新的维度：

**核心发现：不管用什么度量（连续的还是不连续的），某些任务确实存在一个预训练损失的门槛。在这个门槛之上，模型表现等同于随机猜测；在门槛之下，能力突然出现。**

这个发现的精妙之处在于：

1. 他们不看模型的参数量，而是看**预训练损失（pre-training loss）**
2. 他们发现，不同大小的模型，只要预训练损失相同，在下游任务上的表现就相同
3. 对于某些任务，即使用连续度量来评估，在损失越过某个阈值时仍然存在明显的跳跃

这意味着：**Schaeffer 说对了一半——精确匹配确实会制造虚假的涌现幻象。但 Schaeffer 没有完全对——某些任务上确实存在真实的能力阈值，只是这个阈值应该用预训练损失（而非参数量）来定义。**

### 为什么预训练损失是更好的标尺？

参数量是一个粗糙的代理变量。一个 7B 模型训练得足够久、数据足够好，可能比一个粗糙训练的 70B 模型表现更好。预训练损失直接反映了"模型到底学到了多少"，是更本质的指标。

Du 等人的框架可以这样理解：

想象一个任务是"三位数加法"。要正确完成这个任务，模型至少需要：
- 理解数字的位值概念
- 掌握个位进位规则
- 能同时跟踪多步计算

这些子能力各自在预训练过程中逐渐习得。但要完成整个任务，需要**所有子能力同时就位**。这就像一条铁链——只要有一环没准备好，整条链就断了。当预训练损失降到某个阈值以下时，最后一环"咔哒"一声扣上了，整个能力突然出现。

这种"门槛效应"即使用连续度量也能观察到，因为它不是度量标准的幻觉，而是**任务本身结构的产物**。

### 两种涌现

综合 2022-2025 年的研究，当前学界的理解可以总结为：

**第一类"涌现"——度量标准制造的幻觉（Metric-Induced Mirage）：**
- 本质上是连续进步被不连续度量"二值化"了
- 换成连续度量就消失
- 对应 Schaeffer 的发现
- **这不是真正的涌现**

**第二类涌现——任务结构驱动的阈值效应（Task-Structure Threshold）：**
- 某些复杂任务需要多个子能力同时就位
- 即使用连续度量，仍存在明确的损失阈值
- 对应 Du 等人的发现
- **这可能是更接近"真实涌现"的东西**

但要注意——即使是第二类，也不是完全不可预测的。一旦你知道预训练损失和任务的关系，你就可以预测"模型训到什么程度会获得这个能力"。它是突然的，但不是神秘的。

## 第四幕：这场辩论真正在争的是什么？

### 定义之争

仔细看就会发现，两方其实在使用不同的"涌现"定义：

**Wei 等人的定义（操作性定义）**：如果能力在小模型中完全不存在，在大模型中突然出现，就是涌现的。

**Schaeffer 的反驳**：这个定义的"不存在"取决于你怎么测量"存在"。如果你的温度计只能显示整数，水从 99°C 到 100°C 看起来就是突然变热了一度；换个精确到小数的温度计，你会看到连续变化。

**Du 等人的重新定义**：当模型的预训练损失降到某个阈值以下时，某些能力从随机水平突然跳升——不管用什么度量。

这本质上是一个**科学哲学问题**：什么算"新的能力出现"？如果模型从"三位数加法答对率 0%"变成"答对率 80%"，不管中间的渐进过程是什么，这种质的飞跃算不算涌现？

### 为什么这个问题重要

这不只是学术争论。不同的答案有完全不同的实际后果：

**如果涌现是真的：**
- AI 安全需要极度谨慎——下一个模型可能突然获得危险能力
- 投资逻辑成立——继续扩大规模，就会有惊喜
- 对齐研究需要应对"无法预测的新能力"

**如果涌现是幻觉：**
- AI 发展是可预测的——我们只需要更好的度量工具
- 不存在"突然变危险"的风险——一切都是渐进的
- 资源分配可以更理性——不需要"赌一把看会不会出奇迹"

**当前共识（如果有的话）**：

真相在中间。大多数"涌现"确实是度量标准的幻觉，但某些复杂任务确实存在能力阈值。关键区别在于：
- ❌ 涌现不是神秘的、不可预测的魔法
- ✅ 涌现是可解释的阈值效应：复杂任务需要多个子能力同时就位
- ✅ 用预训练损失而非参数量来追踪，涌现变得可预测

### 一个更精确的类比

与其说涌现像物理中的相变（水→蒸汽），不如说更像**学开车**：

你学了方向盘，又学了油门，又学了看后视镜，又学了判断车距。每一项都在缓慢进步。但"能独立上路"这个能力，需要所有子技能同时达到某个最低水平。在此之前，你完全不能安全驾驶（0 分）。一旦全部到位，你突然就"会开车了"。

如果有人只看"能否独立上路"这个二元指标，他会以为你某一天突然开窍了。但如果分别追踪每个子技能，他会看到一切都是渐进的。

**模型的"涌现"也是如此：子能力连续增长 + 任务需要子能力同时就位 = 看起来突然获得新能力。**

## 这意味着什么

这场辩论给我们三个重要教训：

**1. 测量方式决定了你看到什么**

这不只是 AI 的问题。在任何科学领域，选择什么度量标准，直接决定了你会"发现"什么规律。Schaeffer 的论文是一篇出色的方法论警示——当你看到令人兴奋的结果时，先问一句：换个尺子量，结论还成立吗？

**2. 可预测性比"是否涌现"更重要**

对于 AI 安全和工程实践来说，真正关键的问题不是"涌现是不是真的"，而是"我能不能提前知道模型什么时候会获得某种能力"。Du 等人的预训练损失框架给出了一个有希望的方向——即使能力确实是突然出现的，它也可以是可预测的。

**3. 简单问题可能有复杂答案**

"涌现是真的吗？"这个看似非此即彼的问题，最终答案是"取决于你说的'涌现'是什么意思，以及你怎么测量"。这提醒我们：在讨论 AI 能力时，精确的定义和严格的方法比耸人听闻的叙事更重要。

## 关键论文时间线

| 时间 | 论文 | 核心观点 |
|------|------|---------|
| 2022.06 | Wei et al., "Emergent Abilities of Large Language Models" | 首次系统定义并展示 LLM 涌现能力 |
| 2023.04 | Schaeffer et al., "Are Emergent Abilities a Mirage?" | 涌现是非线性度量标准制造的幻觉 |
| 2023.09 | Lu et al., "Are Emergent Abilities just In-Context Learning?" | 涌现可能本质上是 ICL 能力的体现 |
| 2024.03 | Du et al., "Understanding Emergent Abilities from the Loss Perspective" | 预训练损失阈值才是涌现的本质驱动力 |
| 2025.02 | Ferrara et al., "Emergent Abilities in LLMs: A Survey" | 综合调查：两种涌现分类框架 |

---

*这是「LLM 原理深度解析」系列的第 25 篇。在这个系列中，我们试图把 LLM 背后的每一个核心概念讲透——不只是让你知道名字，而是真正理解它为什么是这样的。*
